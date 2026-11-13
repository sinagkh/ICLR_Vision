#!/usr/bin/env python3
"""Refit the paper's DFR(Val) heads and fold them into a ResNet-50 checkpoint for eval_robustness.py.

dfr_official.py saves only metrics. This script repeats its steps exactly (same checkpoints, same
feature extraction, np.random.seed(seed), official dfr_on_validation_tune, then the official dfr_on_validation_eval) and
additionally keeps the averaged head. The official eval function is executed from its own source with one change: its
return statement also returns the scaler and the averaged LogisticRegression. The refit counts only if its test WGA
matches the value saved by dfr_official.py.

Folding: logits = [0, w·(x−μ)/σ + b] → fc.weight[1] = w/σ, fc.bias[1] = b − (w/σ)·μ, row 0 = 0. The class-1 margin
equals the DFR decision function, so eval_robustness.py margins and predictions are DFR's.
usage: dfr_refit_fold.py SEED BASE(erm|full_is)
"""
import ast, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dfr_official as D  # noqa: E402  (load_official_namespace, extract)
from common import CKPT as CKPT_ROOT, OUT as OUT_ROOT  # noqa: E402
from torchvision import models  # noqa: E402

CKPT = {'erm': CKPT_ROOT / 'erm/seed{s}/checkpoints/erm_best.pt',
        'full_is': CKPT_ROOT / 'full_is/seed{s}/checkpoints/steered_finetuned_best.pt'}
SAVED = OUT_ROOT / 'dfr/seed{s}/{b}.json'
OUT = OUT_ROOT / 'dfr_fold'


def official_eval_with_head(ns):
    src = D.OFFICIAL.read_text(); tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'dfr_on_validation_eval')
    code = ast.get_source_segment(src, fn)
    old = 'return test_accs, test_mean_acc, train_accs'
    assert code.count(old) == 1
    code = code.replace(old, old + ', (scaler if preprocess else None), logreg')
    g = dict(ns); exec(code, g)
    return g['dfr_on_validation_eval']


def main():
    seed, base = int(sys.argv[1]), sys.argv[2]
    ns = D.load_official_namespace(); ev = official_eval_with_head(ns)
    dev = torch.device('cuda')
    path = str(CKPT[base]).format(s=seed)
    model = models.resnet50(weights=None, num_classes=2).to(dev)  # same architecture as build_backbone('resnet50'); all weights come from the checkpoint
    ck = torch.load(path, map_location=dev, weights_only=False); model.load_state_dict(ck.get('model', ck)); model.eval()
    emb, ys, gs = {}, {}, {}
    for split, key in [('train', 'train'), ('validation', 'val'), ('test', 'test')]:
        emb[key], _, ys[key], gs[key] = D.extract(model, 'resnet50', split, dev, 128, 16)
    np.random.seed(seed)
    c, w1, w2 = ns['dfr_on_validation_tune'](emb, ys, gs, balance_val=True, add_train=False)
    accs, mean_acc, _, scaler, logreg = ev(c, w1, w2, emb, ys, gs, balance_val=True, add_train=False)
    saved = json.load(open(str(SAVED).format(s=seed, b=base)))['dfr_val']
    wga = float(min(accs)); match = abs(wga - saved['test_wga']) < 1e-9
    print(f'[refit] s{seed} {base} C={c} w=({w1},{w2}) WGA {wga:.4f} (saved {saved["test_wga"]:.4f}) match={match}', flush=True)
    # fold the standardized binary head into a 2-class fc
    w = logreg.coef_.reshape(-1).astype(np.float64); b = float(logreg.intercept_.reshape(-1)[0])
    mu, sd = scaler.mean_.astype(np.float64), scaler.scale_.astype(np.float64)
    W = np.zeros((2, w.size)); W[1] = w / sd; B = np.zeros(2); B[1] = b - float((w / sd) @ mu)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    state['fc.weight'] = torch.tensor(W, dtype=state['fc.weight'].dtype); state['fc.bias'] = torch.tensor(B, dtype=state['fc.bias'].dtype)
    # check: folded fc on raw test features reproduces DFR's predictions
    z = emb['test'] @ W[1] + B[1]; pred_fold = (z > 0).astype(int)
    pred_dfr = logreg.predict(scaler.transform(emb['test']))
    agree = float((pred_fold == pred_dfr).mean())
    (OUT / 'ckpt').mkdir(parents=True, exist_ok=True)
    out = OUT / 'ckpt' / f'{base}_dfrval_s{seed}.pt'
    torch.save({'model': state, 'source_checkpoint': path, 'dfr': {'C': c, 'w1': w1, 'w2': w2, 'test_wga': wga, 'saved_test_wga': saved['test_wga'],
                'fold_agreement': agree}}, out)
    json.dump({'seed': seed, 'base': base, 'C': c, 'w1': w1, 'w2': w2, 'test_group_accs': [float(a) for a in accs], 'test_wga': wga,
               'test_mean_acc': float(mean_acc), 'saved_test_wga': saved['test_wga'], 'match': match, 'fold_agreement': agree},
              open(OUT / f'refit_{base}_s{seed}.json', 'w'), indent=1)
    print(f'[fold] {out} agreement with DFR predictions on test: {agree:.6f}', flush=True)


if __name__ == '__main__':
    main()
