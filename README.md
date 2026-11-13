# Interaction Steering

Code, run commands and results for Interaction Steering (IS) on Waterbirds, UrbanCars, PACO and Colored MNIST.
Datasets and model checkpoints are not included.

| Folder | Contents |
|---|---|
| `waterbirds/` | code; `configs/01_*.sh` to `20_*.sh` run every Waterbirds experiment in order |
| `urbancars/` | dataset generation, code; `configs/*.sh` run every UrbanCars experiment |
| `paco/` | data lists, code; `configs/commands.sh` and `configs/mae_vit.sh` run every PACO experiment |
| `colored_mnist/` | code; `configs/commands.sh` runs the Colored MNIST experiment |
| `results/` | the reported numbers as CSV files, one row per seed plus the mean |

## Setup

Python 3.10 and one 24 GB GPU. Install the requirements of the dataset you run (`<dataset>/requirements.txt`).
Every script reads its locations from environment variables:

```bash
export DATA_ROOT=/path/to/data          # datasets
export CKPT_ROOT=/path/to/checkpoints   # trained models
export OUT_ROOT=/path/to/outputs        # evaluation outputs
```

## Waterbirds

**Data** (under `$DATA_ROOT`):

- `hf_cache/`: `grodino/waterbirds` from the Hugging Face Hub. Download it once with
  `HF_DATASETS_CACHE=$DATA_ROOT/hf_cache python -c "from datasets import load_dataset; load_dataset('grodino/waterbirds')"`.
- `waterbird_complete95_forest2water2/`: the official Waterbirds images and `metadata.csv` (Sagawa et al., 2020).
- `segmentations/`: CUB-200-2011 bird segmentations.
- `places365/val_256/`: Places365 validation images (unseen-scene test only).
- `waterbirds_manifest/`: built from the three folders above with `python waterbirds/data/build_manifest.py`; read by configs 02, 08, 11 and the manifest variant of 03.

Train and evaluate in an environment without `opencv-python`. Only `data/masks_cam_grabcut.py` (config 12) needs OpenCV and scikit-image (listed, commented out, in `waterbirds/requirements.txt`).

**Run** the configs in order (`bash waterbirds/configs/01_erm_resnet50.sh`, ...). Seeds 1, 21, 42; checkpoints are selected by validation worst-group accuracy (WGA).

| Config | Experiment |
|---|---|
| 01 | ERM (ResNet-50) |
| 02 | ERM on the manifest and its anchor-bank audit (the banks are already in `waterbirds/lists/`) |
| 03 | IS; at the end, the manifest variant used for the paper's ResNet-50 IS results |
| 04 | IS without target |
| 05 | tolerance τ and guard weights ×0.5 / ×2 |
| 06 | placebo masks (seed 1) |
| 07 | CF-Aug and CF-Aug + IS on the same edits for the last 20 epochs; CF-Aug with the stand-alone trainer `train/train_cfaug.py` (unseen-scene test) |
| 08 | Full CF-Aug (edits for all 50 epochs) |
| 09 | GroupDRO and GroupDRO + IS |
| 10 | CDEP |
| 11 | DINOv2 ViT-B/14: ERM and IS (test results in each run's `train_summary.json`) |
| 12 | IS with Mask R-CNN masks (test accuracy and common-corner audit in `$OUT_ROOT`); CAM + GrabCut masks for the training split |
| 13 | DaC (official code, commit `8f41e1e`, path in `$DAC_REPO`) |
| 14 | evaluation of all ResNet-50 models: accuracy, Corrupted BG-Gap, Only-BG accuracy, background Δ¹, unseen scenes |
| 15 | DFR on the ERM and IS backbones (official DFR code, commit `9d8d5de`, path in `$DFR_OFFICIAL_SCRIPT`) |
| 16 | writes the CSV files in `results/waterbirds/` |
| 17 | Focal Loss, CVaR, Spectral Decoupling and JTT (ResNet-50) |
| 18 | Focal Loss, CVaR, Spectral Decoupling, JTT and GroupDRO (DINOv2 ViT-B/14) |
| 19 | ExMap (ResNet-50): an ERM model per seed, then LRP pseudo-groups and last-layer retraining (needs `zennit`) |
| 20 | writes `paper_table10_11.csv` and `paper_table13_exmap.csv` in `results/waterbirds/` |

## UrbanCars

**Data.** UrbanCars is generated with the Whac-A-Mole generator and the patch in this package:

```bash
cd urbancars
git clone https://github.com/facebookresearch/Whac-A-Mole.git vendor/Whac-A-Mole
git -C vendor/Whac-A-Mole checkout c91b097043246bc726fccd70f9860873e28d6122
git -C vendor/Whac-A-Mole apply ../../data/whac_a_mole.patch
bash data/download_source_assets.sh   # Stanford Cars, LVIS, COCO 2017, Places365, MaskFormer
bash data/generate_car_masks.sh
bash data/generate_urbancars.sh
```

**Run** (seeds 0, 1, 2 unless noted):

| Config | Experiment |
|---|---|
| `erm_baseline.sh` | ERM, 300 epochs |
| `is_staged.sh` | IS with the staged schedule (paper Table 5) |
| `single_target.sh` | BG-only and CoObj-only IS |
| `third_order.sh` | IS with the third-order term |
| `cfaug_vs_cfaug_is.sh` | CF-Aug and CF-Aug + IS, 20 epochs (seeds 1, 2, 3) |
| `is_no_target_full.sh` | IS without target, full schedule |

Benchmark metrics are in each run's `history.jsonl` at the selected epoch (`train_summary.json`); `eval/stress_test.py` gives the Corrupted BG-Gap. The last two configs end by printing their tables (`eval/score_cfaug.py`, `eval/score_no_target.py`).

## PACO

**Data.** COCO 2017 images (`$COCO_ROOT`) and the PACO-LVIS v1 annotations (`$PACO_ROOT`). The instance lists, evaluation sets (built by `paco/data/build_eval_sets.py`), anchor banks and spoof pairs are in `paco/lists/`.

**Run** `bash paco/configs/commands.sh` from this folder. Every arm fine-tunes the seed-matched ERM checkpoint (seeds 0, 1, 2) for 12 epochs; place the ERM checkpoints at `$CKPT_ROOT/erm_seed{0,1,2}/best_by_val_top1.pt` (available on request). The last step writes `results/paco/`.
The PACO results in the author responses are new runs of this code from the paper's ERM checkpoints, so their values differ from the paper's PACO tables.
`bash paco/configs/mae_vit.sh` repeats plain fine-tuning, IS without the synergy target and IS on an MAE-pretrained ViT-B/16, starting from an ERM trained from the MAE weights.

## Colored MNIST

**Data.** MNIST is downloaded by torchvision to `$DATA_ROOT/MNIST` on first use. `colored_mnist/dataset.py` builds the colored sets: 25% label noise; label–color agreement ρ = 0.95 in training and in the IID test set, ρ = 0.05 in the OOD test set.

**Run** `bash colored_mnist/configs/commands.sh` from this folder (seed 7, under a minute per run on one GPU). It trains ERM (λ = 0) and Steered (λ = 0.5, 0.7, 0.9), writes the IID and OOD test accuracies to `$OUT_ROOT/colored_mnist/`, and writes the table to `results/colored_mnist/`. GPU training is not bit-for-bit deterministic, so retrained models can differ slightly from the reported ones.

## Results

| File | Contents |
|---|---|
| `waterbirds/per_seed.csv`, `mean_over_seeds.csv` | WGA, average accuracy, Corrupted BG-Gap, Only-BG accuracy, background Δ¹ and group accuracies of every ResNet-50 model; the IS and ERM rows are the models of configs 03 and 01 |
| `waterbirds/unseen_scenes.csv` | Corrupted BG-Gap with test birds on unseen Places365 scenes (ERM, IS, and CF-Aug with the stand-alone trainer) |
| `waterbirds/script_reported_test_accuracy.csv` | WGA and test accuracy of Full CF-Aug (its trainer's test evaluation) and of DFR (the official DFR evaluation) |
| `waterbirds/paper_table10_11.csv` | paper Tables 10 and 11 (ResNet-50, DINOv2 ViT-B/14): WGA, average accuracy, ΔWGA and group accuracies of ERM, Focal Loss, CVaR, Spectral Decoupling, JTT, Interaction Steering and GroupDRO; one row per seed, then mean and sample SD; the ERM and IS rows are the models of config 02 and of the manifest variant of 03; `checkpoint` names the file of each row |
| `waterbirds/paper_table13_exmap.csv` | paper Table 13: ERM, ExMap and Interaction Steering (ResNet-50), same columns |
| `waterbirds/predicted_masks.csv` | IS trained with Mask R-CNN bird masks (config 12): WGA, average accuracy and group accuracies per seed, mean and sample SD |
| `urbancars/main_table.csv` | ERM, BG-only IS, CoObj-only IS and IS (paper Table 15) |
| `urbancars/third_order.csv` | IS with and without the third-order term (paper Table 16) |
| `urbancars/cfaug_vs_cfaug_is_20ep.csv` | CF-Aug vs CF-Aug + IS |
| `urbancars/is_vs_is_without_target_full_schedule.csv` | IS vs IS without target |
| `paco/main_table.csv` | top-1, Spoof FPR95 and AUROC of IS, its controls and the baselines with and without IS |
| `paco/spoof_renderings.csv` | Spoof FPR95 with the original hard paste, a colour-harmonized paste with a feathered edge, and a hard paste shifted to a new position |
| `paco/sensitivity.csv`, `sensitivity_k_validation_pick.csv` | IS with one setting changed; K chosen per seed by validation |
| `paco/hidden_part.csv` | recognition of objects with a named part naturally hidden |
| `paco/mae_vit.csv` | top-1, Spoof FPR95 and AUROC on an MAE-pretrained ViT-B/16 |
| `colored_mnist/table9.csv` | IID and OOD test accuracy of ERM and Steered for each λ (paper Table 9; one run, seed 7) |

Measures (accuracies in %, gaps in percentage points):

- **WGA / average accuracy / ID accuracy**: worst-group, average and in-distribution test accuracy. **ΔWGA**: WGA minus the WGA of ERM with the same backbone and seed. Colored MNIST **IID / OOD accuracy**: test accuracy with label–color agreement ρ = 0.95 / ρ = 0.05. UrbanCars **BG, CoObj and BG+CoObj Gap**: accuracy on the groups where the background, the co-occurring object or both conflict, minus ID accuracy (Li et al., 2023).
- **Corrupted BG-Gap** (↓): accuracy where the background matches the class minus accuracy where it conflicts, with only the object corrupted. Waterbirds: mean over Gaussian blur (radius 8), ×16 downsampling and contrast 0.1. UrbanCars: the context is the background and the co-occurring object, mean over the eight degraded conditions of `urbancars/eval/stress_test.py`.
- **Only-BG accuracy** (↓): share of birdless test images (bird removed, or a conflicting donor background shown alone) classified as the background's class.
- **Background Δ¹** (↓): change of the true-class margin when the bird-free background is swapped for a conflicting one (logits); computed for ERM, IS, IS without target, GroupDRO, CDEP and DaC, and on seed 1 for the guard-weight and placebo-mask runs.
- **Spoof FPR95** (↓): share of single-part spoofs (one part of a test object pasted onto an object of an untrained category) accepted at the per-class threshold that keeps 95% of genuine test objects; **AUROC** (↑): the same separation over all thresholds.

## License

MIT (see `LICENSE`). Code adapted from other repositories keeps the license of its source: `urbancars/data/whac_a_mole.patch` (Whac-A-Mole) and the CDEP propagation rules in `waterbirds/train/train_cdep.py`.
