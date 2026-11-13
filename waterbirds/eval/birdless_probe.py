#!/usr/bin/env python3
"""Birdless / donor-background probe on the Waterbirds test set, built with eval/object_background_probes.py.

For every test image: the bird removed (inpainted), a conflicting donor background alone, and the bird pasted on that donor.
`prepare` fixes the donor of every test image (lists/birdless_probe_ids.json) and writes the rendered uint8 cache that
eval_robustness.py, guard_analysis.py and unseen_scenes.py read through `Cached` (run by configs/14_evaluation.sh, step 1).
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'data'))
from common import OUT, PKG, DATA_ROOT, digest, save, event, lookup_table
from waterbirds_data import GroundTruthMaskWaterbirds, configure_ground_truth_class

DEST = OUT / 'birdless_probe'
SOURCE = PKG / 'eval' / 'object_background_probes.py'
CHUNK = 8


def source_module():
    sys.path.insert(0,str(PKG/'train'))
    spec=importlib.util.spec_from_file_location('object_background_probes',SOURCE)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module._HELPERS.CV2=None
    module.HFWaterbirds=configure_ground_truth_class(
        GroundTruthMaskWaterbirds,DATA_ROOT/'hf_cache',module._HELPERS,
        DATA_ROOT/'waterbird_complete95_forest2water2',
        DATA_ROOT/'segmentations')
    return module


class Renders(Dataset):
    def __init__(self,module,base):
        self.removed=module.BackgroundOnlyNaturalDataset(base)
        self.pair=module.ConflictOverridePairDataset(base,seed=42)
    def __len__(self):return len(self.removed)
    def __getitem__(self,i):
        random_state,npstate=random.getstate(),np.random.get_state()
        try:
            s=42*1000003+i*1009
            random.seed(s+17);np.random.seed(s+17)
            removed,y,bg,idx=self.removed[i]
            random.seed(s+31);np.random.seed(s+31)
            donor,with_bird,pair_y,dbg,pidx=self.pair[i]
            assert y==pair_y and idx==pidx and dbg==1-y
            return torch.stack([removed,donor,with_bird]),idx
        finally:
            # Do not let worker scheduling alter another example's fill.
            random.setstate(random_state)
            np.random.set_state(npstate)


class Cached(Dataset):
    def __init__(self):
        self.x=np.load(DEST/'edited_images_uint8.npy',mmap_mode='r')
        self.lut=lookup_table()
    def __len__(self):return len(self.x)
    def __getitem__(self,i):
        encoded=torch.from_numpy(self.x[i].copy()).long()
        return self.lut[torch.arange(3)[None,:,None,None],encoded],i


def prepare(module,base):
    paths=[SOURCE,PKG/'train'/'train_cfaug.py',
           PKG/'data'/'waterbirds_data.py',Path(__file__)]
    render=Renders(module,base)
    ids=[]
    for i,row in enumerate(base.rows):
        assert base.has_mask(i) and module.mask_bbox(base.load_mask(i)) is not None
        y,bg=int(row['label']),int(row['background'])
        donor=render.pair.choose_donor_for_bg(1-y,i)
        assert donor is not None and donor!=i and base.has_mask(donor)
        assert int(base.rows[donor]['background'])==1-y
        ids.append({'idx':i,'y':y,'background':bg,'natural_conflict':y!=bg,
                    'donor_idx':donor,'donor_background':1-y})
    assert len(ids)==5794 and sum(a['natural_conflict'] for a in ids)==2897
    save(DEST/'probe_ids.json',ids)
    manifest={'utc':datetime.now(timezone.utc).isoformat(),'donor_selection_seed':42,
              'n':len(ids),'natural_conflict_n':2897,
              'files':{str(p):digest(p) for p in paths},'id_sha256':digest(DEST/'probe_ids.json'),
              'precision':'FP32, TF32 disabled','image_microbatch':CHUNK,
              'source':'eval/object_background_probes.py',
              'render_seed_formula':{'removed':'42*1000003+idx*1009+17','donor_pair':'42*1000003+idx*1009+31'},
              'gpu':torch.cuda.get_device_name(),'torch':torch.__version__}
    save(DEST/'manifest.json',manifest)
    event(action='probe_ids_fixed',output=str(DEST),id_sha256=manifest['id_sha256'])
    separate=module.ConflictBackgroundOnlyDataset(base,seed=42)
    validations=[]
    for i in range(8):
        s=42*1000003+i*1009+31
        random.seed(s);np.random.seed(s)
        expected=separate[i][0]
        actual=render[i][0][1]
        assert torch.equal(expected,actual)
        validations.append({'idx':i,'separate_vs_override_donor_exact':True})
    encoded=np.lib.format.open_memmap(DEST/'edited_images_uint8.npy',mode='w+',
                                    dtype=np.uint8,shape=(len(ids),3,3,224,224))
    mean=torch.tensor([.485,.456,.406])[None,:,None,None]
    std=torch.tensor([.229,.224,.225])[None,:,None,None]
    lut=lookup_table(); hashes=[];started=time.perf_counter()
    loader=DataLoader(render,batch_size=8,num_workers=12,shuffle=False)
    for batch,index in loader:
        for x,i in zip(batch,index.tolist()):
            u=((x*std+mean)*255).round().to(torch.uint8)
            decoded=lut[torch.arange(3)[None,:,None,None],u.long()]
            assert torch.equal(decoded,x)
            encoded[i]=u.numpy()
            hashes.append({'idx':i,'tensor_sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest()})
        if len(hashes)%512==0:print(f'Rendered {len(hashes)}/{len(ids)}; {time.perf_counter()-started:.1f}s',flush=True)
    encoded.flush();del encoded
    save(DEST/'render_hashes.json',hashes)
    save(DEST/'render_checks.json',{'validation':validations,'all_uint8_decodes_bit_exact':True,
                                  'cache_sha256':digest(DEST/'edited_images_uint8.npy'),'seconds':time.perf_counter()-started})
    return ids
