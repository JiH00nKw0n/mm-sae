import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json
from pathlib import Path
import numpy as np
from scipy.special import expit,softmax
from mm_sae.analysis.data import CachedSplit
root=Path('/mnt/working/mm-sae')
out={}
for cond,source in [('coco-coco','elice-rq1-lexicon2'),('cc3m-coco','cc3m-followup-2026-10-05/cc3m-sae-coco-activations')]:
 data=CachedSplit(root/'runs'/source,'val2017')
 train_ids=json.loads((root/'runs'/source/'index/train2017/concept_ids.json').read_text())
 assert data.ids==train_ids and data.presence.shape==(5000,171)
 train_images=json.loads((root/'runs'/source/'index/train2017/images.json').read_text())
 assert not set(data.image_ids)&{r['image_id'] for r in train_images}
 out[cond]={}
 for path in sorted((root/'runs/oracle-pursuit-2026-10-06/warm-all-losses-backup/results'/('warmstart-'+cond)).glob('*.npz')):
  side=path.stem.rsplit('_',1)[1]
  if side not in ['image','text']: continue
  with np.load(path) as z:
   w=z['raw']/z['scale'][:,None]
   logits=np.asarray(data.activations[side][:,z['feature_ids']]@w)-z['mean']@w+z['intercept']
  pred=expit(logits) if path.name.startswith('binary_ce') else softmax(logits,axis=1) if path.name.startswith('ce_') else logits
  y=np.asarray(data.presence if side=='image' else data.presence[data.parents],dtype=float)
  weights=np.bincount(data.parents,minlength=len(data.image_ids)) if side=='image' else np.ones(len(y))
  mse=float(np.average(np.mean((pred-y)**2,axis=1),weights=weights))
  assert np.isfinite(mse)
  out[cond][path.stem]={'mse':mse,'samples':len(y),'weight_sum':int(weights.sum())}
 print(cond,out[cond],flush=True)
Path('/mnt/working/mm-sae/runs/oracle-validation-2026-10-07.json').write_text(json.dumps(out,indent=2))
