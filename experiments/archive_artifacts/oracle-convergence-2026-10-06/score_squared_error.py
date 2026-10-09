"""Compare saved probes on the identical weighted training population."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit, softmax
from mm_sae.analysis.data import CachedSplit

root=Path('runs/oracle-losses-2026-10-06')
records=[]
for condition in ['coco-coco','cc3m-coco']:
 out=root/condition
 protocol=json.loads((out/'protocol.json').read_text())
 cfg=protocol['config']
 train=CachedSplit(Path(cfg['source_run']),'train2017')
 keep=np.isin(train.image_ids,protocol['population']['fit_image_ids'])
 captions=keep[train.parents]
 with np.load(Path(cfg['parent_run'])/'moments.npz') as z:
  ids={s:z[s+'_ids'] for s in ['image','text']}
 for side in ['image','text']:
  raw=train.activations[side][keep if side=='image' else captions][:,ids[side]]
  labels=np.asarray(train.presence[keep] if side=='image' else train.presence[train.parents[captions]],dtype=float)
  weights=np.bincount(train.parents[captions],minlength=len(keep))[keep] if side=='image' else np.ones(len(labels))
  with np.load(out/f'{side}_moments.npz') as z:
   mean=z['mean'];scale=z['scale'];prevalence=z['label_mean'];variance=z['label_variance'];cov=z['cov'];cross=z['cross']
  methods=['ridge_0.01','ridge_0.1','ridge_1','ridge_2','binary_ce','ce']
  for method in methods:
   with np.load(out/f'{method}_{side}.npz') as z:w=z['raw']
   intercept=prevalence if method.startswith('ridge') else np.array(json.loads((out/f'{method}_{side}.json').read_text())['intercept'])
   scaled=w/scale[:,None];offset=intercept-mean@scaled
   totals=np.zeros(3)
   for start in range(0,len(labels),4096):
    target=labels[start:start+4096];wt=weights[start:start+4096]
    logits=np.asarray(raw[start:start+4096]@scaled)+offset
    pred=logits if method.startswith('ridge') else expit(logits) if method=='binary_ce' else softmax(logits,axis=1)
    totals[0]+=np.sum(wt[:,None]*(pred-target)**2)
    totals[1]+=np.sum(wt[:,None]*(np.clip(pred,0,1)-target)**2)
    totals[2]+=np.sum(wt[:,None]*(prevalence-target)**2)
   sums=totals/weights.sum()
   if method.startswith('ridge'):
    analytic=np.sum(variance-2*np.sum(w*cross,axis=0)+np.sum(w*(cov@w),axis=0))
    np.testing.assert_allclose(sums[0],analytic,rtol=1e-6,atol=1e-7)
   rec=dict(condition=condition,side=side,method=method,squared_error_sum=float(sums[0]),mse_per_concept=float(sums[0]/len(prevalence)),clipped_squared_error_sum=float(sums[1]),baseline=float(sums[2]),decrease_percent=float(100*(1-sums[0]/sums[2])))
   records.append(rec);print(json.dumps(rec),flush=True)
(root/'training_squared_error_comparison.json').write_text(json.dumps(records,indent=2))
