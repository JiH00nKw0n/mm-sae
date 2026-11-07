"""Evaluate rank-matched Sparse CCA16 with exactly the audit's frozen splits/sign policy."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import numpy as np,json,torch,time
from pathlib import Path
from experiments.rotation_audit_20261011 import *
cfg=p.old.load_config(ROOT/p.CONFIG['cc3m-cc3m'])
while not (OUT/'sparse-cca16-512.npz').exists():time.sleep(20)
# Avoid simultaneously allocating both experiment evaluation populations on the GPU.
while not (OUT/'complete.json').exists():
 if (OUT/'error.json').exists():raise RuntimeError('Rotation audit failed; inspect error.json')
 time.sleep(20)
with np.load(Path(cfg['parent_run'])/'moments.npz') as z:
 ni=len(z['image_ids']);mu=z['fit_mean'];cov=z['fit_second']-np.outer(mu,mu)
sd=np.sqrt(np.maximum(cov.diagonal(),0));sd[sd==0]=1;cov/=sd[:,None]*sd[None,:];cv=[tensor(cov[:ni,ni:]),tensor(cov[:ni,:ni]),tensor(cov[ni:,ni:])]
data,parents=p.eval_population(cfg,'test',OUT);eligible=np.isin([c['id'] for c in data['concepts']],data['metadata']['eligible_category_ids']);assert eligible.sum()==50
labels={s:tensor(data['labels'][s][:,eligible].astype(float)) for s in SIDES};evalx={s:tensor((data['raw'][s].toarray()-data['means'][s])/data['scales'][s]) for s in SIDES}
z=np.load(OUT/'evaluation-splits.npz');splits=[(torch.as_tensor(z[f'image_{seed}'],device=DEV),torch.as_tensor(z[f'text_{seed}'],device=DEV)) for seed in range(20)]
for ai,bt in [('image_a','text_b'),('image_b','text_a')]:splits.append((torch.as_tensor(np.flatnonzero(data['masks'][ai]),device=DEV),torch.as_tensor(np.flatnonzero(data['masks'][bt]),device=DEV)))
_,arr,_=p.prepare('cc3m-cc3m',OUT,32768,0)
with np.load(OUT/'sparse-cca16-512.npz') as z:full={s:z[s] for s in SIDES}
for rank in [128,256,512]:
 w={s:tensor(full[s][:,:rank]) for s in SIDES};fitout=[(arr['fit',s]-arr['mean',s]).double()@w[s] for s in SIDES]
 evaluate('sparse-cca16',rank,w,np.eye(rank),fitout,evalx,labels,splits,parents,cv,None,prune=False)
save(OUT/'all-complete.json',dict(complete=True,time=time.time()))
