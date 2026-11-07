import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json,time,numpy as np
from experiments.rotation_audit_20261011 import ROOT,OUT,save
from experiments import apbt_search_20261009 as p
from mm_sae.analysis.sparse_cca import fit_sparse_cca
cfg=p.old.load_config(ROOT/p.CONFIG['cc3m-cc3m'])
with np.load(Path(cfg['parent_run'])/'moments.npz') as z:
 n=len(z['image_ids']);mu=z['fit_mean'];cv=z['fit_second']-np.outer(mu,mu)
sd=np.sqrt(np.maximum(cv.diagonal(),0));sd[sd==0]=1;cv/=sd[:,None]*sd[None,:]
with np.load(ROOT/'runs/mapping-ablation-cc3m-2026-10-04/transforms/cca_3221.npz') as z:wi=z['image'][:,:512];wt=z['text'][:,:512]
dest=OUT/'sparse-cca16-512.npz'
if not dest.exists():
 start=time.time()
 def progress(r):
  save(OUT/'sparse-progress.json',dict(component=r['component'],seconds=time.time()-start,converged=r['converged']));print(r['component'],time.time()-start,flush=True)
 m=fit_sparse_cca(cv[:n,:n],cv[:n,n:],cv[n:,n:],512,k=16,ridge=.01,initial_image=wi,initial_text=wt,max_iter=100,progress=progress)
 np.savez_compressed(dest,image=m.image,text=m.text)
 save(OUT/'sparse-fit.json',dict(seconds=time.time()-start,metadata=m.metadata if hasattr(m,'metadata') else str(m)))
