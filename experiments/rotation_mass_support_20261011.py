"""Fit support-matched sparse CCA; budgets depend only on trained coefficients."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import json,time,traceback
import numpy as np
from pathlib import Path
from experiments.rotation_mass_audit_20261011 import ROOT,OUT,SRC,p,save
from mm_sae.analysis.sparse_cca import fit_sparse_cca

def run():
 cfg=p.old.load_config(ROOT/p.CONFIG['cc3m-cc3m'])
 with np.load(Path(cfg['parent_run'])/'moments.npz') as z:
  n=len(z['image_ids']);mu=z['fit_mean'];cv=z['fit_second']-np.outer(mu,mu)
 sd=np.sqrt(np.maximum(cv.diagonal(),0));sd[sd==0]=1;cv/=sd[:,None]*sd[None,:]
 xx,cross,yy=cv[:n,:n],cv[:n,n:],cv[n:,n:];aa=xx+.01*np.eye(n);bb=yy+.01*np.eye(len(yy))
 dense=np.load(ROOT/'runs/mapping-ablation-cc3m-2026-10-04/transforms/cca_3221.npz')
 folder=OUT/'sparse';folder.mkdir(exist_ok=True)
 for job in json.loads((OUT/'support-jobs.json').read_text()):
  k,r=job['support'],job['rank'];dest=folder/f'cca-k{k}-r{r}.npz'
  if dest.exists():continue
  sources=[SRC/'sparse-cca16-512.npz'] if k==16 else []
  sources += [ROOT/f'runs/cc3m-followup-2026-10-05/cc3m-fit/semantics/sparse-fit/sparse_cca_k{k}.npz',ROOT/f'runs/sparsity-sweep-2026-10-06/cc3m-cc3m/cca_{k}.npz']
  prefix=None;source=None
  for path in sources:
   if path.exists():
    z=np.load(path)
    if 'image' in z and 'text' in z and (prefix is None or z['image'].shape[1]>prefix['image'].shape[1]):
     prefix={s:z[s].astype(float)[:,:r] for s in ['image','text']};source=str(path)
  count=0 if prefix is None else prefix['image'].shape[1];res=cross.copy()
  for j in range(count):
   u,v=prefix['image'][:,j],prefix['text'][:,j];rho=u@res@v;res-=rho*np.outer(aa@u,bb@v)
  start=time.time();meta={'support':k,'rank':r,'cached_prefix_columns':count,'cached_source':source}
  if count<r:
   def progress(rec):
    save(OUT/'support-progress.json',dict(**meta,component=count+rec['component']+1,seconds=time.time()-start,converged=rec['converged']))
   model=fit_sparse_cca(xx,res,yy,r-count,k=k,ridge=.01,initial_image=dense['image'][:,count:r],initial_text=dense['text'][:,count:r],max_iter=100,progress=progress)
   w={s:np.concatenate([prefix[s],getattr(model,s)],axis=1) if prefix is not None else getattr(model,s) for s in ['image','text']}
   meta['new_fit']=model.metadata
  else:w=prefix
  signs=np.sign(np.sum(w['image']*(cross@w['text']),axis=0));signs[signs==0]=1;w['text']*=signs
  np.savez_compressed(dest,**w);save(dest.with_suffix('.json'),dict(**meta,seconds=time.time()-start))
 save(OUT/'support-fit-complete.json',dict(time=time.time()))
if __name__=='__main__':
 try:run()
 except Exception as e:save(OUT/'error-support.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
