import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import numpy as np
from pathlib import Path
import time,json
from scipy.optimize import linear_sum_assignment
import base as old
from run import generator

def decorrelate(w):
 v,q=np.linalg.eigh(w@w.T)
 return (q*(1/np.sqrt(np.maximum(v,1e-14))))@q.T@w

def fit(x,seed):
 x=np.asarray(x,dtype=float);x=x-x.mean(0)
 vals,vec=np.linalg.eigh(x.T@x/len(x));ix=np.argsort(vals)[-8:][::-1]
 k=vec[:,ix]/np.sqrt(np.maximum(vals[ix],1e-10))[None,:]
 z=(x@k).T;w=decorrelate(np.random.default_rng(seed).normal(size=(8,8)))
 delta=1.
 for iteration in range(2000):
  u=w@z;gu=np.tanh(u)
  wn=decorrelate(gu@z.T/len(x)-(1-gu**2).mean(1)[:,None]*w)
  delta=float(np.max(np.abs(np.abs(np.diag(wn@w.T))-1)));w=wn
  if delta<1e-6:break
 return (k@w.T).astype('float32'),dict(iterations=iteration+1,converged=delta<1e-6,delta=delta,eigenvalues=vals[ix].tolist())

out=Path('/mnt/working/mm-sae/runs/synthetic-reconstruction-2026-10-09');count=0
for kind in ['anchors','noanchors']:
 for seed in range(3):
  di=generator(1000+seed,kind);dt=generator(2000+seed,kind)
  for rho in [0,.5,.9,1]:
   dest=out/f'{kind}-seed{seed}-rho{rho:.1f}';dest.mkdir(exist_ok=True)
   x,y,_=old.observations(3000+seed,8192,rho,di,dt)
   means=[x.mean(0),y.mean(0)];scales=[x.std(0).clip(1e-6),y.std(0).clip(1e-6)]
   xn=(x-means[0])/scales[0];yn=(y-means[1])/scales[1]
   a,ai=fit(xn,seed);b,bi=fit(yn,seed+100)
   u=xn@a;v=yn@b;u=(u-u.mean(0))/u.std(0);v=(v-v.mean(0))/v.std(0)
   c=u.T@v/len(x);ri,ci=linear_sum_assignment(-np.abs(c));b=b[:,ci]*np.sign(c[ri,ci])[None,:]
   old.save(dest/'fastica-fit.json',dict(image=ai,text=bi,matching='train absolute Pearson + Hungarian + sign orientation; no labels',text_permutation=ci.tolist(),matched_train_correlation=c[ri,ci].tolist(),support_limit=None))
   tests=dict(iid=old.observations(5000+seed,2048,rho,di,dt),recombined=old.observations(6000+seed,2048,0,di,dt))
   old.evaluate('fastica',a,b,True,means,scales,di,dt,tests,dest)
   count+=1;old.save(out/'ica-progress.json',dict(completed=count,total=24,state='completed' if count==24 else 'running'))
