"""Prefill not-yet-started covariance rotations only after original GPU work completes."""
import os,time,json,argparse
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import numpy as np
from scipy import linalg
import torch
from experiments import rotation_audit_20261011 as a
ROOT=Path('/mnt/working/mm-sae');OUT=ROOT/'runs/conditioned-svd-axes-2026-10-12';torch.set_num_threads(2)
def active(stat,mode=None,seed=None):
 for file in ['progress.json','cca-progress.json']:
  try:
   q=json.loads((OUT/file).read_text());name=q.get('name',q.get('statistic',''))
   if name.startswith(stat) and q.get('stage')!='complete':
    if mode is None:return q.get('stage')=='svd'
    if name==stat+'-'+mode and seed<=int(q.get('seed',-1))+1:return True
  except (FileNotFoundError,json.JSONDecodeError):pass
 return False
def npatomic(path,**z):
 tmp=path.with_name(path.stem+'.prefit.tmp.npz');np.savez_compressed(tmp,**z);tmp.replace(path)
def main():
 while not (ROOT/'runs/cc3m-score-rotation-2026-10-12/complete.json').exists():
  if (ROOT/'runs/cc3m-score-rotation-2026-10-12/error.json').exists() or not Path('/proc/616').exists():break
  time.sleep(30)
 while True:
  free,total=torch.cuda.mem_get_info()
  if free>4*1024**3:break
  time.sleep(30)
 a.save(OUT/'prefit-memory.json',dict(free_bytes=free,total_bytes=total,original_complete=True));print('GPU free bytes',free,flush=True)
 with np.load(ROOT/'runs/mapping-suite-cc3m-2026-10-04/moments.npz') as z:
  n=len(z['image_ids']);mu=z['fit_mean'];sec=z['fit_second'];sd=np.sqrt(sec.diagonal()-mu*mu);cv=(sec-np.outer(mu,mu))/sd[:,None]/sd[None,:]
 sx=cv[:n,:n].copy();sy=cv[n:,n:].copy();C=cv[:n,n:].copy();del cv,sec;ex,Qx=linalg.eigh(sx);ey,Qy=linalg.eigh(sy)
 # Reverse order keeps this worker away from the ongoing CPU condition.
 for stat in ['CCA-1','CCA-0.1','CCA-0.01','W-0.01','W2-1','W2-0.1','W2-0.01']:
  if active(stat):print('leave active CPU statistic',stat,flush=True);continue
  power=.5 if stat.startswith('CCA') else 1.;lam=float(stat.split('-')[1]);lx=(Qx/(ex+lam)**power)@Qx.T;ly=(Qy/(ey+lam)**power)@Qy.T if not stat.startswith('W-') else None;bp=OUT/'bases'/f'{stat}.npz'
  if bp.exists():z=np.load(bp);u,v=z['U'],z['V']
  else:
   M=lx@C;M=M@ly if ly is not None else M;uu,sv,vh=linalg.svd(M,full_matrices=False,check_finite=False);u,v,sv=uu[:,:256],vh[:256].T,sv[:256];err=float(np.linalg.norm(M@v-u*sv)/np.linalg.norm(u*sv));assert err<1e-8;npatomic(bp,U=u,V=v,singular_values=sv,residual=err);del M,uu,vh
  for mode in ['direct','pullback']:
   name=stat+'-'+mode;base=[lx@u if mode=='pullback' else u,ly@v if mode=='pullback' and ly is not None else v];loads=[]
   for co,w in zip([sx,sy],base):l=co@w;l/=np.maximum(np.linalg.norm(l,axis=1,keepdims=True),1e-12);loads.append(a.tensor(l))
   for seed in reversed(range(5)):
    dst=OUT/'fits'/f'{name}-s{seed}.npz'
    if dst.exists() or active(stat,mode,seed):continue
    a.OUT=OUT/'prefit'/name;a.OUT.mkdir(parents=True,exist_ok=True);args=argparse.Namespace(steps=20000,grad_tol=1e-6,obj_tol=1e-10);print('prefit',name,seed,flush=True);r=a.fit(256,'covariance','svd',seed,loads,None,args);meta=json.loads((a.OUT/f'rank256/covariance-svd-{seed}-fit.json').read_text());meta['relative_gradient']=meta['gradient_relative'];meta['device']='GPU after original CC3M complete';meta['sign']='orientation applied by evaluation driver using common CC3M third moments'
    if not dst.exists():npatomic(dst,rotation=r,image=base[0]@r,text=base[1]@r);a.save(dst.with_suffix('.json'),meta)
   del loads;torch.cuda.empty_cache()
 a.save(OUT/'prefit-complete.json',dict(time=time.time()))
if __name__=='__main__':
 try:main()
 except Exception as e:
  import traceback
  a.save(OUT/'prefit-error.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
