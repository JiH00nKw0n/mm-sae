"""Measure frozen final rotations. No fitting or thresholding of the representations."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import json,time
from pathlib import Path
import numpy as np
import experiments.apbt_search_20261009 as p

ROOT=p.ROOT
SRC=ROOT/'runs/rotation-convergence-2026-10-09'
OUT=SRC/'sparsity'
OUT.mkdir(exist_ok=True)

def summary(x):
 return dict(mean=float(np.mean(x)),median=float(np.median(x)),p10=float(np.quantile(x,.1)),p90=float(np.quantile(x,.9)),min=float(np.min(x)),max=float(np.max(x)))

def measure(z,axis):
 """Axis indexes entries of each vector. Zero vectors get zero required/effective count."""
 energy=np.square(z);total=energy.sum(axis=axis);ordered=np.flip(np.sort(energy,axis=axis),axis=axis)
 cumulative=np.cumsum(ordered,axis=axis);den=np.expand_dims(total,axis=axis)
 counts={str(q):np.where(total>0,(cumulative<q*den).sum(axis=axis)+1,0) for q in [.9,.95,.99]}
 eff=np.divide(np.abs(z).sum(axis=axis)**2,total,out=np.zeros_like(total),where=total>0)
 top16=np.take(ordered,np.arange(min(16,z.shape[axis])),axis=axis).sum(axis=axis)
 return dict(shape=list(z.shape),vector_size=z.shape[axis],vector_count=int(total.size),exact_zero_fraction=float(np.mean(z==0)),zero_vector_fraction=float(np.mean(total==0)),required_count={q:summary(v) for q,v in counts.items()},effective_count=summary(eff),top16_squared_mass=summary(np.divide(top16,total,out=np.zeros_like(total),where=total>0)),negative_squared_mass=float(energy[z<0].sum()/energy.sum()))

def main():
 start=time.time();cfg=p.old.load_config(ROOT/p.CONFIG['cc3m-cc3m'])
 data,_=p.eval_population(cfg,'test',OUT)
 with np.load(next(m['path'] for m in cfg['models'] if m['key']=='cca_256')) as z:
  weights={s:z[s].astype(np.float64) for s in ['image','text']}
 base={s:np.asarray(data['raw'][s]@(weights[s]/data['scales'][s][:,None]))-data['means'][s]@(weights[s]/data['scales'][s][:,None]) for s in weights}
 jobs=[('cca_original',np.eye(256))]+[(f.name.replace('-final-evaluated-rotation.npz',''),np.load(f)['rotation']) for f in sorted(SRC.glob('*-final-evaluated-rotation.npz'))]
 assert len(jobs)==22,len(jobs)
 for name,r in jobs:
  dest=OUT/(name+'.json')
  if dest.exists():continue
  result=dict(name=name,condition='CC3M SAE / CC3M mapping',sides={})
  for s in weights:
   result['sides'][s]=dict(coefficients=measure(weights[s]@r,axis=0),outputs=measure(base[s]@r,axis=1))
  dest.write_text(json.dumps(result,indent=2));print(name,round(time.time()-start,1),flush=True)
 (OUT/'protocol.json').write_text(json.dumps(dict(condition='CC3M SAE / CC3M mapping',rank=256,coefficient_basis='Coefficients acting on train-standardized SAE activation; column-wise statistics.',output_population='COCO val2017. All 5000 image rows and all 25014 caption rows, each row counted once in its modality.',output_basis='Centered/scaled with frozen training statistics, followed by frozen CCA and common rotation. Signed outputs, no threshold or further normalization.',counts='Smallest count covering 90/95/99 percent of squared coefficient or output magnitude. Not a count of exactly nonzero entries. Not causal contribution.',effective_count='L1 norm squared divided by L2 norm squared, zero for all-zero vectors.',aggregation='Compute median across coordinates for coefficients or examples for output, then report mean across the same three initializations.',training_performed=False,elapsed_seconds=time.time()-start),indent=2))

if __name__=='__main__':main()
