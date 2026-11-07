"""Prune each frozen A/B column to 99% squared coefficient mass, then evaluate retrieval."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json,time
import numpy as np
import experiments.apbt_search_20261009 as p
ROOT=p.ROOT;SRC=ROOT/'runs/rotation-convergence-2026-10-09';OUT=SRC/'prune99';OUT.mkdir(exist_ok=True)

def prune(w):
 order=np.argsort(-np.square(w),axis=0,kind='stable');energy=np.take_along_axis(w*w,order,axis=0);total=energy.sum(0)
 counts=np.where(total>0,(energy.cumsum(0)<.99*total).sum(0)+1,0)
 mask=np.arange(w.shape[0])[:,None]<counts[None,:];support=np.zeros_like(w,dtype=bool);np.put_along_axis(support,order,mask,axis=0)
 ww=np.where(support,w,0);mass=np.divide((ww*ww).sum(0),total,out=np.ones_like(total),where=total>0)
 assert mass.min()>=.99-1e-12
 return ww,dict(retained_count_median=float(np.median(counts)),retained_count_mean=float(counts.mean()),retained_counts=counts.tolist(),retained_squared_mass_min=float(mass.min()),retained_squared_mass_mean=float(mass.mean()),exact_zero_fraction=float(np.mean(ww==0)))

def main():
 start=time.time();cfg=p.old.load_config(ROOT/p.CONFIG['cc3m-cc3m']);data,parents=p.eval_population(cfg,'test',OUT)
 with np.load(next(m['path'] for m in cfg['models'] if m['key']=='cca_256')) as z:w={s:z[s].astype(float) for s in ['image','text']}
 jobs=[('cca_original',np.eye(256),ROOT/'runs/cca-unsupervised-rotations-2026-10-09/cc3m-cc3m/cca_original-result.json')]
 jobs += [(f.name.replace('-final-evaluated-rotation.npz',''),np.load(f)['rotation'],f.with_name(f.name.replace('-evaluated-rotation.npz','-result.json'))) for f in sorted(SRC.glob('*-final-evaluated-rotation.npz'))]
 assert len(jobs)==22
 for name,r,beforepath in jobs:
  dest=OUT/f'{name}.json'
  if dest.exists():continue
  before=json.loads(beforepath.read_text())['retrieval'];sparse={};stats={};scores={}
  for s in w:
   sparse[s],stats[s]=prune(w[s]@r)
   scaled=sparse[s]/data['scales'][s][:,None]
   scores[s]=np.asarray(data['raw'][s]@scaled)-data['means'][s]@scaled
  ret=p.paired_retrieval(scores['image'],scores['text'],parents,device='cuda',chunk_size=128)
  ret={d:{k:v for k,v in val.items() if k not in ['hits','ranks']} for d,val in ret.items()}
  p.save(dest,dict(name=name,condition='CC3M SAE / CC3M mapping',before=before,after=ret,sparsity=stats,elapsed_seconds=time.time()-start))
  np.savez_compressed(OUT/f'{name}-weights.npz',**sparse)
  print(name,{d:v['recall'] for d,v in ret.items()},round(time.time()-start,1),flush=True)
 p.save(OUT/'protocol.json',dict(condition='CC3M SAE / CC3M mapping',rank=256,retained_squared_mass=.99,selection='Each column separately. Keep descending absolute coefficients until cumulative squared mass reaches at least .99; exact smallest count, stable tie order.',sides='Both image A and text B pruned.',normalization='Unchanged train mean/std. No refit, coefficient renormalization, output centering or whitening. Standard cosine normalization only.',evaluation='COCO val2017 5000 image queries and 25014 caption queries; Recall@1/5/10.',before='Matching frozen-rotation result from the convergence experiment; original CCA baseline from the original rotation experiment.',training_performed=False,number_of_models=22,seconds=time.time()-start))

if __name__=='__main__':main()
