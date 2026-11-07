"""Correlation-matched coordinate removal, native and variance-equalized retrieval."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json,time
import numpy as np
from scipy.optimize import linear_sum_assignment
from experiments import apbt_search_20261009 as p
from mm_sae.analysis.mapping_evaluation import paired_retrieval
ROOT=p.ROOT;OUT=ROOT/'runs/shared-variance-ablation-2026-10-10';OUT.mkdir(exist_ok=True)
cfg=p.old.load_config(ROOT/p.CONFIG['cc3m-cc3m']);start=time.time();sides=['image','text']
with np.load(Path(cfg['parent_run'])/'moments.npz') as z:
 ni=len(z['image_ids']);mu=z['fit_mean'];cv=z['fit_second']-np.outer(mu,mu)
sd=np.sqrt(np.maximum(cv.diagonal(),0));sd[sd==0]=1;cv/=sd[:,None]*sd[None,:]
with np.load(ROOT/'runs/mapping-ablation-cc3m-2026-10-04/transforms/cross_svd_3221.npz') as z:weights={s:z[s][:,:512].astype(float) for s in sides}
data,parents=p.eval_population(cfg,'test',OUT)
base={s:np.asarray(data['raw'][s]@(weights[s]/data['scales'][s][:,None]))-data['means'][s]@(weights[s]/data['scales'][s][:,None]) for s in sides}
vi=np.sum(weights['image']*(cv[:ni,:ni]@weights['image']),axis=0)
vt=np.sum(weights['text']*(cv[ni:,ni:]@weights['text']),axis=0)
cross=np.sum(weights['image']*(cv[:ni,ni:]@weights['text']),axis=0)
variance=np.sqrt(vi*vt);rho=cross/variance
protocol={'condition':'CC3M SAE / CC3M mapping','selection':'Training moments only. Within 8 rho-quantile bins match upper/lower variance halves by absolute rho distance; select 4 closest pairs in each bin. Same 32 removed coordinates per side. Balanced variance is sqrt(Var(image)*Var(text)).',
'evaluation':'Frozen COCO val2017 both retrieval directions. No label fitting. Original and training-variance-equalized outputs. Five random removals each using the same 4-per-correlation-bin count.',
'limits':'Coordinate-removal utility is conditional on retained representation, not intrinsic semantic value. Original scale weighting and redundancy affect utility. Matching is approximate; per-pair rho gaps are reported. Does not prove concept purity or raw embedding reconstruction importance.'}
p.save(OUT/'protocol.json',protocol)
for rank in [256,512]:
 folder=OUT/str(rank);folder.mkdir(exist_ok=True)
 bins=np.array_split(np.argsort(rho[:rank]),8);high=[];low=[];pairs=[]
 for block in bins:
  order=block[np.argsort(variance[block])];lo=order[:len(order)//2];hi=order[len(order)//2:]
  costs=np.abs(rho[hi,None]-rho[None,lo]);ii,jj=linear_sum_assignment(costs)
  chosen=np.argsort(costs[ii,jj])[:4]
  for n in chosen:
   h,l=int(hi[ii[n]]),int(lo[jj[n]]);high.append(h);low.append(l)
   pairs.append(dict(high=h,low=l,rho_high=float(rho[h]),rho_low=float(rho[l]),rho_gap=float(abs(rho[h]-rho[l])),variance_high=float(variance[h]),variance_low=float(variance[l])))
 rng=np.random.default_rng(2041)
 masks={'none':[], 'high_variance':high,'low_variance':low}
 for seed in range(5):masks[f'random_{seed}']=np.concatenate([rng.choice(b,4,replace=False) for b in bins]).tolist()
 p.save(folder/'selection.json',dict(pairs=pairs,masks=masks,variance=variance[:rank].tolist(),rho=rho[:rank].tolist()))
 for mode in ['original','unit_variance']:
  scores={s:base[s][:,:rank].copy() for s in sides}
  if mode=='unit_variance':
   scores['image']/=np.sqrt(vi[:rank]);scores['text']/=np.sqrt(vt[:rank])
  for name,removed in masks.items():
   dest=folder/f'{mode}-{name}.json'
   if dest.exists():continue
   keep=np.ones(rank,bool);keep[removed]=False
   ret=paired_retrieval(scores['image'][:,keep],scores['text'][:,keep],parents,device='cuda',chunk_size=128)
   result={d:{k:v for k,v in o.items() if k not in ['hits','ranks']} for d,o in ret.items()}
   np.savez_compressed(folder/f'{mode}-{name}-ranks.npz',image=ret['image_to_text']['ranks'],text=ret['text_to_image']['ranks'],parents=parents)
   p.save(dest,result);print(rank,mode,name,[o['recall'][5] for o in ret.values()],flush=True)
 p.save(OUT/'progress.json',dict(rank=rank,stage='complete',seconds=time.time()-start))
p.save(OUT/'complete.json',dict(complete=True,seconds=time.time()-start))
