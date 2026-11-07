"""Compare exact-coordinate matching, AUROC ambiguity, and category collapse."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import json,time,traceback
import numpy as np,torch
from pathlib import Path
from experiments.rotation_mass_audit_20261011 import Evaluator,ROOT,SRC,a,p
OUT=ROOT/'runs/hungarian-rotation-diagnosis-2026-10-11';OUT.mkdir(exist_ok=True)
def save(path,z):p.save(path,z)
def run():
 e=Evaluator();cat=[c for c in e.data['concepts'] if c['id'] in e.data['metadata']['eligible_category_ids']];save(OUT/'categories.json',cat)
 def evaluate(name,w):
  dest=OUT/(name+'.json')
  if dest.exists():return
  start=time.time();sc={s:e.x[s]@w[s] for s in ['image','text']};rows=[];aucs=[]
  for split,(ii,tt) in enumerate(e.splits[:20]):
   au=[a.gpu_auc(sc['image'][ii],e.labels['image'][ii]),a.gpu_auc(sc['text'][tt],e.labels['text'][tt])]
   valid=torch.isfinite(au[0])&torch.isfinite(au[1])
   for m in au:m[~valid]=-torch.inf
   top=[torch.topk(m,2,dim=0) for m in au];idx=[m.argmax(0) for m in au];cols=torch.arange(50,device='cuda');match=idx[0]==idx[1]
   row={'split':split,'per_category':match.int().cpu().tolist(),'count':int(match.sum())}
   for j,s in enumerate(['image','text']):
    vals=top[j].values;best=idx[j];other=idx[1-j];counts=torch.bincount(best,minlength=len(au[j]))
    row[s]={'best_axis':best.cpu().tolist(),'best_auc':vals[0].cpu().tolist(),'second_auc':vals[1].cpu().tolist(),'gap':(vals[0]-vals[1]).cpu().tolist(),'counterpart_axis_auc':au[j][other,cols].cpu().tolist(),'unique_best_axes':int((counts>0).sum()),'largest_category_count_on_one_axis':int(counts.max()),'all_axis_auc_quantiles':torch.quantile(au[j][torch.isfinite(au[j])],torch.tensor([.1,.25,.5,.75,.9,.99],device='cuda',dtype=torch.float64)).cpu().tolist(),'axes_above_080':(au[j]>=.8).sum(0).cpu().tolist()}
   # Category discrimination of same-index matches; label criterion unchanged.
   row['matches_both_auc_070']=int((match&(top[0].values[0]>=.7)&(top[1].values[0]>=.7)).sum());row['matches_both_auc_080']=int((match&(top[0].values[0]>=.8)&(top[1].values[0]>=.8)).sum())
   row['top2_overlap_categories']=int(((top[0].indices[:,None,:]==top[1].indices[None,:,:]).any(0).any(0)).sum())
   # Unique coordinates occupied by exact matches limits repeated credit for one axis.
   row['unique_matched_axes']=int(torch.unique(idx[0][match]).numel())
   rows.append(row);aucs.append(np.stack([m.cpu().numpy() for m in au]))
   save(OUT/'progress.json',dict(method=name,split=split+1,seconds=time.time()-start))
  ret=p.paired_retrieval(sc['image'].cpu().numpy(),sc['text'].cpu().numpy(),e.parents,device='cuda',chunk_size=128)
  ret={d:{k:v for k,v in m.items() if k not in ['hits','ranks']} for d,m in ret.items()}
  np.savez_compressed(OUT/(name+'-auc.npz'),auc=np.stack(aucs))
  save(dest,dict(name=name,rank=w['image'].shape[1],rows=rows,retrieval=ret,seconds=time.time()-start))
 cfg=e.cfg
 with np.load(Path(cfg['parent_run'])/'moments.npz') as z:
  ni=len(z['image_ids']);mu=z['fit_mean'];cv=z['fit_second']-np.outer(mu,mu)
 sd=np.sqrt(np.maximum(cv.diagonal(),0));sd[sd==0]=1;cv/=sd[:,None]*sd[None,:]
 mp=np.load(next(m['path'] for m in cfg['models'] if m['key']=='hungarian'))['mapping'];ii,jj=np.nonzero(mp);ix=np.argsort(-cv[ii,ni+jj],kind='stable');ii=ii[ix];jj=jj[ix]
 for rank in [64,128,256,512,len(ii)]:
  w={s:torch.zeros((d,rank),device='cuda',dtype=torch.float64) for s,d in [('image',mp.shape[0]),('text',mp.shape[1])]}
  w['image'][torch.as_tensor(ii[:rank],device='cuda'),torch.arange(rank,device='cuda')]=1;w['text'][torch.as_tensor(jj[:rank],device='cuda'),torch.arange(rank,device='cuda')]=1
  evaluate(f'hungarian-{rank}',w)
 base=e.matrices(ROOT/'runs/mapping-ablation-cc3m-2026-10-04/transforms/cross_svd_3221.npz',256)
 for method in ['covariance-svd','correlation-cayley','coefficient-svd']:
  for seed in range(10):
   rr=a.tensor(np.load(SRC/f'rank256/{method}-{seed}-oriented.npz')['rotation']);evaluate(f'{method}-{seed}',{s:base[s]@rr for s in base})
 save(OUT/'complete.json',dict(time=time.time()))
if __name__=='__main__':
 try:run()
 except Exception as ex:save(OUT/'error.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
