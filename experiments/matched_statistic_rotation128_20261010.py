import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json,time,shutil
import numpy as np
import torch
from experiments import rotation_convergence_20261009 as conv
from mm_sae.analysis.semantic_evaluation import auc_matrix
from mm_sae.analysis.representative_agreement import variable_coordinates
p=conv.p;ROOT=p.ROOT;OUT=ROOT/'runs/matched-statistic-rotation128-2026-10-10';OUT.mkdir(exist_ok=True)
cfg=p.old.load_config(ROOT/p.CONFIG['cc3m-cc3m']);sides=['image','text'];start=time.time()
original_objective=conv.objective
def correlation_objective(r,kind,ts,tune=False):
    values=[]
    for load,g in zip(ts['load'],ts['gram']):
        out_sd=(r*(g@r)).sum(0).clamp_min(1e-20).sqrt()
        corr=(load@r)/out_sd[None,:]
        corr=corr/corr.norm(dim=1,keepdim=True).clamp_min(1e-12)
        values.append((corr.pow(4).mean(0)-corr.square().mean(0).square()).sum())
    return -sum(values)/2
def evaluate_matching(scores):
    arr={}
    for s in sides:
        for half in ['a','b']:
            key=s+'_'+half;mask=data['masks'][key]
            arr[key]=auc_matrix(scores[s][mask],data['labels'][s][mask],chunk_size=32)
            arr[key+'_variable']=variable_coordinates(scores[s][mask])
    arr['eligible']=np.isin([c['id'] for c in data['concepts']],data['metadata']['eligible_category_ids'])
    assert arr['eligible'].sum()==50
    arr['category_ids']=np.array([c['id'] for c in data['concepts']])
    normal=p.old.evaluation(arr,{**cfg,'n_null':0},signed=False)
    swapped={k:v for k,v in arr.items()}
    for s in sides:
        for suff in ['', '_variable']:
            swapped[s+'_a'+suff]=arr[s+'_b'+suff];swapped[s+'_b'+suff]=arr[s+'_a'+suff]
    repeat=p.old.evaluation(swapped,{**cfg,'n_null':0},signed=False)
    return normal,repeat

with np.load(Path(cfg['parent_run'])/'moments.npz') as z:
 ni=len(z['image_ids']);mu=z['fit_mean'];cv=z['fit_second']-np.outer(mu,mu)
sd=np.sqrt(np.maximum(cv.diagonal(),0));sd[sd==0]=1;cv/=sd[:,None]*sd[None,:]
sigmas={'image':cv[:ni,:ni],'text':cv[ni:,ni:]}
data,parents=p.eval_population(cfg,'test',OUT)
for rank in [128]:
 for model,stem,stat in [('pls','cross_svd','covariance'),('cca','cca','correlation')]:
  folder=OUT/f'{model}-{rank}';folder.mkdir(exist_ok=True);conv.OUT=folder
  with np.load(ROOT/f'runs/mapping-ablation-cc3m-2026-10-04/transforms/{stem}_3221.npz') as z:w={s:z[s][:,:rank].astype(float) for s in sides}
  if rank==256:
   with np.load(ROOT/f'runs/mapping-ablation-cc3m-2026-10-04/transforms/{stem}_256.npz') as z:w={s:z[s].astype(float) for s in sides}
  loads=[sigmas[s]@w[s] for s in sides]
  grams=[w[s].T@l for s,l in zip(sides,loads)]
  conv.objective=correlation_objective if model=='cca' else original_objective
  ts={'load':[torch.tensor(l if model=='cca' else l/np.maximum(np.linalg.norm(l,axis=1,keepdims=True),1e-12),device='cuda',dtype=torch.float64) for l in loads],
      'gram':[torch.tensor(g,device='cuda',dtype=torch.float64) for g in grams]}
  method='correlation_varimax' if model=='cca' else 'loading_varimax'
  for seed in range(3):
   if rank==256:
    source=ROOT/('runs/pls-cca-gap-2026-10-10/cca' if model=='cca' else 'runs/pls-loading-rotation-2026-10-09')
    for suffix in ['final.npz','fit.json','history.json']:shutil.copy2(source/f'{method}-{seed}-{suffix}',folder/f'{method}-{seed}-{suffix}')
   else:conv.fit(method,seed,ts,type('Args',(),{'max_steps':6000,'rank':rank})())
  del ts;torch.cuda.empty_cache()
  if rank!=256:
   _,arr,_=p.prepare('cc3m-cc3m',folder,32768,0)
   fit={s:((arr['fit',s]-arr['mean',s])@torch.tensor(w[s],device='cuda',dtype=torch.float32)).cpu().numpy().astype(float) for s in sides}
   del arr;torch.cuda.empty_cache()
  else:
   source=ROOT/('runs/pls-cca-gap-2026-10-10' if model=='cca' else 'runs/pls-cca-gap-2026-10-10')
  base={s:np.asarray(data['raw'][s]@(w[s]/data['scales'][s][:,None]))-data['means'][s]@(w[s]/data['scales'][s][:,None]) for s in sides}
  ret=json.loads((ROOT/f'runs/pls-cca-rank-subspace-2026-10-10/{model}-native-{rank}-result.json').read_text())['retrieval']
  for seed in range(3):
   dest=folder/f'{seed}-result.json'
   if dest.exists():continue
   if rank==256:
    old=json.loads((source/f'{model}-{stat}-{seed}-result.json').read_text());old['retrieval']=ret;p.save(dest,old);continue
   r=np.load(folder/f'{method}-{seed}-final.npz')['rotation']
   skew=sum(np.mean((fit[s]@r)**3,axis=0) for s in sides);r=r*np.where(skew<0,-1.,1.)
   scores={s:base[s]@r for s in sides};normal,repeat=evaluate_matching(scores)
   norm=lambda x:x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-15)
   err=float(np.max(abs(norm(base['image'][:128])@norm(base['text'][:256]).T-norm(scores['image'][:128])@norm(scores['text'][:256]).T)))
   assert err<1e-10
   p.save(dest,dict(matching50=normal,repeat_matching50=repeat,retrieval=ret,sample_cosine_error=err,retrieval_source='Same full-coordinate common-orthogonal-rotation scores; inherited verified native Recall.'))
   np.savez_compressed(folder/f'{seed}-transforms.npz',image=w['image']@r,text=w['text']@r,rotation=r)
   print(model,rank,seed,normal['summary']['agree_at1_count'],repeat['summary']['agree_at1_count'],flush=True)
  p.save(OUT/'progress.json',dict(model=model,rank=rank,stage='evaluated',elapsed=time.time()-start))
p.save(OUT/'complete.json',dict(complete=True,seconds=time.time()-start))
