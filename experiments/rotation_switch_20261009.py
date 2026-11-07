"""Equal-budget two-stage rotations plus winsorized-kurtosis sensitivity. CC3M only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json,time,numpy as np,torch
import experiments.apbt_search_20261009 as p
from mm_sae.analysis.semantic_evaluation import auc_matrix
from mm_sae.analysis.representative_agreement import variable_coordinates
ROOT=p.ROOT;OUT=ROOT/'runs/rotation-switch-2026-10-09';OUT.mkdir(exist_ok=True);torch.set_num_threads(2)
def polar(r):
 u,_,v=torch.linalg.svd(r);return u@v
def main():
 cfg,arr,_=p.prepare('cc3m-cc3m',OUT,32768,0)
 z=np.load(next(m['path'] for m in cfg['models'] if m['key']=='cca_256'));w={s:z[s] for s in ['image','text']}
 fit=[((arr['fit',s]-arr['mean',s])@torch.tensor(w[s],device='cuda',dtype=torch.float32)).detach() for s in w];del arr
 orient=[q.cpu().numpy() for q in fit];fit=[q[:8192] for q in fit];coef=[torch.tensor(v/np.sqrt((v*v).mean()),device='cuda',dtype=torch.float32) for v in w.values()]
 data,_=p.eval_population(cfg,'test',OUT);test={s:np.asarray(data['raw'][s]@(w[s]/data['scales'][s][:,None]))-data['means'][s]@(w[s]/data['scales'][s][:,None]) for s in w}
 def loss(r,kind):
  if kind=='coeff':return sum((m@r).abs().mean() for m in coef)/2
  vals=[x@r for x in fit]
  if kind=='output':return sum((v.abs().sum(1)/v.norm(dim=1).clamp_min(1e-8)).mean() for v in vals)/32
  vals=[(v-v.mean(0))/v.std(0,unbiased=False).clamp_min(1e-8) for v in vals]
  if kind=='robust':
   vals=[v.clamp(-torch.quantile(v.detach().abs(),.99,dim=0),torch.quantile(v.detach().abs(),.99,dim=0)) for v in vals]
  return -sum((v**4).mean()-3 for v in vals)/2
 def evaluate(r):
  rr=r.detach().cpu().numpy().astype(float);u,_,v=np.linalg.svd(rr);rr=u@v;skew=sum(np.mean((a@rr)**3,axis=0) for a in orient);rr*=np.where(skew<0,-1,1)
  au={};met={}
  for s in w:
   a=test[s]@rr;met[s]=dict(output_effective=float(np.mean(np.abs(a).sum(1)**2/(a*a).sum(1))),coefficient_effective=float(np.mean(np.abs(w[s]@rr).sum(0)**2/((w[s]@rr)**2).sum(0))))
   for h in ['a','b']:
    k=s+'_'+h;mask=data['masks'][k];au[k]=auc_matrix(a[mask],data['labels'][s][mask],chunk_size=64);au[k+'_variable']=variable_coordinates(a[mask])
  au['eligible']=np.isin([c['id'] for c in data['concepts']],data['metadata']['eligible_category_ids']);au['category_ids']=np.array([c['id'] for c in data['concepts']]);result=p.old.evaluation(au,{**cfg,'n_null':0},signed=False)
  return rr,dict(matching50=result['summary']['agree_at1_count'],metrics=met)
 schedules=[('coeff','coeff'),('output','output'),('coeff','output'),('output','coeff'),('kurt','kurt'),('robust','robust')]
 for seed in [0,1]:
  for first,second in schedules:
   name=f'{first}-{second}-{seed}';dest=OUT/(name+'.json')
   if dest.exists():continue
   torch.manual_seed(1200+seed);r=torch.eye(256,device='cuda') if seed==0 else polar(torch.randn(256,256,device='cuda'));hist=[];eta0=1.;t0=time.time()
   for step in range(301):
    kind=first if step<150 else second
    if step==150:eta0=1.
    if step%50==0:
     rr,row=evaluate(r);row.update(step=step,kind=kind,elapsed=time.time()-t0);hist.append(row);p.save(dest,dict(name=name,seed=seed,history=hist,completed=False));print(name,row,flush=True)
     np.savez_compressed(OUT/f'{name}-step{step}.npz',rotation=rr)
    if step==300:break
    r=r.detach().requires_grad_(True);f=loss(r,kind);g=torch.autograd.grad(f,r)[0];g=g-r@((r.T@g+g.T@r)/2);gn=float(g.norm());eta=min(eta0*1.5,10)/max(1.,gn)
    with torch.no_grad():
     accepted=False
     for _ in range(16):
      trial=polar(r-eta*g)
      if float(loss(trial,kind))<=float(f)-1e-4*eta*gn*gn:accepted=True;break
      eta*=.5
     if accepted:r=trial;eta0=min(eta*max(1.,gn),10)
     else:r=r.detach()
   p.save(dest,dict(name=name,seed=seed,history=hist,completed=True))
 p.save(OUT/'complete.json',dict(completed=True,protocol='Same two initializations, 300 gradient steps, switch at150, no annotation model selection. COCO val diagnostics every50. Robust fourth moment clips standardized output at per-coordinate99percent absolute value. Exploratory, not an independent confirmation.'))
if __name__=='__main__':main()
