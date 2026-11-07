"""Controlled linear nonnegative/cross-redundancy ablation on full FIT moments.
Not exact nonnegative multivariate CCA: hard within-view whitening is relaxed.
All learned controls share the SAME feasible positive initialization; labels only
used in final representative-agreement evaluation, not fitting/checkpoint choice.
"""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
from pathlib import Path
import argparse,json,time,hashlib,traceback
import numpy as np
import torch
import experiments.apbt_search_20261009 as prior
ROOT=prior.ROOT;BASE=ROOT/'runs/nonnegative-barlow-2026-10-09';prior.BASE=BASE
save=prior.save
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
SPECS=[
 dict(name='signed_correlation',positive=False,diagonal='linear',lam=0,support=None),
 dict(name='nonnegative_correlation',positive=True,diagonal='linear',lam=0,support=None),
 dict(name='signed_barlow',positive=False,diagonal='square',lam=.005,support=None),
 dict(name='nonnegative_diagonal',positive=True,diagonal='square',lam=0,support=None),
 dict(name='nonnegative_barlow',positive=True,diagonal='square',lam=.005,support=None),
 dict(name='nonnegative_barlow_16',positive=True,diagonal='square',lam=.005,support=16)]
def status(**v):
 save(BASE/'progress.json',dict(time_unix=time.time(),**v));print(json.dumps(v),flush=True)
def statistics(a,b,m):
 xx,xy,yy=m
 xa=xx@a;yb=yy@b
 vx=(a*xa).sum(0).clamp_min(1e-10);vy=(b*yb).sum(0).clamp_min(1e-10)
 c=(a.T@(xy@b))/torch.sqrt(vx[:,None]*vy[None,:])
 return c,vx,vy

def objective(c,spec):
 d=c.diag();off=c.square().sum()-d.square().sum();r=len(d)
 diagonal=(1-d).mean() if spec['diagonal']=='linear' else (1-d).square().mean()
 return diagonal+spec['lam']*off/r

def metrics(a,b,m):
 with torch.no_grad():
  c,vx,vy=statistics(a,b,m);d=c.diag();off=c[~torch.eye(len(d),device=c.device,dtype=torch.bool)]
  xx,_,yy=m
  within=[]
  for w,cov in [(a,xx),(b,yy)]:
   cov=w.T@cov@w;var=cov.diag().clamp_min(1e-10);corr=cov/torch.sqrt(var[:,None]*var[None,:]);within.append(float(len(d)**2/corr.square().sum()))
  return dict(mean_diag=float(d.mean()),diag_mse=float((1-d).square().mean()),off_rms=float(off.square().mean().sqrt()),off_max=float(off.abs().max()),variance_min=[float(vx.min()),float(vy.min())],correlation_effective_rank=within)

def matrices(cfg):
 with np.load(Path(cfg['parent_run'])/'moments.npz') as z:
  ni=len(z['image_ids']);mean=z['fit_mean'];scale=np.sqrt(np.maximum(np.diag(z['fit_second'])-mean**2,0));scale=np.where(scale>0,scale,1)
  result={}
  for part in ['fit','tune']:
   covariance=(z[part+'_second']-np.outer(z[part+'_mean'],z[part+'_mean']))/np.outer(scale,scale)
   covariance=(covariance+covariance.T)/2
   result[part]=tuple(torch.tensor(x.copy(),device='cuda',dtype=torch.float32) for x in [covariance[:ni,:ni],covariance[:ni,ni:],covariance[ni:,ni:]])
 return result

@torch.no_grad()
def project(weights,spec,previous):
 for w,prev in zip(weights,previous):
  if spec['positive']:w.clamp_(min=0)
  if spec['support']:
   idx=w.abs().topk(spec['support'],dim=0).indices;w.mul_(torch.zeros_like(w).scatter_(0,idx,1))
  dead=w.norm(dim=0)<1e-12
  if dead.any():w[:,dead]=prev[:,dead]
  w.div_(w.norm(dim=0,keepdim=True).clamp_min(1e-12))

def fit(condition,spec,initial,m,out,steps):
 path=out/(spec['name']+'.npz')
 if path.exists():return
 weights=[torch.nn.Parameter(torch.tensor(z,device='cuda',dtype=torch.float32)) for z in initial]
 project(weights,spec,weights);a,b=weights
 opt=torch.optim.Adam(weights,lr=.003);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=steps,eta_min=.0001)
 best=float('inf');hist=[];t=time.monotonic()
 for step in range(steps+1):
  if step:
   prev=[w.detach().clone() for w in weights]
   c,_,_=statistics(a,b,m['fit']);loss=objective(c,spec);assert torch.isfinite(loss)
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(weights,10);opt.step();project(weights,spec,prev);scheduler.step()
  if step%25==0 or step==steps:
   with torch.no_grad():
    fitloss=float(objective(statistics(a,b,m['fit'])[0],spec));tuneloss=float(objective(statistics(a,b,m['tune'])[0],spec))
   row=dict(step=step,fit_loss=fitloss,tune_loss=tuneloss,lr=opt.param_groups[0]['lr'],elapsed_seconds=time.monotonic()-t,fit=metrics(a,b,m['fit']),tune=metrics(a,b,m['tune']))
   hist.append(row)
   if tuneloss<best:
    best=tuneloss;beststep=step;selected=[w.detach().cpu().numpy().copy() for w in weights]
   save(out/(spec['name']+'-history.json'),hist);status(state='training',condition=condition,method=spec['name'],step=step,total_steps=steps,fit_loss=fitloss,tune_loss=tuneloss,elapsed_seconds=row['elapsed_seconds'])
 np.savez_compressed(path,image=selected[0],text=selected[1],centered=True)
 selected_gpu=[torch.tensor(w,device='cuda') for w in selected]
 save(out/(spec['name']+'-fit.json'),dict(spec=spec,steps=steps,best_step=beststep,selection='Lowest own objective on unlabelled mapping tune moments; no evaluation labels',seconds=time.monotonic()-t,fit=metrics(*selected_gpu,m['fit']),tune=metrics(*selected_gpu,m['tune']),global_optimum=False,final_checks_loss_change=hist[-1]['fit_loss']-hist[-2]['fit_loss']))

def main(condition,steps):
 out=BASE/condition;out.mkdir(parents=True,exist_ok=True);cfg=prior.old.load_config(ROOT/prior.CONFIG[condition]);m=matrices(cfg)
 cca=Path(next(x['path'] for x in cfg['models'] if x['key']=='cca_256'))
 with np.load(cca) as z:wa,wb=z['image'].copy(),z['text'].copy()
 # Resolve the joint sign by positive coefficient energy, without annotations.
 sign=np.where((np.maximum(wa,0)**2).sum(0)+(np.maximum(wb,0)**2).sum(0)>=(np.minimum(wa,0)**2).sum(0)+(np.minimum(wb,0)**2).sum(0),1,-1)
 initial=[np.maximum(w*sign,0) for w in [wa,wb]]
 for w in initial:
  dead=np.linalg.norm(w,axis=0)<1e-12
  if dead.any():w[:,dead]=1
  w/=np.maximum(np.linalg.norm(w,axis=0,keepdims=True),1e-12)
 np.savez_compressed(out/'positive_initialization.npz',image=initial[0],text=initial[1],centered=True)
 save(out/'protocol.json',dict(condition=condition,specs=SPECS,steps=steps,optimizer='Full FIT covariance gradient Adam lr.003 cosine decay to.0001, column L2 projection, gradient norm clip10',common_initialization='All learned variants use exactly same positive projection of sign-oriented CCA256. Top16 arm additionally projects support.',objective='linear=(sum_i 1-Cii)/r; square=sum_i(1-Cii)^2/r; off penalty=lambda*sum_i!=j Cij^2/r. Actual centered Pearson correlation.',baseline_caveat='CCA hard within-view whitening constraints are NOT retained. Nonnegative correlation arm is a relaxed correlation maximizer, not exact multivariate nonnegative CCA.',full_fit_statistics=True,tune_no_labels=True,test_used_for_selection=False,retrieval='All methods use FIT-centered standardized activations and unrescaled projection cosine. Output count256.',fixed_P='Identity on paired columns; no label permutation or annotation-based orientation.',seed=0,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 for spec in SPECS:fit(condition,spec,initial,m,out,steps)
 jobs=[('cca_256',cca),('sparse_cca_16',Path(next(x['path'] for x in cfg['models'] if x['key']=='sparse_cca_16'))),('positive_initialization',out/'positive_initialization.npz')]+[(s['name'],out/(s['name']+'.npz')) for s in SPECS]
 for name,path in jobs:
  with np.load(path) as z:w=[torch.tensor(z[s],device='cuda',dtype=torch.float32) for s in ['image','text']]
  save(out/(name+'-correlation.json'),dict(fit=metrics(*w,m['fit']),tune=metrics(*w,m['tune'])))
 del m;torch.cuda.empty_cache()
 data,parents=prior.eval_population(cfg,'test',out)
 save(out/'evaluation.json',dict(metadata=data['metadata'],eligible_ids=[c['id'] for c,e in zip(data['concepts'],data['eligible']) if e],text_labels='Own caption dictionary mentions, NOT inherited image labels'))
 for name,path in jobs:
  res=prior.evaluate(cfg,data,parents,name,path,out,'test');diag={}
  for mode in ['positive','signed']:
   rows=[r for r in res[mode]['per_category'] if r.get('status')=='ok'];matched=[r for r in rows if r['agree_at1']]
   diag[mode]=dict(distinct_matched_representatives=len({r['image_coordinate'] for r in matched}),matched_both_auc_ge_07=sum(min(r['image_auc'],r['text_auc'])>=.7 for r in matched))
  save(out/(name+'-diagnostics.json'),diag)
 save(out/'complete.json',dict(completed=True,time_unix=time.time()));status(state='condition_complete',condition=condition)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=500);p.add_argument('--condition',choices=list(prior.CONFIG));args=p.parse_args()
 try:
  for c in ([args.condition] if args.condition else list(prior.CONFIG)):main(c,args.steps)
  if not args.condition:save(BASE/'complete.json',dict(completed=True,time_unix=time.time()))
 except Exception as e:status(state='failed',error=repr(e),traceback=traceback.format_exc());raise
