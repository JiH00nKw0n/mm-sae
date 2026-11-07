"""Scale-invariant orthogonal optimization, fixed CC3M CCA and paired sample split.
No semantic labels in stopping/checkpoint selection. Empirical convergence, not global proof.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json,time,argparse,traceback
import numpy as np
import torch
import experiments.cca_unsupervised_rotations_20261009 as old
p=old.prior;ROOT=p.ROOT;OUT=ROOT/'runs/rotation-convergence-2026-10-09';OUT.mkdir(exist_ok=True);torch.set_num_threads(2)
START=time.time()
def status(**kw):p.save(OUT/'progress.json',dict(total_seconds=time.time()-START,**kw));print(json.dumps(kw),flush=True)
def prepare():
 cfg=p.old.load_config(ROOT/p.CONFIG['cc3m-cc3m']);z=np.load(next(m['path'] for m in cfg['models'] if m['key']=='cca_256'));w={s:z[s].astype(float) for s in ['image','text']}
 cache=OUT/'projected.npz'
 if not cache.exists():
  status(stage='prepare');cfg,arr,_=p.prepare('cc3m-cc3m',OUT,32768,0);saved={}
  for kind in ['fit','tune']:
   for s in w:saved[kind+'_'+s]=((arr[kind,s]-arr['mean',s])@torch.tensor(w[s],device='cuda',dtype=torch.float32)).cpu().numpy()
  np.savez(cache,**saved);del arr
 z=np.load(cache);orientation=[z['fit_'+s] for s in w];ts={kind:[torch.tensor(z[kind+'_'+s][:8192] if kind=='fit' else z[kind+'_'+s],device='cuda',dtype=torch.float64) for s in w] for kind in ['fit','tune']}
 with np.load(Path(cfg['parent_run'])/'moments.npz') as z:
  ni=len(z['image_ids']);mu=z['fit_mean'];cv=z['fit_second']-np.outer(mu,mu);sd=np.sqrt(np.maximum(cv.diagonal(),0));sd[sd==0]=1;cv/=sd[:,None]*sd[None,:]
 ts['coeff']=[];ts['coeff_row']=[];ts['load']=[]
 for s,c in [('image',cv[:ni,:ni]),('text',cv[ni:,ni:])]:
  ww=w[s];ts['coeff'].append(torch.tensor(ww/np.sqrt((ww**2).mean()),device='cuda',dtype=torch.float64));ts['coeff_row'].append(torch.tensor(ww/np.maximum(np.linalg.norm(ww,axis=1,keepdims=True),1e-12),device='cuda',dtype=torch.float64));ll=c@ww;ll/=np.maximum(np.linalg.norm(ll,axis=1,keepdims=True),1e-12);ts['load'].append(torch.tensor(ll,device='cuda',dtype=torch.float64))
 return cfg,w,ts,orientation
def objective(r,kind,ts,tune=False):
 if kind=='coefficient_l1':return sum((m@r).abs().mean() for m in ts['coeff'])/2
 if kind in ['coefficient_varimax','loading_varimax']:
  mats=ts['coeff_row'] if kind=='coefficient_varimax' else ts['load'];return -sum(((m@r).pow(4).mean(0)-(m@r).square().mean(0).square()).sum() for m in mats)/2
 a,b=[x@r for x in ts['tune' if tune else 'fit']]
 if kind=='output_l1':return (a.abs().sum(1)/a.norm(dim=1).clamp_min(1e-12)+b.abs().sum(1)/b.norm(dim=1).clamp_min(1e-12)).mean()/32
 a=a-a.mean(0);b=b-b.mean(0);a=a/a.square().mean(0).clamp_min(1e-20).sqrt();b=b/b.square().mean(0).clamp_min(1e-20).sqrt()
 if kind=='output_kurtosis':return -(a.pow(4).mean()+b.pow(4).mean()-6)/2
 if kind=='output_logcosh':
  lc=lambda z:z.abs()+torch.nn.functional.softplus(-2*z.abs())-np.log(2)
  return -((lc(a).mean(0)-.37456720749).square().mean()+(lc(b).mean(0)-.37456720749).square().mean())/2
 if kind=='shared_cokurtosis':return -(a.square()*b.square()).mean()+1+2*(a*b).mean(0).square().mean()
 raise ValueError(kind)
def fit(method,seed,ts,args):
 name=f'{method}-{seed}';dest=OUT/(name+'-fit.json')
 if dest.exists():return
 rank=getattr(args,'rank',256)
 torch.manual_seed(1200+seed);r=torch.eye(rank,device='cuda',dtype=torch.float64) if seed==0 else old.polar(torch.randn(rank,rank,device='cuda')).double();u,_,v=torch.linalg.svd(r);r=u@v
 eye=torch.eye(rank,device='cuda',dtype=torch.float64);history=[];trace=[];best=np.inf;alpha=.25;stop='maximum_steps';first_plateau=None;t0=time.time()
 with torch.no_grad():initial=float(objective(r,method,ts))
 scale=max(abs(initial),1e-12);last_accept=True
 for step in range(args.max_steps+1):
  r=r.detach().requires_grad_(True);f=objective(r,method,ts);rawg=torch.autograd.grad(f,r)[0];omega=(r.T@rawg-rawg.T@r)/2;gn=float(omega.norm());value=float(f);trace.append(value)
  if step%25==0 or step==args.max_steps:
   with torch.no_grad():tune=float(objective(r,method,ts,True))
   if tune<best:best=tune;best_step=step;best_r=r.detach().clone()
   row=dict(step=step,fit=value,tune=tune,gradient_relative=gn/scale,alpha=alpha,seconds=time.time()-t0);history.append(row)
   p.save(OUT/(name+'-history.json'),history)
   if step%250==0:status(method=method,seed=seed,stage='fit',**row)
  if gn/scale<1e-7:stop='relative_gradient';break
  if step>=400 and step%100==0:
   delta=(trace[-201]-value)/max(scale,abs(value),1e-12)
   if delta<1e-6:
    if first_plateau is None:first_plateau=step;alpha=.25
    elif step-first_plateau>=200:stop='plateau_confirmed_after_step_reset';break
   else:first_plateau=None
  if step==args.max_steps:break
  # Unit tangent direction makes updates invariant to a positive rescaling of the objective.
  direction=omega/max(gn,1e-30);trial_alpha=min(alpha*1.5,4.0);accepted=False
  with torch.no_grad():
   for attempt in range(28):
    trial=r@torch.linalg.solve(eye+.5*trial_alpha*direction,eye-.5*trial_alpha*direction);nv=float(objective(trial,method,ts))
    if nv<=value-1e-4*trial_alpha*gn:accepted=True;break
    trial_alpha*=.5
   if not accepted:stop='line_search_stalled';break
   r=trial;alpha=trial_alpha
 # Save both final training iterate and the checkpoint selected by its own label-free tune objective.
 for tag,rr in [('final',r),('tune',best_r)]:
  rr=rr.detach().cpu().numpy();a,_,b=np.linalg.svd(rr);rr=a@b;np.savez_compressed(OUT/(name+'-'+tag+'.npz'),rotation=rr)
 with torch.no_grad():final=float(objective(r,method,ts));final_tune=float(objective(r,method,ts,True))
 last_delta=(trace[max(0,len(trace)-201)]-final)/max(scale,abs(final),1e-12)
 p.save(dest,dict(method=method,seed=seed,steps=step,stop=stop,initial_fit=initial,final_fit=final,final_tune=final_tune,best_tune=best,best_step=best_step,relative_last200_improvement=last_delta,gradient_relative=gn/scale,seconds=time.time()-t0,global_optimum_claimed=False))
 status(method=method,seed=seed,stage='fitted',steps=step,stop=stop,seconds=time.time()-t0)
def main(args):
 methods=args.methods.split(',') if args.methods else old.METHODS;seeds=[int(i) for i in args.seeds.split(',')]
 p.save(OUT/'protocol.json',dict(condition='CC3M SAE / CC3M mapping',rank=256,fit_samples=8192,tune_samples=2048,methods=old.METHODS,seeds=[0,1,2],max_steps=args.max_steps,dtype='float64',optimizer='Unit-norm tangent gradient, Cayley retraction, Armijo backtracking. Invariant to positive objective scaling.',stopping='Relative gradient <1e-7 or last200 relative improvement <1e-6, followed by step reset and >=200 additional steps confirming plateau. Otherwise report line-search or budget limit.',selection='Final training iterate and own unlabelled TUNE checkpoint both evaluated. No semantic labels in optimization or selection.',limits='Nonconvex objectives. Plateau is empirical evidence, not proof of local/global optimum. L1 objectives nonsmooth. All evaluation exploratory.'))
 cfg,w,ts,orientation=prepare()
 for method in methods:
  for seed in seeds:fit(method,seed,ts,args)
 del ts;torch.cuda.empty_cache()
 status(stage='evaluation');data,parents=p.eval_population(cfg,'test',OUT);test={s:np.asarray(data['raw'][s]@(w[s]/data['scales'][s][:,None]))-data['means'][s]@(w[s]/data['scales'][s][:,None]) for s in w}
 presence=np.load(Path(cfg['source_run'])/'index/val2017/presence.npy').astype(bool);names=json.loads((ROOT/'runs/cca-171-category-names-2026-10-09/protocol.json').read_text())['prototype_texts'];raw=np.load(ROOT/'runs/cca-unsupervised-rotations-2026-10-09/cc3m-cc3m/name-activations.npy')
 with np.load(Path(cfg['parent_run'])/'moments.npz') as z:ti=z['text_ids']
 proto=raw[:,ti]@(w['text']/data['scales']['text'][:,None])-data['means']['text']@(w['text']/data['scales']['text'][:,None])
 for method in methods:
  for seed in seeds:
   for tag in ['final','tune']:
    name=f'{method}-{seed}-{tag}';rr=np.load(OUT/(name+'.npz'))['rotation'];old.evaluate(cfg,data,parents,test,proto,presence,names,rr,name,OUT,orientation)
    status(stage='evaluated',name=name)
 p.save(OUT/'invocation-complete.json',dict(methods=methods,seeds=seeds,seconds=time.time()-START))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--methods',default='');parser.add_argument('--seeds',default='0,1,2');parser.add_argument('--max-steps',type=int,default=6000);args=parser.parse_args()
 try:main(args)
 except Exception as e:p.save(OUT/'error.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
