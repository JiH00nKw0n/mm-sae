"""One exploratory full-space joint fit, lambda=1, rank128, fixed CC3M/CC3M."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import time,json
from pathlib import Path
import numpy as np
import torch
from experiments import rotation_convergence_20261009 as conv
from mm_sae.analysis.semantic_evaluation import auc_matrix
from mm_sae.analysis.representative_agreement import variable_coordinates
from mm_sae.analysis.mapping_evaluation import paired_retrieval
p=conv.p;ROOT=p.ROOT;OUT=ROOT/'runs/joint-covariance-warm-2026-10-10';OUT.mkdir(exist_ok=True)
cfg=p.old.load_config(ROOT/p.CONFIG['cc3m-cc3m']);rank=128;sides=['image','text'];start=time.time();torch.set_num_threads(2)
with np.load(Path(cfg['parent_run'])/'moments.npz') as z:
 ni=len(z['image_ids']);mu=z['fit_mean'];cv=z['fit_second']-np.outer(mu,mu)
sd=np.sqrt(np.maximum(cv.diagonal(),0));sd[sd==0]=1;cv/=sd[:,None]*sd[None,:]
with np.load(ROOT/'runs/mapping-ablation-cc3m-2026-10-04/transforms/cross_svd_3221.npz') as z:w0={s:z[s][:,:rank].astype(float) for s in sides}
xx=torch.tensor(cv[:ni,:ni],device='cuda',dtype=torch.float64);yy=torch.tensor(cv[ni:,ni:],device='cuda',dtype=torch.float64);xy=torch.tensor(cv[:ni,ni:],device='cuda',dtype=torch.float64)
a=torch.tensor(w0['image'],device='cuda');b=torch.tensor(w0['text'],device='cuda');fmax=float((a*(xy@b)).sum())
def components(a,b):
 f=(a*(xy@b)).sum()/fmax
 j=[]
 for cov,w in [(xx,a),(yy,b)]:
  load=cov@w;load=load/load.norm(dim=1,keepdim=True).clamp_min(1e-12)
  j.append((load.pow(4).mean(0)-load.square().mean(0).square()).sum())
 return f,sum(j)/2

def retract(w):
 q,r=torch.linalg.qr(w,mode='reduced');return q*torch.where(r.diag()<0,-1.,1.)
p.save(OUT/'protocol.json',dict(condition='CC3M SAE / CC3M mapping',rank=rank,initialization='Existing two-stage seed0 solution. Warm-start diagnostic, not a one-stage end-to-end fit.',objective='Maximize F/Fmax + lambda J, lambda fixed1 without validation selection. F=trace(A.T C B), J=same row-normalized covariance-loading Varimax as rotation. J is bounded above by1 under this normalization.',constraints='A.T A=B.T B=I; Euclidean Stiefel tangent gradient, QR retraction, Armijo line search. Full original activation dimensions, not restricted to SVD subspace.',comparison='Two-stage retains maximal F and optimizes shared orthogonal rotation; joint permits reduction in F. This compares different objectives and feasible freedoms, not solely a numerical-solver change.',max_steps=1500,stopping='Tangent gradient norm<1e-7 or relative last100 objective change<1e-7 at two checks separated by100 steps. No labels for fitting or stopping. Final iterate evaluated.',limits='One initialization and one lambda; exploratory, no global optimum or best tradeoff claim. SVD cost excluded from both fitting times.'))
warm=ROOT/'runs/matched-statistic-rotation128-2026-10-10/pls-128/0-transforms.npz'
with np.load(warm) as z:a=torch.tensor(z['image'],device='cuda');b=torch.tensor(z['text'],device='cuda')
p.save(OUT/'initialization.json',dict(source=str(warm),reason='Two-stage solution has a higher joint objective than the SVD-initialized joint iterate. Check optimization failure separately from objective tradeoff.'))
if (OUT/'joint-final.npz').exists():
 z=np.load(OUT/'joint-final.npz');a=torch.tensor(z['image'],device='cuda');b=torch.tensor(z['text'],device='cuda')
else:
 history=[];trace=[];alpha=.25;stop='maximum_steps';plateau=None;t0=time.time()
 for step in range(1501):
  a=a.detach().requires_grad_(True);b=b.detach().requires_grad_(True);f,j=components(a,b);loss=-f-j
  ga,gb=torch.autograd.grad(loss,(a,b));ga=ga-a@((a.T@ga+ga.T@a)/2);gb=gb-b@((b.T@gb+gb.T@b)/2)
  gn=float(torch.sqrt(ga.square().sum()+gb.square().sum()));value=float(loss);trace.append(value)
  if step%25==0:
   row=dict(step=step,covariance_fraction=float(f),concentration=float(j),loss=value,gradient_norm=gn,seconds=time.time()-t0,alpha=alpha);history.append(row);p.save(OUT/'history.json',history);p.save(OUT/'progress.json',row);print(json.dumps(row),flush=True)
  if gn<1e-7:stop='gradient';break
  if step>=200 and step%100==0 and abs(trace[-101]-value)/max(abs(value),1e-12)<1e-7:
   if plateau is not None and step-plateau>=100:stop='plateau';break
   plateau=step
  else:
   if step%100==0:plateau=None
  if step==1500:break
  trial_alpha=min(alpha*1.5,4.);accepted=False
  with torch.no_grad():
   for attempt in range(28):
    aa=retract(a-trial_alpha*ga/max(gn,1e-30));bb=retract(b-trial_alpha*gb/max(gn,1e-30));ff,jj=components(aa,bb)
    if float(-ff-jj)<=value-1e-4*trial_alpha*gn:accepted=True;break
    trial_alpha*=.5
   if not accepted:stop='line_search_stalled';break
   a,b=aa,bb;alpha=trial_alpha
 with torch.no_grad():f,j=components(a,b)
 np.savez_compressed(OUT/'joint-final.npz',image=a.detach().cpu().numpy(),text=b.detach().cpu().numpy())
 p.save(OUT/'fit.json',dict(steps=step,stop=stop,seconds=time.time()-t0,covariance_fraction=float(f),concentration=float(j),gradient_norm=gn,global_optimum_claimed=False))
# Wait only for the already-running matched experiment's r128 PLS seed0 result.
two=ROOT/'runs/matched-statistic-rotation128-2026-10-10/pls-128'
while not (two/'0-transforms.npz').exists():time.sleep(5)
models={'joint':{s:np.load(OUT/'joint-final.npz')[s] for s in sides},'two_stage':{s:np.load(two/'0-transforms.npz')[s] for s in sides}}
with torch.no_grad():
 stats={}
 for name,w in models.items():
  aa=torch.tensor(w['image'],device='cuda');bb=torch.tensor(w['text'],device='cuda');f,j=components(aa,bb)
  stats[name]=dict(covariance_fraction=float(f),concentration=float(j),joint_objective=float(f+j),orthogonality_max=max(float((q.T@q-torch.eye(rank,device='cuda')).abs().max()) for q in [aa,bb]))
 p.save(OUT/'objective_comparison.json',stats)
del a,b,xx,yy,xy,aa,bb;torch.cuda.empty_cache()
_,arr,_=p.prepare('cc3m-cc3m',OUT,32768,0)
w=models['joint'];fit={s:((arr['fit',s]-arr['mean',s])@torch.tensor(w[s],device='cuda',dtype=torch.float32)).cpu().numpy().astype(float) for s in sides};del arr;torch.cuda.empty_cache()
sign=np.where(sum(np.mean(fit[s]**3,axis=0) for s in sides)<0,-1.,1.);w={s:w[s]*sign for s in sides}
np.savez_compressed(OUT/'joint-oriented.npz',**w)
data,parents=p.eval_population(cfg,'test',OUT)
scores={s:np.asarray(data['raw'][s]@(w[s]/data['scales'][s][:,None]))-data['means'][s]@(w[s]/data['scales'][s][:,None]) for s in sides}
arr={}
for s in sides:
 for half in ['a','b']:
  key=s+'_'+half;mask=data['masks'][key];arr[key]=auc_matrix(scores[s][mask],data['labels'][s][mask],chunk_size=32);arr[key+'_variable']=variable_coordinates(scores[s][mask])
arr['eligible']=np.isin([c['id'] for c in data['concepts']],data['metadata']['eligible_category_ids']);assert arr['eligible'].sum()==50
arr['category_ids']=np.array([c['id'] for c in data['concepts']]);normal=p.old.evaluation(arr,{**cfg,'n_null':0},signed=False)
swapped=dict(arr)
for s in sides:
 for suffix in ['','_variable']:swapped[s+'_a'+suffix]=arr[s+'_b'+suffix];swapped[s+'_b'+suffix]=arr[s+'_a'+suffix]
repeat=p.old.evaluation(swapped,{**cfg,'n_null':0},signed=False)
ret=paired_retrieval(scores['image'],scores['text'],parents,device='cuda',chunk_size=128)
p.save(OUT/'result.json',dict(matching50=normal,repeat_matching50=repeat,retrieval={d:{k:v for k,v in o.items() if k not in ['hits','ranks']} for d,o in ret.items()}))
p.save(OUT/'complete.json',dict(complete=True,seconds=time.time()-start))
print('COMPLETE',normal['summary']['agree_at1_count'],repeat['summary']['agree_at1_count'],flush=True)
