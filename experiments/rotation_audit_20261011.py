"""Frozen PLS spaces: paired rotation, pruning, cross-statistics and optimizer audit."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import argparse,json,time,traceback,hashlib
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from experiments import apbt_search_20261009 as p
ROOT=p.ROOT;OUT=ROOT/'runs/rotation-audit-2026-10-11';OUT.mkdir(exist_ok=True)
SIDES=['image','text'];torch.set_num_threads(2);DT=torch.float64;DEV='cuda'
def save(path,d):p.save(path,d)
def status(**d):save(OUT/'progress.json',dict(time=time.time(),**d));print(json.dumps(d),flush=True)
def tensor(x):return torch.as_tensor(x,device=DEV,dtype=DT)
def polar(x):u,_,v=torch.linalg.svd(x,full_matrices=False);return u@v

def objective(r,ls,grams=None):
 vals=[]
 for i,l in enumerate(ls):
  z=l@r
  if grams is not None:
   z=z/(r*(grams[i]@r)).sum(0).clamp_min(1e-20).sqrt()[None,:]
   z=z/z.norm(dim=1,keepdim=True).clamp_min(1e-12)
  vals.append((z.pow(4).mean(0)-z.square().mean(0).square()).sum())
 return sum(vals)/2

def gradient(r,ls,grams=None):
 if grams is None:
  zs=[l@r for l in ls]
  val=sum((z.pow(4).mean(0)-z.square().mean(0).square()).sum() for z in zs)/2
  g=sum(2/l.shape[0]*(l.T@(z.pow(3)-z*z.square().mean(0))) for l,z in zip(ls,zs))
 else:
  r=r.detach().requires_grad_(True);val=objective(r,ls,grams);g=torch.autograd.grad(val,r)[0]
 omega=(r.T@g-g.T@r)/2
 return float(val.detach()),g.detach(),omega.detach()

def fit(rank,kind,solver,seed,ls,grams,args):
 name=f'{kind}-{solver}-{seed}';folder=OUT/f'rank{rank}';folder.mkdir(exist_ok=True)
 dest=folder/f'{name}-fit.json';weights=folder/f'{name}.npz'
 if dest.exists():return np.load(weights)['rotation']
 torch.manual_seed(1200+seed);r=torch.eye(rank,device=DEV,dtype=DT) if seed==0 else polar(torch.randn(rank,rank,device=DEV,dtype=DT))
 eye=torch.eye(rank,device=DEV,dtype=DT);alpha=.25;hist=[];t0=time.time();scale=max(abs(float(objective(r,ls,grams))),1e-12);stale=0;stop='budget';prev=None
 for step in range(args.steps+1):
  val,g,om=gradient(r,ls,grams);gn=float(om.norm());rel=gn/max(abs(val),scale,1e-12)
  if step%25==0:
   hist.append(dict(step=step,objective=val,gradient=gn,gradient_relative=rel,seconds=time.time()-t0));save(folder/f'{name}-history.json',hist)
  if step%250==0:status(stage='fit',rank=rank,method=name,step=step,objective=val,gradient_relative=rel,seconds=time.time()-t0)
  if step==6000:np.savez_compressed(folder/f'{name}-6000.npz',rotation=r.detach().cpu().numpy());save(folder/f'{name}-6000.json',dict(objective=val,gradient=gn,gradient_relative=rel))
  if rel<args.grad_tol:stop='gradient';break
  if prev is not None:
   change=abs(val-prev)/max(abs(val),scale,1e-12);stale=stale+1 if change<args.obj_tol else 0
  if stale>=50 and rel<1e-4:stop='objective_stable_and_small_gradient';break
  if step==args.steps:break
  prev=val;accepted=False
  with torch.no_grad():
   if solver=='svd':
    trial=polar(g);nv=float(objective(trial,ls,grams))
    accepted=nv>=val-1e-14 and float((trial-r).norm())>1e-12
   if not accepted:
    d=om/max(gn,1e-30);a=min(alpha*1.5,4.)
    for attempt in range(30):
     trial=r@torch.linalg.solve(eye-.5*a*d,eye+.5*a*d);nv=float(objective(trial,ls,grams))
     if nv>=val+1e-4*a*gn:accepted=True;break
     a*=.5
    alpha=a
   if not accepted:stop='line_search_stalled';break
   r=trial.detach()
 r=polar(r.detach());val,g,om=gradient(r,ls,grams)
 np.savez_compressed(weights,rotation=r.cpu().numpy())
 save(dest,dict(rank=rank,kind=kind,solver=solver,seed=seed,steps=step,stop=stop,objective=val,gradient=float(om.norm()),gradient_relative=float(om.norm())/max(abs(val),scale,1e-12),seconds=time.time()-t0,orthogonality=float((r.T@r-eye).abs().max()),global_optimum=False))
 return r.cpu().numpy()

def diag_stats(c):
 diag=c.diag();sq=c.square();energy=float(sq.sum())
 return dict(diagonal_row_max_fraction=float((diag>=c.max(1).values-1e-12).double().mean()),diagonal_abs_row_max_fraction=float((diag.abs()>=c.abs().max(1).values-1e-12).double().mean()),diagonal_squared_share=float(diag.square().sum())/max(energy,1e-30),diagonal_mean=float(diag.mean()),offdiagonal_rms=float(((sq.sum()-diag.square().sum())/(c.numel()-len(diag))).clamp_min(0).sqrt()),trace=float(diag.sum()))
def cross_stats(a,b,cv):
 cc=a.T@cv[0]@b;vi=(a*(cv[1]@a)).sum(0);vt=(b*(cv[2]@b)).sum(0)
 corr=cc/(vi.clamp_min(1e-20).sqrt()[:,None]*vt.clamp_min(1e-20).sqrt()[None,:])
 return dict(covariance=diag_stats(cc),correlation=diag_stats(corr)),cc.cpu().numpy(),corr.cpu().numpy()

def gpu_auc(sc,labels):
 n,k=sc.shape;pos=labels.sum(0);assert bool(((pos>0)&(pos<n)).all())
 v,ix=torch.sort(sc,dim=0,stable=True);r=torch.arange(n,device=DEV)[:,None].expand(n,k)
 start=torch.cat([torch.ones((1,k),device=DEV,dtype=torch.bool),v[1:]!=v[:-1]],0)
 end=torch.cat([v[:-1]!=v[1:],torch.ones((1,k),device=DEV,dtype=torch.bool)],0)
 first=torch.cummax(torch.where(start,r,0),dim=0).values
 last=torch.flip(torch.cummin(torch.flip(torch.where(end,r,n-1),[0]),dim=0).values,[0])
 ranks=torch.empty_like(sc);ranks.scatter_(0,ix,(first+last).to(DT)/2+1)
 auc=(ranks.T@labels-pos*(pos+1)/2)/(pos*(n-pos))
 variable=v[-1]!=v[0];auc[~variable]=-torch.inf
 return auc

def matching(scores,labels,splits):
 rows=[]
 for a,b in splits:
  ia=gpu_auc(scores['image'][a],labels['image'][a]);ta=gpu_auc(scores['text'][b],labels['text'][b])
  valid=torch.isfinite(ia)&torch.isfinite(ta);ia[~valid]=-torch.inf;ta[~valid]=-torch.inf
  assert bool(valid.any(0).all());ii=ia.argmax(0);tt=ta.argmax(0)
  rows.append(dict(count=int((ii==tt).sum()),per_category=(ii==tt).cpu().numpy().astype(int).tolist(),image_coordinates=ii.cpu().tolist(),text_coordinates=tt.cpu().tolist()))
 return rows

def sparse16(w):
 ix=w.abs().topk(16,dim=0).indices;mask=torch.zeros_like(w).scatter_(0,ix,1);return w*mask

def evaluate(name,rank,w,r,fitout,evalx,labels,splits,parents,cv,base_ret,prune=True):
 folder=OUT/f'rank{rank}';dest=folder/f'{name}-result.json'
 if dest.exists() and json.loads(dest.read_text()).get('metric_version')==2:return
 if r is not None:
  rr=tensor(r);skew=sum((o@rr).pow(3).mean(0) for o in fitout);rr=rr*torch.where(skew<0,-1.,1.)[None,:]
  ws={s:w[s]@rr for s in SIDES};np.savez_compressed(folder/f'{name}-oriented.npz',rotation=rr.cpu().numpy())
 else:ws=w
 results={}
 for typ in (['dense','top16'] if prune else ['top16']):
  aw={s:sparse16(ws[s]) if typ=='top16' else ws[s] for s in SIDES}
  scores={s:evalx[s]@aw[s] for s in SIDES}
  status(stage='evaluate',rank=rank,method=name,variant=typ)
  match=matching(scores,labels,splits)
  if typ=='dense' and r is not None:ret=base_ret
  else:
   ret=p.paired_retrieval(scores['image'].cpu().numpy(),scores['text'].cpu().numpy(),parents,device='cuda',chunk_size=128)
   ret={d:{k:v for k,v in z.items() if k not in ['hits','ranks']} for d,z in ret.items()}
  stats,cc,corr=cross_stats(aw['image'],aw['text'],cv)
  # Held-out COCO cross-statistics pair image with each of its own captions.
  uu=scores['image'][torch.as_tensor(parents,device=DEV)];vv=scores['text'];uu=uu-uu.mean(0);vv=vv-vv.mean(0);ec=uu.T@vv/len(uu);er=ec/(uu.square().mean(0).sqrt()[:,None]*vv.square().mean(0).sqrt()[None,:]).clamp_min(1e-20)
  np.savez_compressed(folder/f'{name}-{typ}-cross.npz',fit_covariance=cc,fit_correlation=corr,eval_covariance=ec.cpu().numpy(),eval_correlation=er.cpu().numpy())
  results[typ]=dict(matching=match,retrieval=ret,fit_cross=stats,eval_cross=dict(covariance=diag_stats(ec),correlation=diag_stats(er)),nonzero={s:dict(min=int((aw[s]!=0).sum(0).min()),max=int((aw[s]!=0).sum(0).max())) for s in SIDES})
 save(dest,dict(name=name,rank=rank,metric_version=2,variants=results))

def main(args):
 cfg=p.old.load_config(ROOT/p.CONFIG['cc3m-cc3m']);save(OUT/'protocol.json',dict(condition='CC3M SAE / CC3M mapping; COCO val2017 retrieval and own-caption fixed50 matching',ranks=[128,256,512],seeds=list(range(args.seeds)),random_partitions=args.splits,seed_control='Identity seed0 plus shared random orthogonal initializations 1201..1209 for every method and solver.',sparsity='Top16 absolute coefficients per column, both sides, no refit or renormalization; rank-matched Sparse CCA16 sequential solver.',selection='No labels used for rotation optimization or stopping, all seeds reported.',confidence='Paired crossed bootstrap over seeds and random image partitions, conditional on fixed validation images and 50 categories. Also report split spread; not an untouched-population confidence guarantee.',equivalence='A CI covering zero is not equivalence. Practical margin 1 matched category in 50, report whether entire CI is within [-1,1].',convergence=dict(gradient_relative=args.grad_tol,obj_relative=args.obj_tol,max_steps=args.steps),objectives='Mean of two modality-specific Kaiser-row-normalized Varimax criteria. Covariance and coefficients fixed matrices. Correlation recomputes output variance and row normalization each iteration.',solver='SVD/polar fixed-point Varimax update adapted to two-block equal-weight objective; monotonicity safeguarded with Cayley steps; paired Cayley control.',sources=['https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/decomposition/_factor_analysis.py','https://doi.org/10.1007/BF02289233']))
 status(stage='load')
 with np.load(Path(cfg['parent_run'])/'moments.npz') as z:
  ni=len(z['image_ids']);mu=z['fit_mean'];cov=z['fit_second']-np.outer(mu,mu)
 sd=np.sqrt(np.maximum(cov.diagonal(),0));sd[sd==0]=1;cov/=sd[:,None]*sd[None,:]
 cv=[tensor(cov[:ni,ni:]),tensor(cov[:ni,:ni]),tensor(cov[ni:,ni:])]
 data,parents=p.eval_population(cfg,'test',OUT)
 eligible=np.isin([c['id'] for c in data['concepts']],data['metadata']['eligible_category_ids']);assert eligible.sum()==50
 labels={s:tensor(data['labels'][s][:,eligible].astype(float)) for s in SIDES}
 evalx={s:tensor((data['raw'][s].toarray()-data['means'][s])/data['scales'][s]) for s in SIDES}
 splits=[];splitarrays={}
 for seed in range(args.splits):
  rng=np.random.default_rng(91000+seed);ix=rng.permutation(len(evalx['image']));mask=np.zeros(len(ix),bool);mask[ix[:len(ix)//2]]=True
  a=np.flatnonzero(mask);b=np.flatnonzero(~mask[parents]);splits.append((torch.as_tensor(a,device=DEV),torch.as_tensor(b,device=DEV)));splitarrays[f'image_{seed}']=a;splitarrays[f'text_{seed}']=b
 # Previous fixed split and swap are extra rows, not included in resampled-mean inference.
 for ai,bt in [('image_a','text_b'),('image_b','text_a')]:splits.append((torch.as_tensor(np.flatnonzero(data['masks'][ai]),device=DEV),torch.as_tensor(np.flatnonzero(data['masks'][bt]),device=DEV)))
 np.savez_compressed(OUT/'evaluation-splits.npz',**splitarrays,category_ids=np.array([c['id'] for c in data['concepts']])[eligible])
 cache=OUT/'orientation-512.npz'
 with np.load(ROOT/'runs/mapping-ablation-cc3m-2026-10-04/transforms/cross_svd_3221.npz') as z:fullw={s:z[s][:,:512].astype(float) for s in SIDES}
 if not cache.exists():
  _,arr,_=p.prepare('cc3m-cc3m',OUT,32768,0)
  oo={s:((arr['fit',s]-arr['mean',s]).double()@tensor(fullw[s])).cpu().numpy() for s in SIDES};np.savez_compressed(cache,**oo);del arr
 orient=np.load(cache)
 for rank in args.ranks:
  folder=OUT/f'rank{rank}';folder.mkdir(exist_ok=True);w={s:tensor(fullw[s][:,:rank]) for s in SIDES};fitout=[tensor(orient[s][:,:rank]) for s in SIDES]
  loads=[cv[i+1]@w[s] for i,s in enumerate(SIDES)];grams=[w[s].T@loads[i] for i,s in enumerate(SIDES)]
  basepath=ROOT/f'runs/pls-cca-rank-subspace-2026-10-10/pls-native-{rank}-result.json';ret=json.loads(basepath.read_text())['retrieval']
  evaluate('unrotated',rank,w,np.eye(rank),fitout,evalx,labels,splits,parents,cv,ret)
  methods=[('covariance','svd'),('correlation','cayley'),('coefficient','svd'),('covariance','cayley')]
  for seed in range(args.seeds):
   for kind,solver in methods:
    if kind=='correlation':ls=loads;gs=grams
    else:
     raw=loads if kind=='covariance' else [w[s] for s in SIDES];ls=[l/l.norm(dim=1,keepdim=True).clamp_min(1e-12) for l in raw];gs=None
    r=fit(rank,kind,solver,seed,ls,gs,args)
    evaluate(f'{kind}-{solver}-{seed}',rank,w,r,fitout,evalx,labels,splits,parents,cv,ret)
  # Per-component rotations are compared after maximum-overlap signed permutation.
  stability={}
  for kind,solver in methods:
   rs=[np.load(folder/f'{kind}-{solver}-{seed}.npz')['rotation'] for seed in range(args.seeds)];pairs=[]
   for i in range(len(rs)):
    for j in range(i+1,len(rs)):
     sim=np.abs(rs[i].T@rs[j]);rr,cc=linear_sum_assignment(-sim);v=sim[rr,cc]
     pairs.append(dict(seed_i=i,seed_j=j,mean=float(v.mean()),median=float(np.median(v)),fraction_over_095=float((v>=.95).mean())))
   stability[f'{kind}-{solver}']=pairs
  save(folder/'stability.json',stability)
 status(stage='completed_rotation_and_evaluation')
 save(OUT/'complete.json',dict(completed=True,time=time.time()))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--seeds',type=int,default=10);a.add_argument('--splits',type=int,default=20);a.add_argument('--steps',type=int,default=30000);a.add_argument('--grad-tol',type=float,default=1e-6);a.add_argument('--obj-tol',type=float,default=1e-10);a.add_argument('--ranks',type=int,nargs='+',default=[128,256,512]);args=a.parse_args()
 try:main(args)
 except Exception as e:save(OUT/'error.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
