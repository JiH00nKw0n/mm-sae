"""Staged fixed-subspace sparsification and shared orthogonal sparse rotation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import argparse,json,time,traceback
from pathlib import Path
import numpy as np,torch
from experiments.rotation_mass_audit_20261011 import Evaluator,ROOT,SRC,a,p
OUT=ROOT/'runs/sparse-improvement-2026-10-11';OUT.mkdir(exist_ok=True)
S=['image','text'];BUDGETS=[16,32,64,128,256];MASS=[.9,.95,.99]
def save(path,z):p.save(path,z)
def status(**z):save(OUT/'progress.json',dict(time=time.time(),**z));print(json.dumps(z),flush=True)
def hard(w,k):
 ix=torch.argsort(w.abs(),dim=0,descending=True,stable=True)[:k];return torch.zeros_like(w).scatter_(0,ix,w.gather(0,ix))
def score_prune(w,score,k=None,fraction=None):
 ix=torch.argsort(score,dim=0,descending=True,stable=True)
 if k is None:
  val=score.gather(0,ix);num=(val.cumsum(0)<fraction*val.sum(0)).sum(0)+1
 else:num=torch.full((w.shape[1],),k,device='cuda')
 mask=torch.arange(len(w),device='cuda')[:,None]<num;return w*torch.zeros_like(w).scatter_(0,ix,mask.double())
def retrieval(sc,parents):
 # Same cosine normalization, float32 dot products and lower-index tie rule as legacy evaluator.
 norms={s:sc[s].norm(dim=1) for s in S};x={s:(sc[s]/norms[s].clamp_min(1e-30)[:,None]).float() for s in S};sim=x['image']@x['text'].T
 iv=norms['image']>0;tv=norms['text']>0;sim[~iv,:]=-torch.inf;sim[:,~tv]=-torch.inf
 ni,nt=sim.shape;par=torch.as_tensor(parents,device='cuda');ti=torch.arange(nt,device='cuda');ii=torch.arange(ni,device='cuda');positive=sim[par,ti]
 best=torch.full((ni,),-torch.inf,device='cuda').scatter_reduce_(0,par,positive,reduce='amax',include_self=True)
 bestidx=torch.full((ni,),nt,device='cuda',dtype=torch.long).scatter_reduce_(0,par,torch.where(positive==best[par],ti,nt),reduce='amin',include_self=True)
 ir=1+(sim>best[:,None]).sum(1)+((sim==best[:,None])&(ti[None,:]<bestidx[:,None])).sum(1)
 tr=1+(sim>positive[None,:]).sum(0)+((sim==positive[None,:])&(ii[:,None]<par[None,:])).sum(0)
 ir[~iv|~torch.isfinite(best)]=nt+1;tr[~tv|~iv[par]]=ni+1
 return {d:dict(recall={str(k):float((r<=k).double().mean()) for k in [1,5,10]}) for d,r in [('image_to_text',ir),('text_to_image',tr)]}
class Eval(Evaluator):
 def __init__(self):
  super().__init__()
  with np.load(Path(self.cfg['parent_run'])/'moments.npz') as z:
   n=len(z['image_ids']);mu=z['fit_mean'];cv=z['fit_second']-np.outer(mu,mu)
  sd=np.sqrt(np.maximum(cv.diagonal(),0));sd[sd==0]=1;cv/=sd[:,None]*sd[None,:]
  self.cov={s:a.tensor(v) for s,v in [('image',cv[:n,:n]),('text',cv[n:,n:])]};self.cross=a.tensor(cv[:n,n:]);self.base=self.matrices(ROOT/'runs/mapping-ablation-cc3m-2026-10-04/transforms/cross_svd_3221.npz',256)
  # Validate accelerated retrieval against original frozen dense result before using it.
  dense=retrieval({s:self.x[s]@self.base[s] for s in S},self.parents);old=json.loads((ROOT/'runs/rotation-mass-audit-2026-10-11/rank256/pls-dense.json').read_text())['retrieval']
  err=max(abs(dense[d]['recall'][str(k)]-old[d]['recall'][str(k)]) for d in dense for k in [1,5,10]);assert err<1e-12,err
  save(OUT/'retrieval-validation.json',dict(max_recall_difference=err,dense=dense))
 def oriented_rotation(self,r):
  if not hasattr(self,'orientation_cache'):
   with np.load(SRC/'orientation-512.npz') as fit:self.orientation_cache={s:a.tensor(fit[s][:,:256]) for s in S}
  # Saved projections are unrotated PLS scores; orientation uses CC3M, never labels.
  skew=sum((self.orientation_cache[s]@r).pow(3).mean(0) for s in S);return r*torch.where(skew<0,-1.,1.)
 def evaluate(self,stage,name,seed,variant,ws,original=None,meta=None):
  folder=OUT/stage;folder.mkdir(exist_ok=True);dest=folder/f'{name}-s{seed}-{variant}.json'
  if dest.exists():return
  t=time.time();sc={s:self.x[s]@ws[s] for s in S};rows=[]
  for ai,bt in self.splits[:20]:
   au=[a.gpu_auc(sc['image'][ai],self.labels['image'][ai]),a.gpu_auc(sc['text'][bt],self.labels['text'][bt])];valid=torch.isfinite(au[0])&torch.isfinite(au[1])
   for q in au:q[~valid]=-torch.inf
   idx=[q.argmax(0) for q in au];col=torch.arange(50,device='cuda');va=[q[ind,col] for q,ind in zip(au,idx)];match=idx[0]==idx[1];row=dict(count=int(match.sum()),per_category=match.int().cpu().tolist())
   for threshold in [.7,.8,.9]:row[str(threshold)]=int((match&(va[0]>=threshold)&(va[1]>=threshold)).sum())
   rows.append(row)
  ret=retrieval(sc,self.parents);stats={}
  for s in S:
   cnt=(ws[s]!=0).sum(0).double();stats[s]=dict(mean=float(cnt.mean()),min=int(cnt.min()),max=int(cnt.max()))
   if original is not None:
    w=original[s];den=(w*(self.cov[s]@w)).sum(0);var=(ws[s]*(self.cov[s]@ws[s])).sum(0)/den;diff=w-ws[s];err=(diff*(self.cov[s]@diff)).sum(0)/den
    stats[s].update(variance_ratio_mean=float(var.mean()),variance_ratio_min=float(var.min()),variance_ratio_max=float(var.max()),relative_output_mse_mean=float(err.mean()))
  save(dest,dict(method=name,seed=seed,variant=variant,counts=stats,matching=rows,retrieval=ret,meta=meta or {},seconds=time.time()-t));status(stage=stage,method=name,seed=seed,variant=variant,seconds=time.time()-t)
 def stage1(self):
  for method in ['covariance-svd','correlation-cayley']:
   for seed in range(10):
    r=a.tensor(np.load(SRC/f'rank256/{method}-{seed}-oriented.npz')['rotation']);ws={s:self.base[s]@r for s in S}
    err=max(float(((ws[s]/a.tensor(self.data['scales'][s])[:,None])*a.tensor(self.data['scales'][s])[:,None]-ws[s]).abs().max()) for s in S)
    assert err<1e-12
    self.evaluate('stage1',method,seed,'dense',ws)
    for criterion in ['coefficient','variance_contribution']:
     scores={s:ws[s].square() if criterion=='coefficient' else (ws[s]*(self.cov[s]@ws[s])).abs() for s in S}
     for frac in MASS:
      ww={s:score_prune(ws[s],scores[s],fraction=frac) for s in S};self.evaluate('stage1',method,seed,f'{criterion}-mass{round(100*frac)}',ww,ws,dict(sd_equivalence_error=err))
     for k in BUDGETS:
      ww={s:score_prune(ws[s],scores[s],k=k) for s in S};self.evaluate('stage1',method,seed,f'{criterion}-k{k}',ww,ws)
  save(OUT/'stage1-complete.json',dict(time=time.time()))
 def stage2(self):
  for seed in range(10):
   for k in BUDGETS:
    r,fit=fit_alternating(self.base,k,seed);r=self.oriented_rotation(r);ws={s:self.base[s]@r for s in S}
    self.evaluate('stage2','alternating',seed,f'k{k}',{s:hard(ws[s],k) for s in S},ws,fit)
    dense_err=float((r.T@r-torch.eye(256,device='cuda')).abs().max());assert dense_err<1e-8
   r,fit=fit_l1(self.base,seed);r=self.oriented_rotation(r);ws={s:self.base[s]@r for s in S}
   for k in BUDGETS:self.evaluate('stage2','l1-rotation',seed,f'k{k}',{s:hard(ws[s],k) for s in S},ws,fit)
  # Sparse CCA with actual support k, deterministic solver, never ten duplicate models.
  self.sparse_baselines()
  save(OUT/'stage2-complete.json',dict(time=time.time()))
 def sparse_baselines(self):
  from mm_sae.analysis.sparse_cca import fit_sparse_cca
  dense=self.matrices(ROOT/'runs/mapping-ablation-cc3m-2026-10-04/transforms/cca_3221.npz',256)
  for k in BUDGETS:
   choices=[SRC/'sparse-cca16-512.npz'] if k==16 else []
   choices +=[ROOT/f'runs/cc3m-followup-2026-10-05/cc3m-fit/semantics/sparse-fit/sparse_cca_k{k}.npz',ROOT/f'runs/sparsity-sweep-2026-10-06/cc3m-cc3m/cca_{k}.npz',OUT/f'sparse-cca-k{k}.npz']
   source=next((f for f in choices if f.exists()),None)
   if source is None:
    status(stage='sparse_baseline_fit',k=k)
    model=fit_sparse_cca(self.cov['image'].cpu().numpy(),self.cross.cpu().numpy(),self.cov['text'].cpu().numpy(),256,k=k,ridge=.01,initial_image=dense['image'].cpu().numpy(),initial_text=dense['text'].cpu().numpy(),progress=lambda q:status(stage='sparse_baseline_fit',k=k,component=q['component']))
    source=OUT/f'sparse-cca-k{k}.npz';np.savez_compressed(source,image=model.image,text=model.text)
   w=self.oriented(self.matrices(source,256));self.evaluate('stage2','sparse-cca',0,f'k{k}',w,meta=dict(source=str(source)))
  # Hungarian uses existing fixed coefficient and matching results, threshold counts retained.
  h=json.loads((ROOT/'runs/hungarian-rotation-diagnosis-2026-10-11/hungarian-256.json').read_text());rows=[]
  for q in h['rows']:
   mm=np.array(q['per_category']);v=[np.array(q[s]['best_auc']) for s in S];row=dict(count=q['count'],per_category=q['per_category'])
   for t in [.7,.8,.9]:row[str(t)]=int((mm*(v[0]>=t)*(v[1]>=t)).sum())
   rows.append(row)
  save(OUT/'stage2/hungarian-s0-k1.json',dict(method='hungarian',seed=0,variant='k1',matching=rows,counts={s:dict(mean=1,min=1,max=1) for s in S},retrieval=h['retrieval']))
def initial(seed):
 torch.manual_seed(1200+seed);return torch.eye(256,device='cuda',dtype=torch.float64) if seed==0 else a.polar(torch.randn(256,256,device='cuda',dtype=torch.float64))
def fit_alternating(base,k,seed):
 folder=OUT/'fits';folder.mkdir(exist_ok=True);dest=folder/f'alternating-k{k}-s{seed}.npz';meta=dest.with_suffix('.json')
 if dest.exists():return a.tensor(np.load(dest)['rotation']),json.loads(meta.read_text())
 r=initial(seed);hist=[];start=time.time();lastmask=None;stop='iteration_limit'
 for step in range(1,2001):
  z={s:base[s]@r for s in S};sp={s:hard(z[s],k) for s in S};m=sum(base[s].T@sp[s] for s in S);rn=a.polar(m)
  old=sum((z[s]-sp[s]).square().sum() for s in S)/2
  zz={s:base[s]@rn for s in S};val=sum((zz[s]-hard(zz[s],k)).square().sum() for s in S)/2
  assert float(val)<=float(old)+1e-8
  hist.append(float(val));delta=float((rn-r).norm());r=rn
  if step>10 and abs(hist[-1]-hist[-2])<=1e-10*max(1,abs(hist[-2])) and delta<1e-5:stop='objective_and_rotation_stable';break
  if step%100==0:status(stage='alternating_fit',k=k,seed=seed,step=step,objective=float(val))
 fit=dict(steps=step,stop=stop,seconds=time.time()-start,objective=hist[-1],history=hist,rotation_change=delta,objective_definition='0.5 sum_s ||base_s R - S_s||F^2, at most k entries per column per side; same R; no column rescaling')
 save(meta,fit);np.savez_compressed(dest,rotation=r.cpu().numpy());return r,fit
def fit_l1(base,seed):
 folder=OUT/'fits';dest=folder/f'l1-s{seed}.npz';meta=dest.with_suffix('.json')
 if dest.exists():return a.tensor(np.load(dest)['rotation']),json.loads(meta.read_text())
 r=initial(seed);start=time.time();hist=[];eps=1e-6;stop='iteration_limit';step_size=.1
 def objective(rr):return sum(torch.sqrt((base[s]@rr).square()+eps**2).sum() for s in S)/(2*256)
 for step in range(1,4001):
  zs={s:base[s]@r for s in S};g=sum(base[s].T@(zs[s]/torch.sqrt(zs[s].square()+eps**2)) for s in S)/(2*256);omega=(r.T@g-g.T@r)/2;grad=r@omega;val=objective(r);gn=float(grad.norm());alpha=min(step_size*1.4,10.)
  for ls in range(30):
   rn=a.polar(r-alpha*grad);v=objective(rn)
   if float(v)<=float(val)-1e-4*alpha*gn**2:break
   alpha*=.5
  if ls==29:stop='line_search_stall';break
  r=rn;step_size=alpha;hist.append(float(v))
  if gn<1e-6:stop='gradient';break
  if step%200==0:status(stage='l1_fit',seed=seed,step=step,objective=float(v),gradient=gn)
 fit=dict(steps=step,stop=stop,seconds=time.time()-start,objective=float(objective(r)),gradient=gn,epsilon=eps,history=hist,objective_definition='Smoothed coefficient L1 mean over paired256 columns; shared orthogonal rotation')
 save(meta,fit);np.savez_compressed(dest,rotation=r.cpu().numpy());return r,fit
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['1','2'],required=True);args=ap.parse_args()
 try:
  save(OUT/'protocol.json',dict(rank=256,budgets=BUDGETS,seeds=list(range(10)),splits=20,training='CC3M SAE and mapping',evaluation='COCO val2017, fixed50, own-caption text annotations',pruning='Coefficient energy or absolute Euler variance allocation a_j (Sigma a)_j. SD-weighted raw coefficients equal standardized coefficients exactly. Mass percentages refer to score mass, not output variance retention.',support='Equal fixed per-column counts on each modality. No test-score selection. Fraction curves also retain exact mean counts.',stage3='Not executed; user decides.'))
  e=Eval();e.stage1() if args.stage=='1' else e.stage2()
 except Exception as ex:save(OUT/f'error-stage{args.stage}.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
