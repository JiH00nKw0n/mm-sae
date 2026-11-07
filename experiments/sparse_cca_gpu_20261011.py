"""Float64 GPU implementation of the existing sequential sparse CCA support solver.
Same ridge, initial supports, swap shortlist and stopping rule. Batched Cholesky only.
"""
from experiments.sparse_improvement_20261011 import *
def solve(metric,linear,support):
 local=metric[support[:,None],support[None,:]];inv=torch.cholesky_inverse(torch.linalg.cholesky(local));sol=inv@linear[support];val=(linear[support]*sol).sum().clamp_min(0);return inv,sol,val
def update(metric,linear,current,k):
 support=torch.nonzero(current).flatten()
 if len(support)<k:
  order=torch.argsort(linear.abs(),descending=True,stable=True);extra=order[~torch.isin(order,support)][:k-len(support)];support=torch.sort(torch.cat([support,extra])).values
 inv,sol,val=solve(metric,linear,support);changes=0
 for _ in range(2):
  outside=torch.arange(len(metric),device='cuda');outside=outside[~torch.isin(outside,support)];coupling=metric[outside[:,None],support[None,:]];trans=coupling@inv
  cond=metric.diagonal()[outside]-(trans*coupling).sum(1);rem=linear[outside]-coupling@sol;gains=rem.square()/cond.clamp_min(torch.finfo(torch.float64).tiny)
  candidates=outside[torch.argsort(gains,descending=True,stable=True)[:8]];expanded=torch.cat([support.expand(len(candidates),-1),candidates[:,None]],1)
  local=metric[expanded[:,:,None],expanded[:,None,:]];ei=torch.cholesky_inverse(torch.linalg.cholesky(local));es=(ei@linear[expanded][:,:,None]).squeeze(2);ev=(linear[expanded]*es).sum(1).clamp_min(0)
  loss=es.square()/ei.diagonal(dim1=-2,dim2=-1);drops=loss.argmin(1);values=ev-loss[torch.arange(len(candidates),device='cuda'),drops]
  best=val;chosen=None
  for j in range(len(candidates)):
   if float(values[j])>float(best)+1e-7*max(1.,float(best)):best=values[j];chosen=j
  if chosen is None:break
  supp=expanded[chosen];support=torch.sort(supp[torch.arange(len(supp),device='cuda')!=drops[chosen]]).values
  inv,sol,val=solve(metric,linear,support);changes+=1
 out=torch.zeros_like(current)
 if float(val)>torch.finfo(torch.float64).tiny:out[support]=sol/val.sqrt()
 else:
  out=current.clone()
  if not bool(out.any()):out[support[0]]=1
  out/=torch.sqrt(out@(metric@out))
 return out,changes
def normprune(v,metric,k):
 q=hard(v[:,None],k)[:,0]
 if not bool(q.any()):q[0]=1
 return q/torch.sqrt(q@(metric@q))
def fit(xx,cross,yy,initial_i,initial_t,dimensions=256,k=256):
 aa=xx+.01*torch.eye(len(xx),device='cuda',dtype=torch.float64);bb=yy+.01*torch.eye(len(yy),device='cuda',dtype=torch.float64);res=cross.clone();wi=torch.zeros((len(xx),dimensions),device='cuda',dtype=torch.float64);wt=torch.zeros((len(yy),dimensions),device='cuda',dtype=torch.float64);records=[];start=time.time()
 for j in range(dimensions):
  u=normprune(initial_i[:,j],aa,k);v=normprune(initial_t[:,j],bb,k);obj=float(u@(res@v))
  if obj<0:v=-v;obj=-obj
  history=[obj];converged=False
  for it in range(1,101):
   oi=u!=0;ot=v!=0;u,_=update(aa,res@v,u,k);v,_=update(bb,res.T@u,v,k);nv=float(u@(res@v));assert nv>=obj-1e-9*max(1,abs(obj))
   history.append(nv);converged=bool(torch.equal(oi,u!=0) and torch.equal(ot,v!=0) and nv-obj<=1e-7*max(1,abs(obj)));obj=nv
   if converged:break
  wi[:,j]=u;wt[:,j]=v;original=float(u@(cross@v));res-=obj*torch.outer(aa@u,bb@v)
  if original<0:wt[:,j]*=-1
  records.append(dict(component=j,iterations=it,converged=converged,history=history))
  save(OUT/'sparse-gpu-progress.json',dict(component=j+1,total=dimensions,seconds=time.time()-start,converged=converged))
  print(json.dumps(dict(component=j+1,seconds=time.time()-start)),flush=True)
 return wi,wt,records
if __name__=='__main__':
 try:
  cfg=p.old.load_config(ROOT/p.CONFIG['cc3m-cc3m'])
  with np.load(Path(cfg['parent_run'])/'moments.npz') as z:
   n=len(z['image_ids']);mu=z['fit_mean'];cv=z['fit_second']-np.outer(mu,mu)
  sd=np.sqrt(np.maximum(cv.diagonal(),0));sd[sd==0]=1;cv/=sd[:,None]*sd[None,:];xx,cross,yy=[a.tensor(q) for q in [cv[:n,:n],cv[:n,n:],cv[n:,n:]]]
  z=np.load(ROOT/'runs/mapping-ablation-cc3m-2026-10-04/transforms/cca_3221.npz');ii=a.tensor(z['image'][:,:256]);tt=a.tensor(z['text'][:,:256])
  # Validate the side update against the original CPU implementation before fitting.
  from mm_sae.analysis.sparse_cca import _update_side
  checks=[]
  for metric,lin,cur in [(xx+.01*torch.eye(len(xx),device='cuda'),cross@tt[:,0],ii[:,0]),(yy+.01*torch.eye(len(yy),device='cuda'),cross.T@ii[:,0],tt[:,0])]:
   cur=normprune(cur,metric,256);cpu,ch=_update_side(metric.cpu().numpy(),lin.cpu().numpy(),cur.cpu().numpy(),256,swap_steps=2,candidate_pool=8,tolerance=1e-7);gpu,cg=update(metric,lin,cur,256)
   err=float(np.max(np.abs(cpu-gpu.cpu().numpy())));same=bool(np.array_equal(cpu!=0,gpu.cpu().numpy()!=0));checks.append(dict(max_coefficient_error=err,same_support=same,cpu_swaps=ch,gpu_swaps=cg));assert same and err<1e-8 and ch==cg,checks
  save(OUT/'sparse-gpu-validation.json',checks)
  wi,wt,recs=fit(xx,cross,yy,ii,tt)
  tmp=OUT/'sparse-cca-k256-tmp.npz';np.savez_compressed(tmp,image=wi.cpu().numpy(),text=wt.cpu().numpy());tmp.replace(OUT/'sparse-cca-k256.npz');save(OUT/'sparse-gpu-complete.json',dict(records=recs))
 except Exception as ex:save(OUT/'error-sparse-gpu.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
