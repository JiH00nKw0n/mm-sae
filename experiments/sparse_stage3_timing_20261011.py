"""Timing only. No parameter updates and no stage-3 model training."""
from experiments.sparse_improvement_20261011 import *
cfg=p.old.load_config(ROOT/p.CONFIG['cc3m-cc3m'])
with np.load(Path(cfg['parent_run'])/'moments.npz') as z:
 n=len(z['image_ids']);mu=z['fit_mean'];cv=z['fit_second']-np.outer(mu,mu)
sd=np.sqrt(np.maximum(cv.diagonal(),0));sd[sd==0]=1;cv/=sd[:,None]*sd[None,:]
xx,cross,yy=[a.tensor(q) for q in [cv[:n,:n],cv[:n,n:],cv[n:,n:]]]
z=np.load(ROOT/'runs/mapping-ablation-cc3m-2026-10-04/transforms/cross_svd_3221.npz');aa=a.tensor(z['image'][:,:256]);bb=a.tensor(z['text'][:,:256]);target=aa.T@cross@bb
results={}
for mode in ['direct_AB_gradient_and_variance_projection','nonorthogonal_T_gradient_and_singular_value_projection']:
 times=[]
 for repeat in range(12):
  torch.cuda.synchronize();start=time.time()
  if mode.startswith('direct'):
   u=aa.clone().requires_grad_();v=bb.clone().requires_grad_();loss=(u.T@cross@v-target).square().sum()/target.square().sum()+.001*(u.abs().sum()+v.abs().sum());torch.autograd.grad(loss,[u,v])
   with torch.no_grad():vi=(u*(xx@u)).sum(0);vt=(v*(yy@v)).sum(0);unused_u=u/vi.sqrt();unused_v=v/vt.sqrt()
  else:
   t=torch.eye(256,device='cuda',dtype=torch.float64).requires_grad_();loss=(aa@t).abs().sum()+(bb@torch.linalg.inv(t).T).abs().sum();torch.autograd.grad(loss,t)
   with torch.no_grad():u,s,v=torch.linalg.svd(t);unused=(u*s.clamp(.5,2))@v
  torch.cuda.synchronize();times.append(time.time()-start)
 results[mode]=dict(median_seconds=float(np.median(times[2:])),raw_2000_steps_seconds=float(np.median(times[2:])*2000),times=times,note='Gradient and projection cost only, no optimizer parameter updates; excludes line searches, convergence requirements, loading and evaluation')
save(OUT/'stage3-timing-only.json',results)
