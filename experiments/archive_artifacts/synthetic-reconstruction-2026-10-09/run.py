import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import base as old

def generator(seed,kind):
 d,_=old.loadings(seed)
 if kind=='noanchors':
  rng=np.random.default_rng(seed+7100)
  for col in np.where((d>0).sum(0)==1)[0]:
   j=np.argmax(d[:,col]);other=int(rng.choice([k for k in range(8) if k!=j]))
   d[other,col]=d[j,col]*rng.uniform(.75,1.25)
  d/=np.linalg.norm(d,axis=1,keepdims=True)
  assert not np.any((d>0).sum(0)==1)
 return d

def train(x,y,tx,ty,initial,seed,epochs,alpha,beta,out,name):
 torch.manual_seed(seed);rng=np.random.default_rng(seed)
 x,y,tx,ty=[torch.tensor(v,device='cuda') for v in [x,y,tx,ty]]
 a,b=[torch.nn.Parameter(torch.tensor(v,device='cuda').abs()) for v in initial]
 for w in [a,b]:old.project(w,positive=True)
 di,dt=[torch.nn.Parameter(torch.rand((8,128),device='cuda')*.01) for _ in range(2)]
 initial_weights=[v.detach().cpu().numpy().copy() for v in [a,b,di,dt]]
 np.savez_compressed(out/(name+'-initial.npz'),image=initial_weights[0],text=initial_weights[1],decoder_image=initial_weights[2],decoder_text=initial_weights[3])
 opt=torch.optim.Adam([a,b,di,dt],lr=.003)
 denom=[(x*x).mean().detach(),(y*y).mean().detach()]
 def objective(xx,yy):
  u,v=xx@a,yy@b
  logits=F.normalize(u,dim=1)@F.normalize(v,dim=1).T/.07
  ids=torch.arange(len(xx),device='cuda')
  nce=(F.cross_entropy(logits,ids)+F.cross_entropy(logits.T,ids))/2
  rec=((u@di-xx).square().mean()/denom[0]+(v@dt-yy).square().mean()/denom[1])/2
  act=(u.abs().mean()+v.abs().mean())/2
  return nce+alpha*rec+beta*act,nce,rec,act
 best=float('inf');hist=[];began=time.monotonic()
 for ep in range(epochs+1):
  if ep:
   perm=rng.permutation(len(x))
   for start in range(0,len(x),512):
    idx=torch.tensor(perm[start:start+512],device='cuda');opt.zero_grad(set_to_none=True)
    loss,*_=objective(x[idx],y[idx]);assert torch.isfinite(loss)
    loss.backward();torch.nn.utils.clip_grad_norm_([a,b,di,dt],5);opt.step()
    for w in [a,b]:old.project(w,positive=True)
    with torch.no_grad():
     for w in [di,dt]:w.clamp_(min=0);w.div_(w.norm(dim=1,keepdim=True).clamp_min(1))
  with torch.no_grad():vals=np.mean([[float(v) for v in objective(tx[j:j+512],ty[j:j+512])] for j in range(0,len(tx),512)],axis=0)
  hist.append(dict(epoch=ep,total=vals[0],nce=vals[1],relative_mse=vals[2],mean_activity=vals[3]))
  if vals[0]<best:
   best=vals[0];bestep=ep;weights=[w.detach().cpu().numpy().copy() for w in [a,b,di,dt]]
 old.save(out/(name+'-fit.json'),dict(history=hist,best_epoch=bestep,seconds=time.monotonic()-began,alpha=alpha,beta=beta))
 np.savez_compressed(out/(name+'-decoder.npz'),image=weights[2],text=weights[3])
 return weights[:2]

def main(args):
 torch.set_num_threads(2)
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 old.save(out/'protocol.json',dict(seeds=args.seeds,epochs=args.epochs,methods={'nce':[0,0],'nce_reconstruction':[1,0],'nce_reconstruction_activity':[1,.1]},train_n=8192,tune_n=2048,test_n=2048,dimensions=128,combinations=8,max_support=16,normalization='Train coordinate std, no centering',decoder='nonnegative rows norm <=1',encoder='nonnegative columns unit norm, <=16 entries',selection='lowest own total tune objective; labels never used',activity='mean absolute combination value',reconstruction='mean square error divided by train mean square input, averaged modalities',P='identity fixed',noanchors='Each formerly exclusive input additionally loads on one random other factor. No factor-exclusive input remains. Exact pure recovery may be infeasible under nonnegative linear encoders.'))
 count=0;began=time.monotonic()
 for kind in ['anchors','noanchors']:
  for seed in range(args.seeds):
   di=generator(1000+seed,kind);dt=generator(2000+seed,kind)
   for rho in [0,.5,.9,1]:
    dest=out/f'{kind}-seed{seed}-rho{rho:.1f}';dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():count+=1;continue
    old.save(out/'progress.json',dict(state='running',condition=dest.name,completed=count,total=args.seeds*8,seconds=time.monotonic()-began))
    x,y,z=old.observations(3000+seed,8192,rho,di,dt);tx,ty,_=old.observations(4000+seed,2048,rho,di,dt)
    means=[x.mean(0),y.mean(0)];scales=[x.std(0).clip(1e-6),y.std(0).clip(1e-6)]
    cv=np.cov(np.column_stack([(x-means[0])/scales[0],(y-means[1])/scales[1]]).astype(float),rowvar=False,bias=True)
    base=old.fit_common_projection(cv[:128,:128],cv[:128,128:],cv[128:,128:],8,whiten=True,ridge=.01)
    model=old.fit_sparse_cca(cv[:128,:128],cv[:128,128:],cv[128:,128:],8,k=16,ridge=.01,initial_image=base.image,initial_text=base.text,max_iter=100,swap_steps=2,candidate_pool=8)
    initial=[model.image.astype('float32'),model.text.astype('float32')]
    tests=dict(iid=old.observations(5000+seed,2048,rho,di,dt),recombined=old.observations(6000+seed,2048,0,di,dt))
    np.savez_compressed(dest/'generator.npz',image=di,text=dt,mean_image=means[0],mean_text=means[1],scale_image=scales[0],scale_text=scales[1])
    for name,alpha,beta in [('nce',0,0),('nce_reconstruction',1,0),('nce_reconstruction_activity',1,.1)]:
     weights=train(x/scales[0],y/scales[1],tx/scales[0],ty/scales[1],initial,seed,args.epochs,alpha,beta,dest,name)
     old.evaluate(name,*weights,False,means,scales,di,dt,tests,dest)
    old.save(dest/'complete.json',dict(completed=True));count+=1
 old.save(out/'progress.json',dict(state='completed',completed=count,total=args.seeds*8,seconds=time.monotonic()-began))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=100);p.add_argument('--seeds',type=int,default=3)
 a=p.parse_args()
 try:main(a)
 except Exception as e:old.save(Path(a.output)/'progress.json',dict(state='failed',error=repr(e)));raise
