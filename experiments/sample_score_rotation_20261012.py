"""Full-training paired score fourth-moment rotations and fixed-axis cluster complements."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import experiments.pair_direction_rotation_20261012 as pair
from experiments.pair_direction_rotation_20261012 import *
import argparse
PAIR=OUT;OUT=ROOT/'runs/sample-score-rotation-2026-10-12'
for sub in ['fits','eval','directions','checkpoints']:(OUT/sub).mkdir(parents=True,exist_ok=True)
def log(**v):save(OUT/'progress.json',dict(time=time.time(),**v));print(v,flush=True)
exec(inspect.getsource(oracle.measure).replace("['dense','column64','global64']","['dense','column64']"),globals())
original_measure=measure

def measure(e,name,seed,r,train,selected,heldout,meta):
 original_measure(e,name,seed,r,train,selected,heldout,meta)
 for cut in ['dense','column64']:
  p=OUT/'eval'/f'{name}-{cut}-s{seed}.json';q=json.loads(p.read_text())
  if 'coefficient_n90' in q:continue
  q['coefficient_n90']={}
  for side in S:
   w=e.base[side]@r
   if cut=='column64':w=hard(w,64)
   v=w.square().sort(0,descending=True).values;num=(v.cumsum(0)<.9*v.sum(0)).sum(0)+1;num=torch.where(v.sum(0)>0,num,0);nn=num.double()
   q['coefficient_n90'][side]=dict(mean=float(nn.mean()),median=float(nn.median()),min=int(nn.min()),max=int(nn.max()),per_axis=num.cpu().tolist())
  save(p,q)

def tangent(r,g):return g-r@((r.T@g+g.T@r)/2)
class ScoreObjective:
 def __init__(self,e,data):
  self.x=[a.tensor(data[s]) for s in S];self.n=len(self.x[0]);self.second=[x.T@x/self.n for x in self.x];self.cov=[s-torch.outer(x.mean(0),x.mean(0)) for x,s in zip(self.x,self.second)]
  self.l=[]
  for side in S:
   l=e.cov[side]@e.base[side];self.l.append(l/l.norm(dim=1,keepdim=True).clamp_min(1e-12))
  self.scales={};r=a.tensor(np.load(SRC/'rank256/covariance-svd-0-oriented.npz')['rotation'])
  for norm in [False,True]:
   vals,_=self.components(r,norm)
   for kind in ['var','cross','sum']:
    raw=vals[0] if kind=='var' else vals[1] if kind=='cross' else vals[0]+vals[1];self.scales[(kind,norm)]=abs(float(raw))
   self.scales[('loading',norm)]=abs(float(vals[2]))
  save(OUT/'objective-scales.json',{f'{k[0]}-std{k[1]}':v for k,v in self.scales.items()})
 def components(self,r,norm):
  q=[(r*(s@r)).sum(0) for s in self.second];v=[(r*(s@r)).sum(0).clamp_min(1e-12) for s in self.cov] if norm else [torch.ones_like(q[0])]*2
  fourth=[torch.zeros_like(q[0]) for _ in S];joint=torch.zeros_like(q[0]);gf=[torch.zeros_like(r) for _ in S];gc=[torch.zeros_like(r) for _ in S]
  for start in range(0,self.n,32768):
   xx=[x[start:start+32768] for x in self.x];z=[x@r for x in xx];zz=[z0.square() for z0 in z]
   for j in [0,1]:fourth[j]+=zz[j].square().sum(0);gf[j]+=4*xx[j].T@(z[j]*zz[j]);gc[j]+=2*xx[j].T@(z[j]*zz[1-j])
   joint+=(zz[0]*zz[1]).sum(0)
  fourth=[f/self.n for f in fourth];joint/=self.n;gf=[g/self.n for g in gf];gc=[g/self.n for g in gc]
  va=sum(((f-q0.square())/v0.square()).sum() for f,q0,v0 in zip(fourth,q,v));ga=sum((g-4*(s@r)*q0)/v0.square() for g,s,q0,v0 in zip(gf,self.second,q,v))
  numerator=joint-q[0]*q[1];vb=(numerator/(v[0]*v[1])).sum();gb=(gc[0]+gc[1]-2*(self.second[0]@r)*q[1]-2*(self.second[1]@r)*q[0])/(v[0]*v[1])
  if norm:
   ga-=sum(4*(c@r)*(f-q0.square())/v0.pow(3) for c,f,q0,v0 in zip(self.cov,fourth,q,v))
   gb-=2*numerator*((self.cov[0]@r)/(v[0].square()*v[1])+(self.cov[1]@r)/(v[1].square()*v[0]))
  lc=torch.zeros((),device=r.device,dtype=r.dtype);gl=torch.zeros_like(r)
  for l in self.l:
   z=l@r;mu=z.square().mean(0);lc+=(z.pow(4).mean(0)-mu.square()).sum()/2;gl+=2/len(l)*l.T@(z.pow(3)-z*mu)
  return [va,vb,lc],[ga,gb,gl]
 def evaluate(self,r,kind,norm):
  vals,gs=self.components(r,norm)
  if kind in ['var','cross']:
   j=0 if kind=='var' else 1;return float(vals[j]/self.scales[(kind,norm)]),gs[j]/self.scales[(kind,norm)],list(map(float,vals))
  value=(vals[0]+vals[1])/self.scales[('sum',norm)];g=(gs[0]+gs[1])/self.scales[('sum',norm)]
  if kind=='mixed':value+=vals[2]/self.scales[('loading',norm)];g+=gs[2]/self.scales[('loading',norm)]
  return float(value),g,list(map(float,vals))

def fit_score(obj,r,kind,norm,name,limit=1500):
 path=OUT/'fits'/f'{name}.npz'
 if path.exists():return a.tensor(np.load(path)['rotation']),json.loads(path.with_suffix('.json').read_text())
 cp=OUT/'checkpoints'/f'{name}.npz';history=[]
 if cp.exists():r=a.tensor(np.load(cp)['rotation']);history=json.loads(cp.with_suffix('.json').read_text())['history']
 start=time.time();value,g,components=obj.evaluate(r,kind,norm);grad=-tangent(r,g);ss=[];ys=[];stop='iteration_limit'
 for it in range(len(history),limit):
  gn=float(grad.norm())/max(abs(value),1.);history.append(dict(step=it,objective=value,relative_gradient=gn,components=components))
  if gn<1e-6:stop='gradient';break
  q=grad.clone();aa=[];rho=[]
  for s,y in reversed(list(zip(ss,ys))):
   rh=1/float((s*y).sum());alpha=rh*float((s*q).sum());q-=alpha*y;aa.append(alpha);rho.append(rh)
  scale=float((ss[-1]*ys[-1]).sum()/ys[-1].square().sum()) if ss else 1.;direction=q*scale
  for s,y,alpha,rh in zip(ss,ys,reversed(aa),reversed(rho)):direction+=s*(alpha-rh*float((y*direction).sum()))
  direction=-tangent(r,direction);slope=float((grad*direction).sum())
  if slope>=-1e-20:direction=-grad;ss=[];ys=[];slope=-float(grad.square().sum())
  step=1.;accepted=False
  for trial in range(24):
   rn=a.polar(r+step*direction);nv,ng,nc=obj.evaluate(rn,kind,norm)
   if nv>=value-1e-4*step*slope:accepted=True;break
   step*=.5
  if not accepted:stop='line_search_stalled';break
  newgrad=-tangent(rn,ng);transportgrad=tangent(rn,grad);s=tangent(rn,rn-r);y=newgrad-transportgrad
  ss=[tangent(rn,v) for v in ss];ys=[tangent(rn,v) for v in ys]
  if float((s*y).sum())>1e-12*float(s.norm()*y.norm()):ss.append(s);ys.append(y);ss=ss[-10:];ys=ys[-10:]
  r=rn;value=nv;grad=newgrad;components=nc
  if it%20==0:
   np.savez_compressed(cp,rotation=r.cpu().numpy());save(cp.with_suffix('.json'),dict(history=history));log(stage='score_fit',name=name,step=it,objective=value,relative_gradient=gn,seconds=time.time()-start)
 # Final gradient is evaluated at the matrix that is saved.
 value,g,components=obj.evaluate(r,kind,norm);gn=float(tangent(r,g).norm())/max(abs(value),1.)
 if gn<1e-6:stop='gradient'
 meta=dict(kind=kind,standardized=norm,stop=stop,steps=len(history),objective=value,relative_gradient=gn,components=components,history=history,seconds=time.time()-start,orthogonality=float((r.T@r-torch.eye(256,device='cuda')).abs().max()),solver='Riemannian L-BFGS, polar retraction, projected vector transport, monotone Armijo line search',sample_count=obj.n)
 np.savez_compressed(path,rotation=r.cpu().numpy());save(path.with_suffix('.json'),meta);return r,meta

def check_gradient(e,data):
 # Exact autograd agreement on a small subset for all three analytic components.
 tiny={s:data[s][:257] for s in S};obj=ScoreObjective(e,tiny);r=initial(7).detach().requires_grad_(True);report=[]
 for norm in [False,True]:
  z=[x@r for x in obj.x]
  if norm:z=[v/v.std(0,unbiased=False).clamp_min(1e-12) for v in z]
  vals=[sum((v.pow(4).mean(0)-v.square().mean(0).square()).sum() for v in z),((z[0].square()*z[1].square()).mean(0)-z[0].square().mean(0)*z[1].square().mean(0)).sum(),sum(((l@r).pow(4).mean(0)-(l@r).square().mean(0).square()).sum()/2 for l in obj.l)]
  av,ag=obj.components(r.detach(),norm)
  for j in range(3):
   g=torch.autograd.grad(vals[j],r,retain_graph=True)[0];err=float((g-ag[j]).norm()/g.norm());assert err<1e-9,(norm,j,err);assert abs(float(vals[j])-float(av[j]))/max(1,abs(float(vals[j])))<1e-10;report.append(dict(normalized=norm,component=j,gradient_relative_error=err))
 save(OUT/'gradient-validation.json',report)

def coefficient_complement(e,r,k,name,limit=10000):
 path=OUT/'fits'/f'{name}.npz'
 if path.exists():return a.tensor(np.load(path)['rotation']),json.loads(path.with_suffix('.json').read_text())
 q=r[:,:k].clone();n=r[:,k:].clone();h=torch.eye(256-k,device='cuda',dtype=torch.float64);b={s:e.base[s]@n for s in S};fixed={s:e.base[s]@q for s in S};history=[];recent=[];best=float('inf');besth=None;stop='iteration_limit';cycle=None
 cp=OUT/'checkpoints'/f'{name}.npz'
 if cp.exists():
  ck=np.load(cp);h=a.tensor(ck['h']);besth=a.tensor(ck['besth']);mh=json.loads(cp.with_suffix('.json').read_text());history=mh['history'];best=mh['best'];besti=mh['besti']
 for it in range(len(history),limit):
  z={s:torch.cat([fixed[s],b[s]@h],1)[None] for s in S};t={s:target(z[s],'rowfirst',1,'mass90') for s in S};val=float(residual(z,t)[0]);hn=a.polar(sum(b[s].T@t[s][0,:,k:] for s in S));change=float((hn-h).norm())
  if val<best:best=val;besth=h.clone();besti=it
  history.append(dict(step=it,residual=val,change=change,best=best))
  if change<1e-5 and it>10:stop='fixed_point';break
  if len(recent)>=48:
   for period in range(2,17):
    if max(float((h-recent[-j*period]).norm()) for j in [1,2,3])<1e-9:stop='cycle';cycle=period;break
   if cycle:break
  recent.append(h.clone());recent=recent[-48:];h=hn
  if it%100==0:
   np.savez_compressed(cp,h=h.cpu().numpy(),besth=besth.cpu().numpy());save(cp.with_suffix('.json'),dict(history=history,best=best,besti=besti));log(stage='cluster_complement',name=name,step=it,objective=val,best=best,change=change)
 r=torch.cat([q,n@besth],1);sg=positive_sign({s:e.base[s]@r for s in S});sg[:k]=1;r=r*sg
 meta=dict(stop=stop,cycle=cycle,steps=len(history),best_step=besti,best_objective=best,history=history,assigned_axes_change=float((r[:,:k]-q).abs().max()),orthogonality=float((r.T@r-torch.eye(256,device='cuda')).abs().max()),selection='minimum residual among all visited states; assigned prototype axes fixed')
 np.savez_compressed(path,rotation=r.cpu().numpy());save(path.with_suffix('.json'),meta);return r,meta

def main(args):
 e=RC_Eval();train=dict(np.load(OLDOUT/'directions/train.npz'));data=dict(np.load(PAIR/'directions/pairs.npz'))
 for name in ['train-membership.npz','training.json']:shutil.copy2(PAIR/name,OUT/name)
 save(OUT/'protocol.json',dict(samples=len(data['image']),space='fixed CC3M SAE / CC3M PLS256',rotation_data='COCO FIT pairs, exactly prior paired-data experiment, no evaluation overlap',score_variants=['sum Var(z_j^2)+Var(w_j^2)','sum Cov(z_j^2,w_j^2)','sum of preceding two','sum objective normalized at fixed covariance seed0 plus loading covariance Varimax normalized at same fixed point'],score_standardization='optional divide each rotated coordinate by its full-training standard deviation; no additional centering; objective only, retrieval uses original rotated scores',starts=['covariance','random'],seeds=10,evaluation_splits=20,cluster_counts=[16,32,48,64,96],cluster_seeds=3,complement_initializations=10,cuts=['dense','column64'],orientation='shared coefficient-positive-energy signs as in baseline; no evaluation labels',oracle='43.07 measured supervised reference, not proven maximum'))
 for seed in range(10):
  for ref,src in [('covariance',SRC/f'rank256/covariance-svd-{seed}-oriented.npz'),('cycle-best',ROOT/'runs/feature-redundancy-cycle-2026-10-12/fits'/f'cycle-best-s{seed}.npz'),('mean-all50',OLDOUT/'fits'/f'mean-all50-s{seed}.npz')]:
   r=a.tensor(np.load(src)['rotation'])
   for cut in ['dense','column64']:
    dest=OUT/'eval'/f'{ref}-{cut}-s{seed}.json'
    if not dest.exists():shutil.copy2(OLDOUT/'eval'/dest.name,dest)
   measure(e,ref,seed,r,train,[],list(range(50)),dict(source=str(src)))
 if args.phase in ['all','scores']:
  check_gradient(e,data);obj=ScoreObjective(e,data)
  for kind in ['var','cross','sum','mixed']:
   for norm in [False,True]:
    for startname in ['covariance','random']:
     for seed in range(10):
      r0=a.tensor(np.load(SRC/f'rank256/covariance-svd-{seed}-oriented.npz')['rotation']) if startname=='covariance' else initial(100+seed)
      name=f'score-{kind}-std{int(norm)}-{startname}';r,meta=fit_score(obj,r0.clone(),kind,norm,name+f'-s{seed}',args.limit);sg=positive_sign({s:e.base[s]@r for s in S});r=r*sg;meta['evaluation_signs']=sg.cpu().tolist();np.savez_compressed(OUT/'fits'/f'{name}-s{seed}.npz',rotation=r.cpu().numpy());save(OUT/'fits'/f'{name}-s{seed}.json',meta);measure(e,name,seed,r,train,[],list(range(50)),meta)
      if args.pilot:return
  del obj;torch.cuda.empty_cache();save(OUT/'scores-complete.json',dict(time=time.time()))
 if args.phase in ['all','clusters']:
  pair.OUT=OUT/'cluster-data';(pair.OUT/'directions').mkdir(parents=True,exist_ok=True)
  for g0 in range(3):
   for suffix in ['.npz','.json','-memberships.npz']:
    src=PAIR/'directions'/f'cluster64-s{g0}{suffix}'
    if src.exists():shutil.copy2(src,pair.OUT/'directions'/src.name)
  for g in range(3):
   for k in [16,32,48,64,96]:
    ds=pair.clusters(data,k,g)
    for seed in range(10):
     r0=np.load(SRC/f'rank256/covariance-svd-{seed}-oriented.npz')['rotation'];rr,fit=fit_supervised(ds['image'],ds['text'],list(range(k)),r0)
     for mode in ['covariance','row90']:
      name=f'cluster{k}-g{g}-{mode}';path=OUT/'fits'/f'{name}-s{seed}.npz'
      if path.exists():r=a.tensor(np.load(path)['rotation']);meta=json.loads(path.with_suffix('.json').read_text())
      elif mode=='covariance':
       r,meta=complement_rotate(e,a.tensor(rr),k);meta['direction_fit']=fit;np.savez_compressed(path,rotation=r.cpu().numpy());save(path.with_suffix('.json'),meta)
      else:r,meta=coefficient_complement(e,a.tensor(rr),k,name+f'-s{seed}');meta['direction_fit']=fit;save(path.with_suffix('.json'),meta)
      measure(e,name,seed,r,train,[],list(range(50)),meta)
  save(OUT/'clusters-complete.json',dict(time=time.time()))
 if (OUT/'scores-complete.json').exists() and (OUT/'clusters-complete.json').exists():save(OUT/'complete.json',dict(fits=460,evaluations=980,new_evaluations=920,reused_reference_evaluations=60,time=time.time()))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['all','scores','clusters'],default='all');parser.add_argument('--pilot',action='store_true');parser.add_argument('--limit',type=int,default=1500);args=parser.parse_args()
 try:main(args)
 except Exception as ex:save(OUT/'error.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
