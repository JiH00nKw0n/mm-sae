"""Fixed PLS256 shared orthogonal rotations towards row-sparse/nonnegative targets."""
from experiments.energy_alternating_20261011 import *
import experiments.energy_alternating_20261011 as energy
from scipy.optimize import linear_sum_assignment
OUT=ROOT/'runs/row-rotation-2026-10-12'
for sub in ['fits','eval','checkpoints']:(OUT/sub).mkdir(parents=True,exist_ok=True)
energy.OUT=OUT
VARIANTS=['nonnegative','row1','row2','row4','positive_row1']

def positive_sign(z):
 gain=sum((v*v.abs()).sum(-2) for v in z.values())
 return torch.where(gain<0,-1.,1.)

def project(z,kind):
 out={}
 for s,w in z.items():
  if kind=='nonnegative':out[s]=w.clamp_min(0);continue
  v=w.clamp_min(0) if kind=='positive_row1' else w
  k=1 if kind=='positive_row1' else int(kind[3:])
  ix=v.abs().topk(k,dim=-1).indices
  out[s]=torch.zeros_like(w).scatter_(-1,ix,v.gather(-1,ix))
 return out

def residual(z,t):return sum((z[s]-t[s]).square().sum((-2,-1)) for s in S)/2

def signed_projection(base,r,kind):
 z={s:base[s][None]@r for s in S};t=project(z,kind)
 if kind in ['nonnegative','positive_row1']:
  sign=positive_sign(z);candidate={s:z[s]*sign[:,None,:] for s in S};ct=project(candidate,kind)
  # The combined constraint is not separable over column signs. Accept this
  # label-free sign proposal only when it decreases the actual joint objective.
  accept=residual(candidate,ct)<residual(z,t)
  d=torch.where(accept[:,None],sign,torch.ones_like(sign));r=r*d[:,None,:]
  z={s:z[s]*d[:,None,:] for s in S};t=project(z,kind)
 return r,z,t

def fit_batch_row(base,kind,startmode,seeds,limit=10000):
 names=[f'{kind}-{startmode}-s{s}' for s in seeds]
 pending=[i for i,n in enumerate(names) if not (OUT/'fits'/f'{n}.npz').exists()]
 if not pending:return
 seeds=[seeds[i] for i in pending];names=[names[i] for i in pending]
 starts=[initial(s) if startmode=='original' else a.tensor(np.load(SRC/f'rank256/covariance-svd-{s}-oriented.npz')['rotation']) for s in seeds]
 r=torch.stack(starts);hist=[[] for _ in names];done=[False]*len(names);steps=[0]*len(names)
 for j,name in enumerate(names):
  cp=OUT/'checkpoints'/f'{name}.npz'
  if cp.exists():
   r[j]=a.tensor(np.load(cp)['rotation']);data=json.loads(cp.with_suffix('.json').read_text());hist[j]=data['history'];steps[j]=len(hist[j])
 began=time.time()
 for it in range(limit):
  if all(done):break
  before={s:base[s][None]@r for s in S};oldobj=residual(before,project(before,kind))
  signed,z,t=signed_projection(base,r,kind);obj=residual(z,t)
  m=sum(base[s].T[None]@t[s] for s in S);rn=polar(m)
  if it==0:assert float((rn-a.polar(m)).abs().max())<1e-9
  zn={s:base[s][None]@rn for s in S};fixedobj=residual(zn,t);newobj=residual(zn,project(zn,kind));delta=(rn-r).norm(dim=(1,2))
  for j,name in enumerate(names):
   if done[j]:continue
   steps[j]+=1
   row=dict(step=steps[j],before=float(oldobj[j]),after_sign=float(obj[j]),after_rotation=float(fixedobj[j]),after_projection=float(newobj[j]),rotation_change=float(delta[j]))
   row['increase']=max(row['after_sign']-row['before'],row['after_rotation']-row['after_sign'],row['after_projection']-row['after_rotation'])
   if row['increase']>1e-8:raise RuntimeError(('nonmonotone',name,row))
   hist[j].append(row);r[j]=rn[j]
   converged=steps[j]>10 and max(v['rotation_change'] for v in hist[j][-5:])<1e-5
   if converged or steps[j]>=limit:
    done[j]=True;np.savez_compressed(OUT/'fits'/f'{name}.npz',rotation=r[j].cpu().numpy())
    save(OUT/'fits'/f'{name}.json',dict(kind=kind,start=startmode,seed=seeds[j],steps=steps[j],stop='fixed_point' if converged else 'iteration_limit',history=hist[j],seconds=time.time()-began,orthogonality_error=float((r[j].T@r[j]-torch.eye(256,device='cuda')).abs().max())))
  if it%100==0:
   for j,n in enumerate(names):
    if not done[j]:
     np.savez(OUT/'checkpoints'/f'{n}.npz',rotation=r[j].cpu().numpy());save(OUT/'checkpoints'/f'{n}.json',dict(history=hist[j]))
   save(OUT/'progress.json',dict(names=names,steps=steps,done=done,seconds=time.time()-began));print(names,steps,done,flush=True)

def coeff_stats(w):
 energy=w.square();total=float(energy.sum());v=energy.sort(dim=1,descending=True).values;active=v.sum(1)>0
 n=((v.cumsum(1)<.9*v.sum(1,keepdim=True)).sum(1)+1)*active
 ab=w.abs();g=ab.T@ab;mask=~torch.eye(w.shape[1],dtype=torch.bool,device=w.device)
 norms=ab.norm(dim=0);cos=g/(norms[:,None]*norms[None,:]).clamp_min(1e-30)
 return dict(negative_energy=float(energy[w<0].sum())/total,row90_mean=float(n.double().mean()),row90_active_mean=float(n[active].double().mean()),row90_median=float(n.double().median()),row90_max=int(n.max()),zero_row_fraction=float((~active).double().mean()),absolute_overlap=float(g[mask].mean()),absolute_cosine_overlap=float(cos[mask].mean()),column_count=float((w!=0).sum(0).double().mean()))

class RowEval(EnergyEval):
 def run_one(self,name,seed,r):
  raw={s:self.base[s]@r for s in S};sign=positive_sign(raw);positive_r=r*sign
  for orient,rr in [('positive',positive_r),('legacy',self.oriented_rotation(r))]:
   w={s:self.base[s]@rr for s in S}
   for cut in ['dense','k64']:
    namefull=f'{name}-{orient}-{cut}';dest=OUT/'eval'/f'{namefull}-s{seed}.json'
    if dest.exists():continue
    ws=w if cut=='dense' else {s:hard(w[s],64) for s in S}
    meta=dict(coefficient_stats={s:coeff_stats(ws[s]) for s in S},orientation=orient,cut=cut)
    self.measure(namefull,seed,ws,meta)

def main():
 save(OUT/'protocol.json',dict(rank=256,variants=VARIANTS,seeds=10,splits=20,starts='original protocol identity seed0 and nine random seeds1201..1209; covariance solutions with same seed',objective='0.5 sum modalities ||base R - target||F^2',evaluation='rotated coefficients, not projection targets; dense and top64 per column; positive-energy common signs primary and training-skew signs sensitivity',combined_sign='positive-energy sign proposal accepted only if actual joint residual decreases; not claimed global sign optimum',stop='5 successive Frobenius steps<1e-5 after10, cap10000',reference='CC3M/CC3M fixed PLS256, own-caption labels, saved20splits'))
 with np.load(ROOT/'runs/mapping-ablation-cc3m-2026-10-04/transforms/cross_svd_3221.npz') as f:base={s:a.tensor(f[s][:,:256]) for s in S}
 # First measure coefficient-only baselines before starting new fits.
 baseline={}
 for seed in range(10):
  for name,r in [('svd',torch.eye(256,device='cuda',dtype=torch.float64)),('covariance',a.tensor(np.load(SRC/f'rank256/covariance-svd-{seed}-oriented.npz')['rotation'])),('fixed64',a.tensor(np.load(OLD/'fits'/f'alternating-k64-s{seed}.npz')['rotation']))]:
   w={s:base[s]@r for s in S};sg=positive_sign(w);w={s:w[s]*sg for s in S}
   baseline[f'{name}-s{seed}']={cut:{s:coeff_stats(w[s] if cut=='dense' else hard(w[s],64)) for s in S} for cut in ['dense','k64']}
 save(OUT/'baseline-coefficients.json',baseline)
 # Quantify the earlier 99% motion independent of signs and permutations.
 motion=[]
 for seed in range(10):
  r0=initial(seed);path=ROOT/'runs/energy-alternating-2026-10-11/fits'/f'ind99-s{seed}.npz';rf=a.tensor(np.load(path)['rotation']);cross=(r0.T@rf).cpu().numpy();ii,jj=linear_sum_assignment(-abs(cross));q=abs(cross[ii,jj]);meta=json.loads(path.with_suffix('.json').read_text())
  motion.append(dict(seed=seed,mean_axis_cosine_after_assignment=float(q.mean()),aligned_frobenius=float(np.sqrt(2*256-2*q.sum())),raw_frobenius=float((r0-rf).norm()),initial_objective=meta['history'][0]['residual'],final_objective=meta['history'][-1]['residual'],stop=meta['stop']))
 save(OUT/'energy99-motion.json',motion)
 for kind in VARIANTS:
  for start in ['original','covariance']:
   for first in [0,5]:fit_batch_row(base,kind,start,list(range(first,first+5)))
 save(OUT/'fits-complete.json',dict(time=time.time()))
 e=RowEval()
 for seed in range(10):
  if seed==0:e.run_one('svd',0,torch.eye(256,device='cuda',dtype=torch.float64))
  e.run_one('covariance',seed,a.tensor(np.load(SRC/f'rank256/covariance-svd-{seed}-oriented.npz')['rotation']))
  e.run_one('fixed64',seed,a.tensor(np.load(OLD/'fits'/f'alternating-k64-s{seed}.npz')['rotation']))
  for kind in VARIANTS:
   for start in ['original','covariance']:
    name=f'{kind}-{start}';e.run_one(name,seed,a.tensor(np.load(OUT/'fits'/f'{name}-s{seed}.npz')['rotation']))
 save(OUT/'complete.json',dict(time=time.time(),fits=100,evaluations=484))
if __name__=='__main__':
 try:main()
 except Exception as ex:save(OUT/'error.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
