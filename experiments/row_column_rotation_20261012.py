"""Shared PLS256 rotation with sequential/union row and column targets."""
from experiments.row_rotation_20261012 import *
import experiments.energy_alternating_20261011 as energy
OUT=ROOT/'runs/row-column-rotation-2026-10-12'
for sub in ['fits','eval','checkpoints']:(OUT/sub).mkdir(parents=True,exist_ok=True)
energy.OUT=OUT

def row_cut(w,q):
 ix=w.abs().topk(q,dim=-1).indices
 return torch.zeros_like(w).scatter_(-1,ix,w.gather(-1,ix))
def col_cut(w,mode,sparse_hint=False):
 if mode=='k64':
  ix=w.abs().topk(64,dim=-2).indices
  return torch.zeros_like(w).scatter_(-2,ix,w.gather(-2,ix))
 if sparse_hint:
  cap=max(1,int((w!=0).sum(-2).max()))
  ix=w.abs().topk(cap,dim=-2,sorted=True).indices;v=w.gather(-2,ix);energy=v.square()
  count=(energy.cumsum(-2)<.9*energy.sum(-2,keepdim=True)).sum(-2)+1
  mask=torch.arange(cap,device=w.device)[None,:,None]<count[:,None,:]
  return torch.zeros_like(w).scatter_(-2,ix,v*mask)
 return truncate(w,fraction=.9)
def target(w,order,q,mode):
 if order=='rowfirst':return col_cut(row_cut(w,q),mode,sparse_hint=True)
 if order=='colfirst':return row_cut(col_cut(w,mode),q)
 return torch.where((row_cut(w,q)!=0)|(col_cut(w,mode)!=0),w,0.)
def fit(base,order,q,mode,startmode,seeds,limit=10000):
 prefix=f'{order}-row{q}-{mode}-{startmode}';names=[f'{prefix}-s{s}' for s in seeds]
 todo=[j for j,n in enumerate(names) if not (OUT/'fits'/f'{n}.npz').exists()]
 if not todo:return
 seeds=[seeds[j] for j in todo];names=[names[j] for j in todo]
 starts=[initial(s) if startmode=='original' else a.tensor(np.load(SRC/f'rank256/covariance-svd-{s}-oriented.npz')['rotation']) for s in seeds]
 r=torch.stack(starts);hist=[[] for _ in names];done=[False]*len(names);began=time.time()
 for j,n in enumerate(names):
  cp=OUT/'checkpoints'/f'{n}.npz'
  if cp.exists():r[j]=a.tensor(np.load(cp)['rotation']);hist[j]=json.loads(cp.with_suffix('.json').read_text())['history']
 cache_z=None;cache_t=None;recent=[]
 for it in range(limit):
  if all(done):break
  if any(done):
   keep=[j for j,d in enumerate(done) if not d];r=r[keep];names=[names[j] for j in keep];seeds=[seeds[j] for j in keep];hist=[hist[j] for j in keep];done=[False]*len(keep)
   if cache_z is not None:cache_z={s:cache_z[s][keep] for s in S};cache_t={s:cache_t[s][keep] for s in S}
   recent=[v[keep] for v in recent]
  z={s:base[s][None]@r for s in S} if cache_z is None else cache_z;t={s:target(z[s],order,q,mode) for s in S} if cache_t is None else cache_t;obj=residual(z,t)
  m=sum(base[s].T[None]@t[s] for s in S);rn=polar(m)
  if it==0:assert float((rn-a.polar(m)).abs().max())<1e-9
  zn={s:base[s][None]@rn for s in S};fixed=residual(zn,t)
  tn={s:target(zn[s],order,q,mode) for s in S};end=residual(zn,tn);delta=(rn-r).norm(dim=(1,2));cache_z=zn;cache_t=tn
  cycle=[None]*len(names)
  if it%100==0 and len(recent)>=48:
   for period in range(2,17):
    dist=torch.stack([(rn-recent[-k*period]).norm(dim=(1,2)) for k in [1,2,3]]).max(0).values.cpu().tolist()
    for j,er in enumerate(dist):
     if cycle[j] is None and er<1e-10 and float(delta[j])>1e-5:cycle[j]=dict(period=period,error=er)
  recent.append(rn.clone());recent=recent[-48:]
  values=torch.stack([obj,fixed,end,delta],dim=1).cpu().tolist()
  for j,n in enumerate(names):
   if done[j]:continue
   row=dict(step=len(hist[j])+1,before=values[j][0],after_rotation=values[j][1],after_target=values[j][2],rotation_change=values[j][3],target_increase=values[j][2]-values[j][1],total_increase=values[j][2]-values[j][0])
   if row['after_rotation']>row['before']+1e-8:raise RuntimeError(('Procrustes increased loss',n,row))
   hist[j].append(row);r[j]=rn[j]
   conv=len(hist[j])>10 and max(v['rotation_change'] for v in hist[j][-5:])<1e-5
   if conv or cycle[j] is not None or len(hist[j])>=limit:
    done[j]=True;np.savez_compressed(OUT/'fits'/f'{n}.npz',rotation=r[j].cpu().numpy())
    save(OUT/'fits'/f'{n}.json',dict(order=order,row_q=q,column_mode=mode,start=startmode,seed=seeds[j],steps=len(hist[j]),stop='fixed_point' if conv else ('cycle' if cycle[j] is not None else 'iteration_limit'),cycle=cycle[j],history=hist[j],seconds=time.time()-began,orthogonality_error=float((r[j].T@r[j]-torch.eye(256,device='cuda')).abs().max()),target_column_counts={s:(tn[s][j]!=0).sum(0).cpu().tolist() for s in S}))
  if it%100==0:
   for j,n in enumerate(names):
    if not done[j]:np.savez(OUT/'checkpoints'/f'{n}.npz',rotation=r[j].cpu().numpy());save(OUT/'checkpoints'/f'{n}.json',dict(history=hist[j]))
   save(OUT/'progress.json',dict(names=names,steps=[len(h) for h in hist],done=done,seconds=time.time()-began));print(names,[len(h) for h in hist],done,flush=True)
class RC_Eval(EnergyEval):
 def run_one(self,name,seed,r,source):
  w={s:self.base[s]@r for s in S};sg=positive_sign(w);w={s:w[s]*sg for s in S}
  for cut in ['k64','mass90']:
   dest=OUT/'eval'/f'{name}-{cut}-s{seed}.json'
   if dest.exists():continue
   ws={s:col_cut(w[s][None],cut)[0] for s in S}
   self.measure(f'{name}-{cut}',seed,ws,dict(coefficient_stats={s:coeff_stats(ws[s]) for s in S},cut=cut,source=source,orientation='shared sign maximizing combined positive coefficient energy'))

def main():
 save(OUT/'protocol.json',dict(rank=256,train='CC3M/CC3M',seeds=10,splits=20,methods='row first then column; column first then row; union of independently computed masks',row_q=[1,2],column_modes=['top64','90% squared coefficient energy'],energy_denominator='input to that column operation; row-first uses the already row-pruned column energy',starts='covariance solutions primary; previous identity1+random9 initial matrices sensitivity',objective='squared distance to target /2; target heuristic not claimed exact projection to intersection; no monotone objective assumption',stopping='5 consecutive R Frobenius changes <1e-5 after10 updates; cap10000; additionally stop as nonconverged cycle if periods2..16 recur at three successive periods within1e-10 while update>1e-5, checked every100 updates',evaluation='same saved20splits, own-caption labels, fixed50 categories; evaluate rotated matrix after independent final column-top64 or column-energy90 pruning; not the target matrix'))
 with np.load(ROOT/'runs/mapping-ablation-cc3m-2026-10-04/transforms/cross_svd_3221.npz') as f:base={s:a.tensor(f[s][:,:256]) for s in S}
 for mode in ['k64','mass90']:
  for order in ['rowfirst','colfirst','union']:
   for q in [1,2]:
    for start in ['covariance','original']:
     for first in [0,5]:fit(base,order,q,mode,start,list(range(first,first+5)))
 save(OUT/'fits-complete.json',dict(time=time.time()))
 e=RC_Eval()
 for seed in range(10):
  refs={'covariance':SRC/f'rank256/covariance-svd-{seed}-oriented.npz','fixed64':OLD/'fits'/f'alternating-k64-s{seed}.npz','row1':ROOT/'runs/row-rotation-2026-10-12/fits'/f'row1-covariance-s{seed}.npz','energy90':ROOT/'runs/energy-alternating-2026-10-11/fits'/f'ind90-s{seed}.npz'}
  for name,path in refs.items():e.run_one(name,seed,a.tensor(np.load(path)['rotation']),str(path))
  for mode in ['k64','mass90']:
   for order in ['rowfirst','colfirst','union']:
    for q in [1,2]:
     for start in ['covariance','original']:
      name=f'{order}-row{q}-{mode}-{start}';path=OUT/'fits'/f'{name}-s{seed}.npz';e.run_one(name,seed,a.tensor(np.load(path)['rotation']),str(path))
 save(OUT/'complete.json',dict(time=time.time(),fits=240,evaluations=560))
if __name__=='__main__':
 try:main()
 except Exception as ex:save(OUT/'error.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
