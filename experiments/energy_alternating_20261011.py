"""Energy-fraction alternating shared rotation, fixed protocol, no evaluation tuning."""
from experiments.sparse_improvement_20261011 import *
OLD=OUT
OUT=ROOT/'runs/energy-alternating-2026-10-11'
OUT.mkdir(exist_ok=True)
for sub in ['fits','eval','checkpoints']:(OUT/sub).mkdir(exist_ok=True)
torch.set_num_threads(2)

def truncate(w,fraction=None,k=None):
 ix=torch.argsort(w.abs(),dim=-2,descending=True,stable=True)
 if k is None:
  v=w.gather(-2,ix).square();num=(v.cumsum(-2)<fraction*v.sum(-2,keepdim=True)).sum(-2)+1
 else:
  num=torch.ones(w.shape[0],w.shape[-1],device=w.device,dtype=torch.long)*torch.as_tensor(k,device=w.device).reshape(-1,1)
 mask=torch.arange(w.shape[-2],device=w.device)[None,:,None]<num[:,None,:]
 return w*torch.zeros_like(w).scatter_(-2,ix,mask.double())

def polar(m):
 # Positive rescaling preserves the polar factor. Frobenius scaling puts all
 # singular values in (0,1], the Newton-Schulz convergence interval.
 q=m/m.norm(dim=(-2,-1),keepdim=True);eye=torch.eye(256,device=m.device,dtype=m.dtype)
 for _ in range(40):
  gram=q.transpose(-2,-1)@q
  if float((gram-eye).abs().max())<2e-14:break
  q=.5*q@(3*eye-gram)
 if float((q.transpose(-2,-1)@q-eye).abs().max())>1e-11:return a.polar(m)
 return q

def fit_batch(base,names,seeds,starts,fraction=None,ks=None,limit=10000):
 todo=[i for i,name in enumerate(names) if not (OUT/'fits'/f'{name}.npz').exists()]
 if not todo:return
 names=[names[i] for i in todo];seeds=[seeds[i] for i in todo];starts=[starts[i] for i in todo]
 ks=None if ks is None else [ks[i] for i in todo]
 r=torch.stack(starts);hist=[[] for _ in names];done=[False]*len(names);steps=[0]*len(names)
 # Resume only this exact per-condition checkpoint.
 for j,name in enumerate(names):
  cp=OUT/'checkpoints'/f'{name}.npz';jp=cp.with_suffix('.json')
  if cp.exists() and jp.exists():
   r[j]=a.tensor(np.load(cp)['rotation']);data=json.loads(jp.read_text());hist[j]=data['history'];steps[j]=data['step']
 start=time.time();eye=torch.eye(256,device='cuda',dtype=torch.float64)
 for iteration in range(limit):
  if all(done):break
  z={s:base[s][None]@r for s in S};sp={s:truncate(z[s],fraction,ks) for s in S}
  m=sum(base[s].T[None]@sp[s] for s in S);rn=polar(m)
  if iteration==0:
   sv=a.polar(m);err=float((rn-sv).abs().max());assert err<1e-10,err
   save(OUT/f'polar-validation-{names[0]}.json',dict(max_difference_from_svd=err))
  delta=(rn-r).norm(dim=(1,2));objective=sum((z[s]-sp[s]).square().sum((1,2)) for s in S)/2
  counts={s:(sp[s]!=0).sum(1).double().mean(1).cpu().tolist() for s in S}
  for j,name in enumerate(names):
   if done[j]:continue
   steps[j]+=1;d=float(delta[j]);obj=float(objective[j]);hist[j].append(dict(step=steps[j],residual=obj,rotation_change=d,image_count=counts['image'][j],text_count=counts['text'][j]))
   # A small fixed-point residual, rather than assumed monotone energy objective.
   converged=steps[j]>10 and len(hist[j])>=5 and max(q['rotation_change'] for q in hist[j][-5:])<1e-5
   capped=steps[j]>=limit
   if converged or capped:
    r[j]=rn[j];done[j]=True
    meta=dict(seed=seeds[j],fraction=fraction,k=None if ks is None else ks[j],steps=steps[j],stop='fixed_point' if converged else 'iteration_limit',rotation_change=d,history=hist[j],seconds=time.time()-start,orthogonality_max_error=float((rn[j].T@rn[j]-eye).abs().max()),column_renormalization=False)
    save(OUT/'fits'/f'{name}.json',meta);np.savez_compressed(OUT/'fits'/f'{name}.npz',rotation=rn[j].cpu().numpy())
   else:r[j]=rn[j]
  if iteration%100==0:
   for j,name in enumerate(names):
    if done[j]:continue
    np.savez(OUT/'checkpoints'/f'{name}.npz',rotation=r[j].cpu().numpy());save(OUT/'checkpoints'/f'{name}.json',dict(step=steps[j],history=hist[j]))
   progress=dict(names=names,steps=steps,done=done,rotation_change=delta.cpu().tolist(),counts=counts,seconds=time.time()-start)
   save(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)

def fit_all():
 with np.load(ROOT/'runs/mapping-ablation-cc3m-2026-10-04/transforms/cross_svd_3221.npz') as f:base={s:a.tensor(f[s][:,:256]) for s in S}
 # Run smaller independent batches to remain inside the allocated 10 GiB GPU.
 for variant,frac,parent in [('ind99',.99,None),('ind95',.95,None),('ind90',.90,None),('seq95',.95,'ind99'),('seq90',.90,'seq95')]:
  for first in [0,5]:
   seeds=list(range(first,first+5));starts=[initial(s) if parent is None else a.tensor(np.load(OUT/'fits'/f'{parent}-s{s}.npz')['rotation']) for s in seeds]
   fit_batch(base,[f'{variant}-s{s}' for s in seeds],seeds,starts,fraction=frac)
 save(OUT/'energy-fits-complete.json',dict(time=time.time()))
 # Match each seed's average nonzero count across image and text to nearest integer.
 # Budgets depend only on trained matrices, never on held-out metrics.
 for variant,frac in [('ind99',.99),('ind95',.95),('ind90',.9),('seq95',.95),('seq90',.9)]:
  targets=[]
  for seed in range(10):
   r=a.tensor(np.load(OUT/'fits'/f'{variant}-s{seed}.npz')['rotation'])
   count=sum(float((truncate((base[s]@r)[None],fraction=frac)!=0).sum(-2).double().mean()) for s in S)/2
   targets.append(max(1,int(np.floor(count+.5))))
  save(OUT/f'matched-budgets-{variant}.json',dict(budgets=targets))
  for first in [0,5]:
   seeds=list(range(first,first+5));names=[f'fixed-k{targets[s]}-s{s}' for s in seeds]
   # Reuse identical fixed-k fits already measured, retaining their stop flags.
   for name,seed in zip(names,seeds):
    old=OLD/'fits'/f'alternating-k{targets[seed]}-s{seed}.npz'
    if old.exists() and not (OUT/'fits'/f'{name}.npz').exists():
     import shutil
     shutil.copy2(old,OUT/'fits'/f'{name}.npz');shutil.copy2(old.with_suffix('.json'),OUT/'fits'/f'{name}.json')
   fit_batch(base,names,seeds,[initial(s) for s in seeds],ks=[targets[s] for s in seeds])
 save(OUT/'all-fits-complete.json',dict(time=time.time()))

class EnergyEval(Eval):
 def measure(self,name,seed,ws,meta):
  dest=OUT/'eval'/f'{name}-s{seed}.json'
  if dest.exists():return
  sc={s:self.x[s]@ws[s] for s in S};rows=[];axisrows={s:[] for s in S}
  for pair in self.splits[:20]:
   au=[a.gpu_auc(sc[s][ix],self.labels[s][ix]) for s,ix in zip(S,pair)]
   for s,q in zip(S,au):axisrows[s].append({str(t):(q>=t).sum(0).cpu().tolist() for t in [.8,.9,.95]})
   valid=torch.isfinite(au[0])&torch.isfinite(au[1])
   for q in au:q[~valid]=-torch.inf
   ind=[q.argmax(0) for q in au];col=torch.arange(50,device='cuda');v=[q[k,col] for q,k in zip(au,ind)];match=ind[0]==ind[1]
   row=dict(count=int(match.sum()),per_category=match.int().cpu().tolist())
   for t in [.7,.8,.9]:row[str(t)]=int((match&(v[0]>=t)&(v[1]>=t)).sum())
   rows.append(row)
  counts={}
  for s in S:
   nn=(ws[s]!=0).sum(0).cpu().numpy();counts[s]=dict(mean=float(nn.mean()),median=float(np.median(nn)),min=int(nn.min()),max=int(nn.max()),per_axis=nn.tolist())
  save(dest,dict(name=name,seed=seed,counts=counts,retrieval=retrieval(sc,self.parents),matching=rows,axis_counts=axisrows,meta=meta));print('evaluated',name,seed,flush=True)

def eval_all():
 e=EnergyEval()
 for seed in range(10):
  for variant,frac in [('ind99',.99),('ind95',.95),('ind90',.9),('seq95',.95),('seq90',.9)]:
   if not (OUT/'eval'/f'{variant}-s{seed}.json').exists():
    path=OUT/'fits'/f'{variant}-s{seed}.npz';r=e.oriented_rotation(a.tensor(np.load(path)['rotation']));w={s:truncate((e.base[s]@r)[None],frac)[0] for s in S};e.measure(variant,seed,w,json.loads(path.with_suffix('.json').read_text())|dict(history=None))
   k=json.loads((OUT/f'matched-budgets-{variant}.json').read_text())['budgets'][seed];name=f'fixed-k{k}'
   if not (OUT/'eval'/f'{name}-s{seed}.json').exists():
    path=OUT/'fits'/f'{name}-s{seed}.npz';r=e.oriented_rotation(a.tensor(np.load(path)['rotation']));e.measure(name,seed,{s:hard(e.base[s]@r,k) for s in S},json.loads(path.with_suffix('.json').read_text())|dict(history=None))
  for method in ['covariance-svd','correlation-cayley']:
   r=a.tensor(np.load(SRC/f'rank256/{method}-{seed}-oriented.npz')['rotation']);w={s:e.base[s]@r for s in S}
   for frac in [None,.99,.95,.9]:
    name=method+('-dense' if frac is None else f'-mass{round(frac*100)}')
    if not (OUT/'eval'/f'{name}-s{seed}.json').exists():e.measure(name,seed,w if frac is None else {s:truncate(w[s][None],frac)[0] for s in S},dict(source='original rotation'))
  if not (OUT/'eval'/f'fixed64-s{seed}.json').exists():
   path=OLD/'fits'/f'alternating-k64-s{seed}.npz';r=e.oriented_rotation(a.tensor(np.load(path)['rotation']));e.measure('fixed64',seed,{s:hard(e.base[s]@r,64) for s in S},json.loads(path.with_suffix('.json').read_text())|dict(history=None))
 save(OUT/'complete.json',dict(time=time.time(),seeds=10,splits=20,rank=256))

if __name__=='__main__':
 try:
  save(OUT/'protocol.json',dict(rank=256,seeds=10,splits=20,training='CC3M SAE / CC3M mapping',fractions=[.99,.95,.9],continuation=[.99,.95,.9],matching='fixed 50 concepts, own-caption text labels, same 20 saved splits',stopping='five consecutive Frobenius rotation changes <1e-5, maximum10000 updates',pruning='minimum count attaining per-column coefficient squared-energy fraction, no column rescaling',polar='Newton-Schulz polar, verified against SVD, SVD fallback',matched_fixed='per initialization nearest integer to pooled image/text mean count; determined only from trained weights',reference='SPCArt T-en; original paper normalizes columns, this controlled variant does not'))
  fit_all();eval_all()
 except Exception as ex:save(OUT/'error.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
