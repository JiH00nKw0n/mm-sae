"""Frozen baseline/legacy controls and per-column coefficient-energy pruning."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import argparse,json,time,traceback,hashlib
from pathlib import Path
import numpy as np,torch
from experiments import rotation_audit_20261011 as a
p=a.p;ROOT=a.ROOT;SRC=a.OUT;OUT=ROOT/'runs/rotation-mass-audit-2026-10-11';OUT.mkdir(exist_ok=True)
S=['image','text'];MASS=[.90,.95,.99]
def save(path,d):p.save(path,d)
def status(**x):save(OUT/'progress.json',dict(time=time.time(),**x));print(json.dumps(x),flush=True)
def prune(w,frac):
 order=torch.argsort(w.square(),dim=0,descending=True,stable=True);en=w.square().gather(0,order);total=en.sum(0);n=(en.cumsum(0)<frac*total).sum(0)+1;n=torch.where(total>0,n,0)
 mask=torch.arange(len(w),device=w.device)[:,None]<n[None,:];keep=torch.zeros_like(w).scatter_(0,order,mask.to(w.dtype));out=w*keep
 kept=(out.square().sum(0)/total.clamp_min(1e-30));assert bool((kept[total>0]>=frac-1e-12).all())
 return out,dict(mean=float(n.double().mean()),median=float(n.double().median()),min=int(n.min()),max=int(n.max()),counts=n.cpu().tolist(),retained_mass_min=float(kept[total>0].min()))
class Evaluator:
 def __init__(self):
  self.cfg=p.old.load_config(ROOT/p.CONFIG['cc3m-cc3m']);data,self.parents=p.eval_population(self.cfg,'test',OUT);self.data=data
  eligible=np.isin([c['id'] for c in data['concepts']],data['metadata']['eligible_category_ids']);assert eligible.sum()==50
  self.labels={s:a.tensor(data['labels'][s][:,eligible].astype(float)) for s in S};self.x={s:a.tensor((data['raw'][s].toarray()-data['means'][s])/data['scales'][s]) for s in S}
  z=np.load(SRC/'evaluation-splits.npz');self.splits=[(torch.as_tensor(z[f'image_{i}'],device='cuda'),torch.as_tensor(z[f'text_{i}'],device='cuda')) for i in range(20)]
  for ai,bt in [('image_a','text_b'),('image_b','text_a')]:self.splits.append((torch.as_tensor(np.flatnonzero(data['masks'][ai]),device='cuda'),torch.as_tensor(np.flatnonzero(data['masks'][bt]),device='cuda')))
  self.fit=None
 def oriented(self,w):
  if self.fit is None:
   _,arr,_=p.prepare('cc3m-cc3m',OUT,32768,0);self.fit={s:(arr['fit',s]-arr['mean',s]).double() for s in S};del arr
  sign=torch.where(sum((self.fit[s]@w[s]).pow(3).mean(0) for s in S)<0,-1.,1.)
  return {s:w[s]*sign for s in S}
 def eval(self,name,rank,w,frac=None,metadata=None):
  folder=OUT/f'rank{rank}';folder.mkdir(exist_ok=True);tag='dense' if frac is None else f'mass{round(frac*100)}';dest=folder/f'{name}-{tag}.json'
  if dest.exists():return
  stats={};ww={}
  for s in S:
   if frac is None:
    ww[s]=w[s];n=(w[s]!=0).sum(0).double();stats[s]=dict(mean=float(n.mean()),median=float(n.median()),min=int(n.min()),max=int(n.max()))
   else:ww[s],stats[s]=prune(w[s],frac)
  status(stage='evaluate',rank=rank,method=name,variant=tag)
  sc={s:self.x[s]@ww[s] for s in S};match=a.matching(sc,self.labels,self.splits)
  ret=p.paired_retrieval(sc['image'].cpu().numpy(),sc['text'].cpu().numpy(),self.parents,device='cuda',chunk_size=128);ret={d:{k:v for k,v in m.items() if k not in ['hits','ranks']} for d,m in ret.items()}
  save(dest,dict(name=name,rank=rank,variant=tag,coefficient_energy_fraction=frac,counts=stats,retrieval=ret,matching=match,metric_version=2,metadata=metadata or {}))
 def matrices(self,path,rank):
  with np.load(path) as z:return {s:a.tensor(z[s][:,:rank].astype(float)) for s in S}
 def baselines(self):
  for rank in [128,256,512]:
   for model,stem in [('pls','cross_svd'),('cca','cca')]:
    w=self.oriented(self.matrices(ROOT/f'runs/mapping-ablation-cc3m-2026-10-04/transforms/{stem}_3221.npz',rank));self.eval(model,rank,w,metadata=dict(initialization='deterministic SVD solution, common training-skew sign convention'))
  # Unmodified Hungarian is full one-to-one matching, not a rank-k learned basis.
  cfg=self.cfg
  with np.load(Path(cfg['parent_run'])/'moments.npz') as z:
   ni=len(z['image_ids']);mu=z['fit_mean'];cv=z['fit_second']-np.outer(mu,mu)
  sd=np.sqrt(np.maximum(cv.diagonal(),0));sd[sd==0]=1;cv/=sd[:,None]*sd[None,:]
  mp=np.load(next(m['path'] for m in cfg['models'] if m['key']=='hungarian'))['mapping'];i,j=np.nonzero(mp);assert len(np.unique(i))==len(i) and len(np.unique(j))==len(j)
  order=np.argsort(-cv[i,ni+j],kind='stable');i=i[order];j=j[order]
  for rank in [128,256,512,len(i)]:
   w={'image':torch.zeros((mp.shape[0],rank),device='cuda',dtype=torch.float64),'text':torch.zeros((mp.shape[1],rank),device='cuda',dtype=torch.float64)};w['image'][torch.as_tensor(i[:rank],device='cuda'),torch.arange(rank,device='cuda')]=1;w['text'][torch.as_tensor(j[:rank],device='cuda'),torch.arange(rank,device='cuda')]=1
   self.eval('hungarian-full' if rank==len(i) else 'hungarian-toplinks',rank,w,metadata=dict(selection='Full matching' if rank==len(i) else 'Top-k learned edges by CC3M training coactivation correlation; additional restricted candidate control, not canonical rank-k Hungarian',signs='Original positive activation directions retained',retrieval='Cosine over matched original coordinates on both sides'))
  # Keep the old oriented weights frozen; only evaluation split changes.
  for stat in ['covariance','correlation']:
   for seed in range(3):
    path=ROOT/f'runs/pls-cca-gap-2026-10-10/pls-{stat}-{seed}-transforms.npz';self.eval(f'legacy-{stat}-{seed}',256,self.matrices(path,256),metadata=dict(source=str(path),signs='Frozen old evaluation signs, no reorientation'))
  save(OUT/'baselines-complete.json',dict(time=time.time()))
 def masses(self):
  # Reuse learned rotations. Do not retrain or pick seeds using labels.
  for rank in [128,256,512]:
   for model,stem in [('pls','cross_svd'),('cca','cca')]:
    w=self.oriented(self.matrices(ROOT/f'runs/mapping-ablation-cc3m-2026-10-04/transforms/{stem}_3221.npz',rank))
    for frac in MASS:self.eval(model,rank,w,frac)
   base=self.matrices(ROOT/'runs/mapping-ablation-cc3m-2026-10-04/transforms/cross_svd_3221.npz',rank)
   for method in ['covariance-svd','correlation-cayley','coefficient-svd']:
    for seed in range(10):
     path=SRC/f'rank{rank}/{method}-{seed}-oriented.npz'
     while not path.exists():
      if (SRC/'error.json').exists():raise RuntimeError('Parent rotation experiment failed')
      status(stage='waiting_rotation',rank=rank,method=method,seed=seed);time.sleep(30)
     rr=a.tensor(np.load(path)['rotation']);w={s:base[s]@rr for s in S}
     for frac in MASS:self.eval(f'{method}-{seed}',rank,w,frac)
  save(OUT/'mass-complete.json',dict(time=time.time()))
 def supports(self):
  # Support targets selected only from trained coefficients, not retrieval/matching.
  rows=[]
  for rank in [128,256,512]:
   folder=OUT/f'rank{rank}';groups={}
   for path in folder.glob('*-mass*.json'):
    z=json.loads(path.read_text());family=z['name'].rsplit('-',1)[0] if z['name'][-1].isdigit() else z['name'];key=(family,z['variant']);groups.setdefault(key,[]).append(z)
   for (family,var),zs in groups.items():
    mi=np.mean([z['counts']['image']['mean'] for z in zs]);mt=np.mean([z['counts']['text']['mean'] for z in zs]);target=(mi+mt)/2
    # Fixed coefficient-count grid; report actual count mismatch on each modality.
    step=8 if target<128 else 16 if target<512 else 32;k=max(1,int(round(target/step)*step));rows.append(dict(rank=rank,family=family,variant=var,image_count=float(mi),text_count=float(mt),pooled_target=float(target),sparse_cca_support=k,relative_gap=float(abs(k-target)/target)))
  save(OUT/'support-targets.json',rows)
  # Fit jobs per support can reuse all leading sequential components across requested ranks.
  jobs={}
  for z in rows:jobs[z['sparse_cca_support']]=max(jobs.get(z['sparse_cca_support'],0),z['rank'])
  jobs[16]=512;save(OUT/'support-jobs.json',[dict(support=k,rank=r) for k,r in sorted(jobs.items())])
 def sparse_evaluate(self):
  plan=json.loads((OUT/'support-jobs.json').read_text())
  for job in plan:
   path=OUT/'sparse'/f"cca-k{job['support']}-r{job['rank']}.npz"
   while not path.exists():time.sleep(30)
   for rank in [128,256,512]:
    if rank<=job['rank']:
     w=self.oriented(self.matrices(path,rank));self.eval(f"sparse-cca-k{job['support']}",rank,w,metadata=dict(initialization='Deterministic sequential sparse ridge CCA initialized by dense CCA, no artificial random repetitions',support_target_source='Trained coefficient counts only'))
  save(OUT/'all-complete.json',dict(time=time.time()))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['baseline','mass','sparse'],required=True);args=parser.parse_args()
 try:
  save(OUT/'protocol.json',dict(condition='CC3M/CC3M',ranks=[128,256,512],mass_definition='Retain smallest per-column descending-absolute-coefficient support reaching 90/95/99% of sum of squared coefficients; no refit, no coefficient/output rescaling.',signs='Common training third-moment sign orientation, frozen before pruning. Hungarian keeps original positive coordinate polarity.',matching='Identical fixed50 categories, same20 image-disjoint partitions plus old fixed split and swap, native paired coordinate agreement.',initializations='10 actual learned rotations. Dense SVD/CCA and Hungarian deterministic. Sequential Sparse CCA uses fixed dense initialization, no arbitrary duplicates.',confidence='Paired crossed seed and partition bootstrap, conditional on fixed val images/categories.',sparse_budget='Nearest predeclared support grid to pooled mean image/text retained counts; report each side and mismatch, no test-score selection.'))
  e=Evaluator()
  if args.phase=='baseline':e.baselines()
  elif args.phase=='mass':
   e.masses();e.supports()
  else:e.sparse_evaluate()
 except Exception as exc:save(OUT/f'error-{args.phase}.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
