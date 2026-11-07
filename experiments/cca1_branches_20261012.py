"""Requested six continuations of fixed regularized CCA lambda=1, no new conditions."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
from types import SimpleNamespace
import json,time,traceback,argparse
import numpy as np
import torch
from scipy import sparse
from experiments import cc3m_score_rotation_20261012 as cc
from experiments import conditioned_svd_axes_cca_20261012 as ev
from experiments import row_column_rotation_20261012 as rc
m=cc.m;a=cc.a;S=['image','text'];ROOT=Path('/mnt/working/mm-sae');OUT=ROOT/'runs/cca1-branches-2026-10-12';AX=ROOT/'runs/conditioned-svd-axes-2026-10-12'
for part in ['fits','eval','auroc','checkpoints','bases']:(OUT/part).mkdir(parents=True,exist_ok=True)
def save(p,x):cc.save(p,x)
def log(**v):save(OUT/'progress.json',dict(time=time.time(),**v));print(v,flush=True)
m.OUT=OUT;m.log=log;cc.OUT=OUT;ev.OUT=OUT;ev.log=log
KINDS=['fixed64','row1','cycle-best','cross','mixed','both-b1']
def start(seed):return a.tensor(np.load(AX/'fits'/f'CCA-1-pullback-s{seed}.npz')['rotation'])
def target(w,kind):
 if kind=='fixed64':return cc.hard(w,64)
 if kind=='row1':return rc.row_cut(w,1)
 return rc.target(w[None],'rowfirst',1,'mass90')[0]
def alternate(base,kind,seed):
 name=f'{kind}-s{seed}';path=OUT/'fits'/f'{name}.npz'
 if path.exists():return
 r=start(seed);hist=[];recent=[];best=float('inf');bestr=r.clone();besti=0;beg=time.time();stop='iteration_limit';cycle=None
 for it in range(10001):
  z={s:base[s]@r for s in S};t={s:target(z[s],kind) for s in S};val=float(sum((z[s]-t[s]).square().sum() for s in S)/2)
  if val<best:best=val;bestr=r.clone();besti=it
  if it==10000:break
  rn=a.polar(sum(base[s].T@t[s] for s in S));fixed=float(sum((base[s]@rn-t[s]).square().sum() for s in S)/2);change=float((rn-r).norm())
  assert fixed<=val+1e-8*max(1,val),(kind,it,val,fixed)
  hist.append(dict(step=it,objective=val,best=best,after_rotation_fixed_target=fixed,rotation_change=change))
  if kind!='cycle-best' and len(hist)>1:assert val<=hist[-2]['objective']+1e-8*max(1,val)
  if it>10 and max(h['rotation_change'] for h in hist[-5:])<1e-5:stop='fixed_point';break
  if it%100==0 and len(recent)>=48:
   for period in range(2,17):
    if max(float((r-recent[-k*period]).norm()) for k in [1,2,3])<1e-10 and change>1e-5:stop='cycle';cycle=period;break
   if cycle:break
  recent.append(r.clone());recent=recent[-48:];r=rn
  if it%100==0:
   log(stage='alternating',kind=kind,seed=seed,step=it,objective=val,best=best,seconds=time.time()-beg)
   np.savez_compressed(OUT/'checkpoints'/f'{name}.npz',rotation=r.cpu().numpy(),best_rotation=bestr.cpu().numpy());save(OUT/'checkpoints'/f'{name}.json',dict(history=hist))
 if kind=='cycle-best' and cycle:
  phases=[];values=[]
  for phase in range(cycle):
   zz={s:base[s]@r for s in S};tt={s:target(zz[s],kind) for s in S};phases.append(r.clone());values.append(float(sum((zz[s]-tt[s]).square().sum() for s in S)/2));r=a.polar(sum(base[s].T@tt[s] for s in S))
  phase=int(np.argmin(values));r=phases[phase]
 elif kind=='cycle-best' and stop=='iteration_limit':r=bestr
 # Same common coefficient-positive sign as previous row/score branches.
 r=r*cc.positive_sign({s:base[s]@r for s in S})
 meta=dict(kind=kind,seed=seed,steps=it,stop=stop,cycle=cycle,best_step=besti,best_objective=best,history=hist,seconds=time.time()-beg,orthogonality=float((r.T@r-torch.eye(256,device='cuda')).abs().max()),start='saved CCA lambda1 covariance rotation of same seed',selection='minimum objective within terminal cycle; terminal state for fixed-point convergence; best visited if capped' if kind=='cycle-best' else 'terminal state',sign='shared positive coefficient energy')
 save(path.with_suffix('.json'),meta);np.savez_compressed(path,rotation=r.cpu().numpy(),image=(base['image']@r).cpu().numpy(),text=(base['text']@r).cpu().numpy())
class Objective(cc.Objective):
 def __init__(self,e,data):
  self.x=[a.tensor(data[s]) for s in S];self.n=len(self.x[0]);self.second=[x.T@x/self.n for x in self.x];self.cov=[q-torch.outer(x.mean(0),x.mean(0)) for x,q in zip(self.x,self.second)]
  self.l=[]
  for s in S:
   l=e.cov[s]@e.base[s];self.l.append(l/l.norm(dim=1,keepdim=True).clamp_min(1e-12))
  vals,_=self.components(start(0),False);self.scales={('var',False):abs(float(vals[0])),('cross',False):abs(float(vals[1])),('sum',False):abs(float(vals[0]+vals[1])),('loading',False):abs(float(vals[2]))};self.cal=None
  save(OUT/'objective-scales.json',{str(k):v for k,v in self.scales.items()})
def train():
 z=np.load(AX/'bases/CCA-1-pullback-coefficients.npz');base={s:a.tensor(z[s+'_active']) for s in S}
 save(OUT/'protocol.json',dict(space='CC3M moments, regularized CCA lambda1, rank256, pullback coefficients',branches=KINDS,seeds=list(range(5)),splits=20,score_pairs=200000,score_steps=500,score_standardized=False,score_sampling='same saved CC3M rows as previous C score experiment',score_scales='fixed at CCA covariance seed0; mixed uses objective magnitudes; both penalties each match tangent gradient norm at same-seed starting R',alternating_limit=10000,cycle_selection='minimum objective within terminal cycle as historical C variant; fixed point uses terminal state; no evaluation use',sign='before evaluation all new branches use shared third moment on the same32768 CC3M training pairs as CCA covariance baseline; training stores provisional positive coefficient energy signs',F1=False))
 for kind in KINDS[:3]:
  for seed in range(5):alternate(base,kind,seed)
 with np.load(ROOT/'runs/mapping-suite-cc3m-2026-10-04/moments.npz') as f:
  mu=f['fit_mean'];sd=np.sqrt(f['fit_second'].diagonal()-mu*mu);cv=(f['fit_second']-np.outer(mu,mu))/sd[:,None]/sd[None,:];ni=len(f['image_ids']);ids={s:f[s+'_ids'] for s in S}
 cov={'image':a.tensor(cv[:ni,:ni]),'text':a.tensor(cv[ni:,ni:])};del cv
 cache=OUT/'cc3m-pairs.npz'
 if cache.exists():data=dict(np.load(cache))
 else:
  selected=np.load(ROOT/'runs/cc3m-score-rotation-2026-10-12/sample-indices.npz')['activation_rows'];data={}
  for s,sl in [('image',slice(0,ni)),('text',slice(ni,None))]:
   raw=sparse.load_npz(ROOT/'runs/cc3m-sae-2026-10-04/activations/train2017'/f'{s}.npz').tocsr();w=base[s].cpu().numpy()/sd[sl,None];data[s]=np.asarray(raw[selected][:,ids[s]]@w)-mu[sl]@w;del raw
  np.savez_compressed(cache,**data);assert len(data['image'])==200000
 e=SimpleNamespace(base=base,cov=cov);cc.validate(e,data);obj=Objective(e,data)
 for kind in ['cross','mixed','both']:
  name=kind if kind!='both' else 'both-b1'
  for seed in range(5):
   r=start(seed);obj.calibrate(r,kind,1. if kind=='both' else 0.)
   r,meta=m.fit_score(obj,r,kind,False,f'{name}-s{seed}',500);r=r*cc.positive_sign({s:base[s]@r for s in S});meta['calibration']=obj.cal;meta['sign']='shared positive coefficient energy';meta['start']='CCA lambda1 covariance rotation of same seed'
   # Readiness is a separate marker because fit_score writes its file first.
   np.savez_compressed(OUT/'fits'/f'{name}-s{seed}.npz',rotation=r.cpu().numpy(),image=(base['image']@r).cpu().numpy(),text=(base['text']@r).cpu().numpy());save(OUT/'fits'/f'{name}-s{seed}.json',meta);save(OUT/'fits'/f'{name}-s{seed}.ready.json',dict(time=time.time()))
 save(OUT/'train-complete.json',dict(time=time.time(),fits=30))
def evaluate():
 cfg=ev.p.old.load_config(ROOT/ev.p.CONFIG['cc3m-cc3m']);data,parents=ev.p.eval_population(cfg,'test',OUT);eligible=np.isin([c['id'] for c in data['concepts']],data['metadata']['eligible_category_ids']);assert eligible.sum()==50
 labels=[data['labels'][s][:,eligible].astype(float) for s in S];data['selected_concepts']=[c for c,ok in zip(data['concepts'],eligible) if ok];save(OUT/'categories.json',data['selected_concepts']);ss=np.load(ev.REF/'evaluation-splits.npz');splits=[(ss[f'image_{j}'],ss[f'text_{j}']) for j in range(20)]
 # Same training-only polarity convention as covariance references, without rotating axes.
 f=np.load(ROOT/'runs/mapping-suite-cc3m-2026-10-04/moments.npz');ni=len(f['image_ids']);mu=f['fit_mean'];sd=np.sqrt(f['fit_second'].diagonal()-mu*mu);indices=np.load(ev.REF/'split.npz');train=[]
 for s,key in zip(S,['fit_rows','fit_text_rows']):
  raw=sparse.load_npz(ROOT/'runs/cc3m-sae-2026-10-04/activations/train2017'/f'{s}.npz').tocsr();train.append(raw[indices[key]][:,f[s+'_ids']]);del raw
 for stat in ['C','CCA-0.01','CCA-1']:
  z=np.load(AX/'bases'/f'{stat}-pullback-coefficients.npz');base=[z[s+'_active'] for s in S];skew=0
  for raw,w,sl in zip(train,base,[slice(0,ni),slice(ni,None)]):o=np.asarray(raw@(w/sd[sl,None]))-mu[sl]@(w/sd[sl,None]);skew=skew+(o**3).mean(0)
  r=np.diag(np.where(skew<0,-1.,1.));ret=json.loads((AX/'bases'/f'{stat}-pullback-retrieval.json').read_text());name='unrotated-'+stat
  ev.measure(name,0,base,r,dict(no_rotation=True,sign='common training third-moment sign',statistic=stat),data,parents,labels,splits,ret)
  np.savez_compressed(OUT/'bases'/f'{name}.npz',image=base[0]@r,text=base[1]@r)
 z=np.load(AX/'bases/CCA-1-pullback-coefficients.npz');base=[z[s+'_active'] for s in S];ret=json.loads((AX/'bases/CCA-1-pullback-retrieval.json').read_text())
 orient=[np.asarray(raw@(w/sd[sl,None]))-mu[sl]@(w/sd[sl,None]) for raw,w,sl in zip(train,base,[slice(0,ni),slice(ni,None)])];del train
 for kind in KINDS:
  for seed in range(5):
   path=OUT/'fits'/f'{kind}-s{seed}.npz';ready=path if kind in KINDS[:3] else path.with_suffix('.ready.json')
   while not ready.exists():
    if (OUT/'train-error.json').exists():raise RuntimeError((OUT/'train-error.json').read_text())
    time.sleep(10)
   r=np.load(path)['rotation'];meta=json.loads(path.with_suffix('.json').read_text());skew=sum(((o@r)**3).mean(0) for o in orient);r*=np.where(skew<0,-1.,1.);meta['evaluation_sign']='shared third moment on same32768 CC3M training pairs as CCA covariance baseline';np.savez_compressed(path,rotation=r,image=base[0]@r,text=base[1]@r);save(path.with_suffix('.json'),meta);ev.measure(kind,seed,base,r,meta,data,parents,labels,splits,ret)
 save(OUT/'eval-complete.json',dict(time=time.time(),new_evaluations=66))
if __name__=='__main__':
 arg=argparse.ArgumentParser();arg.add_argument('phase',choices=['train','evaluate']);args=arg.parse_args()
 try:train() if args.phase=='train' else evaluate()
 except Exception as ex:save(OUT/f'{args.phase}-error.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
