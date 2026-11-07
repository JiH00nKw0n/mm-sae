"""Frozen true rotations: participation ratio and permuted-pair coefficient thresholds."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
from types import SimpleNamespace
import json,time,traceback,shutil,hashlib,argparse
import numpy as np
from scipy import sparse,linalg,stats
import torch
from experiments import conditioned_svd_axes_cca_20261012 as ev
from experiments import rotation_audit_20261011 as a
from experiments import cca1_pruning_20261012 as prior
ROOT=ev.ROOT;AX=prior.AX;OUT=ROOT/'runs/cca1-adaptive-pruning-2026-10-13';S=['image','text'];torch.set_num_threads(2)
for d in ['eval','auroc','null','weights']:(OUT/d).mkdir(parents=True,exist_ok=True)
def save(p,x):ev.save(p,x)
def log(**v):save(OUT/'progress.json',dict(time=time.time(),**v));print(v,flush=True)
def top(w,k):
 ix=np.argsort(-abs(w),axis=0,kind='stable');v=np.take_along_axis(w,ix,0);out=np.zeros_like(w);np.put_along_axis(out,ix,v*(np.arange(len(w))[:,None]<np.asarray(k)),0);return out
def pr_cut(w):
 s=(w*w).sum(0);pr=s*s/np.maximum((w**4).sum(0),1e-300);k=np.clip(np.floor(pr+.5).astype(int),1,len(w));return top(w,k),pr

def describe(w):
 n=(w!=0).sum(0);q=ev.coeffstats(w);v=np.sort(w*w,axis=0)[::-1];n90=np.where(n>0,(v.cumsum(0)<.9*v.sum(0)).sum(0)+1,0)
 q.update(mean=float(n.mean()),minimum=int(n.min()),median=float(np.median(n)),maximum=int(n.max()),empty_axes=int((n==0).sum()),counts=n.tolist(),n90_mean=float(n90.mean()),n90_median=float(np.median(n90)),n90_min=int(n90.min()),n90_max=int(n90.max()));return q

def null_thresholds():
 dst=OUT/'thresholds.json'
 if dst.exists():return json.loads(dst.read_text())
 z=np.load(ROOT/'runs/mapping-suite-cc3m-2026-10-04/moments.npz');ids={s:z[s+'_ids'] for s in S};ni=len(ids['image']);mu=z['fit_mean'];sec=z['fit_second'];sd=np.sqrt(sec.diagonal()-mu*mu);cv=(sec-np.outer(mu,mu))/sd[:,None]/sd[None,:];cov=[cv[:ni,:ni].copy(),cv[ni:,ni:].copy()];del cv,sec
 log(stage='marginal_inverse_square_roots');lx=[]
 for c in cov:
  val,vec=linalg.eigh(c,check_finite=False);lx.append((vec/np.sqrt(val+1))@vec.T)
 rows=np.load(ROOT/'runs/cc3m-caption-concepts-2026-10-09/extraction/activation-rows.npy');N=len(rows);assert N==int(z['fit_n']);raw=[]
 for s in S:
  log(stage='load_CC3M_activations',side=s,pairs=N);f=ROOT/'runs/cc3m-sae-2026-10-04/activations/train2017'/f'{s}.npz';v=sparse.load_npz(f).tocsr();v=v[rows][:,ids[s]].tocsr();raw.append(v)
 # Check that the retained pairs are exactly the source population of the frozen moments.
 audit=[]
 for j,v in enumerate(raw):
  sl=slice(0,ni) if j==0 else slice(ni,None);mm=np.bincount(v.indices,weights=v.data,minlength=v.shape[1])/N;err=float(abs(mm-mu[sl]).max());assert err<1e-5,(j,err);audit.append(err)
 save(OUT/'null/population.json',dict(pairs=N,mean_max_abs_error=audit,rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),active_features={s:len(ids[s]) for s in S}))
 for seed in range(3):
  fp=OUT/'null'/f'shuffle{seed}-coefficients.npz'
  if fp.exists():continue
  cp=OUT/'null'/f'shuffle{seed}-cross.npz';beg=time.time()
  if cp.exists():C=np.load(cp)['cross']
  else:
   perm=np.random.default_rng(131300+seed).permutation(N);np.save(OUT/'null'/f'shuffle{seed}-permutation.npy',perm);cross=torch.zeros((ni,len(ids['text'])),device='cuda',dtype=torch.float64)
   for st in range(0,N,16384):
    end=min(st+16384,N);x=torch.as_tensor(raw[0][st:end].toarray(),device='cuda',dtype=torch.float64);y=torch.as_tensor(raw[1][perm[st:end]].toarray(),device='cuda',dtype=torch.float64);cross.addmm_(x.T,y);del x,y
    if st%(16384*10)==0:log(stage='shuffled_cross_moment',shuffle=seed,done=end,total=N,seconds=time.time()-beg)
   C=(cross.cpu().numpy()/N-np.outer(mu[:ni],mu[ni:]))/sd[:ni,None]/sd[None,ni:];del cross;torch.cuda.empty_cache();np.savez_compressed(cp,cross=C)
  log(stage='null_svd',shuffle=seed,seconds=time.time()-beg);M=lx[0]@C@lx[1];u,d,vt=linalg.svd(M,full_matrices=False,check_finite=False);u=u[:,:256];v=vt[:256].T;res=float(np.linalg.norm(M@v-u*d[:256])/np.linalg.norm(u*d[:256]));assert res<1e-8
  base=[lx[0]@u,lx[1]@v];loads=[]
  for c,w in zip(cov,base):q=c@w;q/=np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-12);loads.append(a.tensor(q))
  a.OUT=OUT/'null'/f'shuffle{seed}';a.OUT.mkdir(exist_ok=True);r=a.fit(256,'covariance','svd',0,loads,None,SimpleNamespace(steps=20000,grad_tol=1e-6,obj_tol=1e-10));meta=json.loads((a.OUT/'rank256/covariance-svd-0-fit.json').read_text());ws=[w@r for w in base]
  np.savez_compressed(fp,image=ws[0],text=ws[1],rotation=r,singular_values=d[:256]);save(fp.with_suffix('.json'),dict(shuffle=seed,permutation_seed=131300+seed,rotation_initialization='identity for each shuffle; same baseline solver',svd_residual=res,fit=meta,seconds=time.time()-beg));log(stage='null_complete',shuffle=seed,seconds=time.time()-beg)
 del raw
 thresholds={};stability={}
 for side in S:
  vals=[abs(np.load(OUT/'null'/f'shuffle{k}-coefficients.npz')[side]).ravel() for k in range(3)];thresholds[side]={str(q):float(np.quantile(np.concatenate(vals),q)) for q in [.99,.999]};stability[side]={str(q):[float(np.quantile(v,q)) for v in vals] for q in [.99,.999]}
 out=dict(thresholds=thresholds,per_shuffle=stability,pooling='all active-feature coefficients across256rotated axes and3shuffles, separately by modality',reason='rotated axis indices are not identified between shuffled fits and true fits',quantiles=[.99,.999],inference='empirical coefficient cutoff, not calibrated familywise error or FDR guarantee');save(dst,out);return out

class Eval:
 def __init__(self):
  cfg=ev.p.old.load_config(ROOT/ev.p.CONFIG['cc3m-cc3m']);self.data,self.parents=ev.p.eval_population(cfg,'test',OUT);d=self.data;eligible=np.isin([c['id'] for c in d['concepts']],d['metadata']['eligible_category_ids']);assert eligible.sum()==50;self.labels=[d['labels'][s][:,eligible].astype(float) for s in S];self.cats=[c for c,k in zip(d['concepts'],eligible) if k];save(OUT/'categories.json',self.cats);ss=np.load(ev.REF/'evaluation-splits.npz');self.splits=[(ss[f'image_{j}'],ss[f'text_{j}']) for j in range(20)];shutil.copy2(ev.REF/'evaluation-splits.npz',OUT/'evaluation-splits.npz');self.weights=[dict(np.load(AX/'fits'/f'CCA-1-pullback-s{seed}.npz')) for seed in range(5)]
 def evaluate(self,name,allws,oldrule=None):
  for seed,ws in enumerate(allws):
   dst=OUT/'eval'/f'{name}-s{seed}.json'
   if dst.exists():continue
   t=time.time();meta=dict(new_true_rotation=False,source=str(AX/'fits'/f'CCA-1-pullback-s{seed}.npz'));cs={s:describe(w) for s,w in zip(S,ws)}
   if oldrule:
    path=ROOT/'runs/cca1-pruning-2026-10-12/eval'/f'CCA1-{oldrule}-s{seed}.json';old=json.loads(path.read_text());rows=old['rows'];ret=old['retrieval'];meta.update(reused=True,source_evaluation=str(path))
    for s in S:assert abs(cs[s]['mean']-old['coefficients'][s]['mean'])<1e-9
   else:
    meta['reused']=False;d=self.data;sc=[np.asarray(d['raw'][s]@(w/d['scales'][s][:,None]))-d['means'][s]@(w/d['scales'][s][:,None]) for s,w in zip(S,ws)];rows=[];full=[[],[]];variables=[[],[]]
    for split,pair in enumerate(self.splits):
     aa=[];vary=[]
     for j,ix in enumerate(pair):
      x=sc[j][ix];lab=self.labels[j][ix];pos=lab.sum(0);n=len(x);rank=stats.rankdata(x,axis=0,method='average');au=(rank.T@lab-pos*(pos+1)/2)/(pos*(n-pos));assert np.isfinite(au).all();vv=np.ptp(x,axis=0)>0;vary.append(vv);variables[j].append(vv);aa.append(au);full[j].append(au.T.copy())
     valid=vary[0]&vary[1];responsive=[float((q>=.8).sum(0).mean()) for q in aa]
     for q in aa:q[~valid]=-np.inf
     if valid.any():
      ii=[q.argmax(0) for q in aa];best=[q[i,np.arange(50)] for q,i in zip(aa,ii)];dup=[np.bincount(i,minlength=256)[i]>1 for i in ii];match=ii[0]==ii[1];solo=match&~dup[0]&~dup[1]
     else:ii=[np.full(50,-1)]*2;best=[np.full(50,np.nan)]*2;dup=[np.zeros(50,bool)]*2;match=solo=np.zeros(50,bool)
     rows.append(dict(split=split,match=int(match.sum()),solo=int(solo.sum()),overlap_image=int(dup[0].sum()),overlap_text=int(dup[1].sum()),responsive_image=responsive[0],responsive_text=responsive[1],image_axes=ii[0].tolist(),text_axes=ii[1].tolist(),image_auc=best[0].tolist(),text_auc=best[1].tolist(),eligible_axes=int(valid.sum())))
    ret=ev.retrieval(sc,self.parents);np.savez_compressed(OUT/'auroc'/f'{name}-s{seed}.npz',image=np.array(full[0]),text=np.array(full[1]),image_variable=np.array(variables[0]),text_variable=np.array(variables[1]),category_ids=[c['id'] for c in self.cats],axis_ids=np.arange(256),split_ids=np.arange(20))
   np.savez_compressed(OUT/'weights'/f'{name}-s{seed}.npz',image=ws[0],text=ws[1]);save(dst,dict(condition=name,seed=seed,rows=rows,retrieval=ret,coefficients=cs,meta=meta,seconds=time.time()-t));log(stage='evaluated',condition=name,seed=seed,seconds=time.time()-t)
 def candidate(self,name,fn):
  ws=[[fn(z[s]) for s in S] for z in self.weights];self.evaluate(name,ws);counts=[float(np.mean([describe(w[j])['mean'] for w in ws])) for j in range(2)];k=[int(np.floor(x+.5)) for x in counts];allfixed=[[top(z[s],kk) for s,kk in zip(S,k)] for z in self.weights];save(OUT/(name+'-matched-budget.json'),dict(candidate=name,average_counts=dict(zip(S,counts)),fixed_counts=dict(zip(S,k)),rounding='nearest integer per modality using all5seeds; same count for every axis and seed'));self.evaluate(name+'-fixed',allfixed)

def main():
 save(OUT/'protocol.json',dict(rank=256,true_rotation='frozen CCA lambda1 pullback covariance rotations seeds0..4',null_rotations='3 independent permutations of all2324763 CC3M FIT pairs, identity-start covariance rotation for each',threshold='global within each modality, pooled across3shuffles, active features only',participation_ratio='round-half-up ((sum a^2)^2/sum a^4), separately per axis and modality',matching='same20splits50categories, exclusive among all50 on both modalities, constants jointly excluded',fixed_comparator='rounded mean retained count over5seeds256axes, separately image/text; no evaluation-based choice',F1=False))
 e=Eval()
 for rule in ['k32','k64','k128','mass90']:e.evaluate(rule,[[prior.cut(z[s],rule) for s in S] for z in e.weights],rule)
 e.candidate('participation',lambda w:pr_cut(w)[0])
 th=null_thresholds()
 for q,name in [(.99,'null99'),(.999,'null999')]:
  ws=[[z[s]*(abs(z[s])>th['thresholds'][s][str(q)]) for s in S] for z in e.weights];e.evaluate(name,ws);counts=[float(np.mean([describe(w[j])['mean'] for w in ws])) for j in range(2)];k=[int(np.floor(x+.5)) for x in counts];save(OUT/(name+'-matched-budget.json'),dict(candidate=name,average_counts=dict(zip(S,counts)),fixed_counts=dict(zip(S,k)),rounding='nearest integer per modality over5seeds256axes'));e.evaluate(name+'-fixed',[[top(z[s],kk) for s,kk in zip(S,k)] for z in e.weights])
 save(OUT/'complete.json',dict(time=time.time(),conditions=10,evaluations=50,new_evaluations=30,reused=20))
if __name__=='__main__':
 try:main()
 except Exception as ex:save(OUT/'error.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
