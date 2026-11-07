"""CC3M moment-conditioned SVD axes, CPU only, fixed COCO evaluation, no F1."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import json,time,traceback,csv
import numpy as np
from scipy import linalg,stats,sparse
import experiments.apbt_search_20261009 as p
import torch
torch.set_num_threads(2)
ROOT=Path('/mnt/working/mm-sae');OUT=ROOT/'runs/conditioned-svd-axes-2026-10-12';REF=ROOT/'runs/rotation-audit-2026-10-11';SIDES=['image','text']
for d in ['fits','eval','bases','auroc','checkpoints']:(OUT/d).mkdir(parents=True,exist_ok=True)
def save(path,x):p.save(path,x)
def log(**x):save(OUT/'cca-wide-progress.json',dict(time=time.time(),**x));print(x,flush=True)
def polar(x):u,_,v=linalg.svd(x,full_matrices=False,check_finite=False);return u@v

def fit(base,covs,seed,name):
 dst=OUT/'fits'/f'{name}-s{seed}.npz'
 if dst.exists():return np.load(dst)['rotation'],json.loads(dst.with_suffix('.json').read_text())
 loads=[]
 for w,c in zip(base,covs):
  l=c@w;l/=np.maximum(np.linalg.norm(l,axis=1,keepdims=True),1e-12);loads.append(l)
 torch.manual_seed(1200+seed);r=np.eye(256) if seed==0 else polar(torch.randn(256,256,dtype=torch.float64).numpy());hist=[];start=time.time();alpha=.25;scale=None;stale=0;prev=None
 def vg(r):
  zs=[l@r for l in loads];z2=[z*z for z in zs];val=sum(np.sum((zz*zz).mean(0)-zz.mean(0)**2) for zz in z2)/2;g=sum(2/len(l)*l.T@(z*zz-z*zz.mean(0)) for l,z,zz in zip(loads,zs,z2));om=(r.T@g-g.T@r)/2;return val,g,om
 for it in range(20001):
  val,g,om=vg(r);gn=np.linalg.norm(om);scale=max(abs(val),1e-12) if scale is None else scale;rel=gn/max(abs(val),scale,1e-12)
  if it%25==0:hist.append(dict(step=it,objective=val,gradient=gn,relative_gradient=rel,seconds=time.time()-start));save(dst.with_suffix('.history.json'),hist)
  if it%100==0:log(stage='rotation',name=name,seed=seed,step=it,relative_gradient=rel,seconds=time.time()-start);np.savez(OUT/'checkpoints'/f'{name}-s{seed}.npz',rotation=r)
  stop='iteration_limit'
  if rel<1e-6:stop='gradient';break
  if prev is not None:stale=stale+1 if abs(val-prev)/max(abs(val),scale,1e-12)<1e-10 else 0
  if stale>=50 and rel<1e-4:stop='objective_stable_and_small_gradient';break
  if it==20000:break
  prev=val;trial=polar(g);nv=vg(trial)[0];accepted=nv>=val-1e-14 and np.linalg.norm(trial-r)>1e-12
  if not accepted:
   direction=om/max(gn,1e-30);a=min(alpha*1.5,4.);eye=np.eye(256)
   for attempt in range(30):
    trial=r@linalg.solve(eye-.5*a*direction,eye+.5*a*direction,check_finite=False);nv=vg(trial)[0]
    if nv>=val+1e-4*a*gn:accepted=True;break
    a*=.5
   alpha=a
  if not accepted:stop='line_search_stalled';break
  r=trial
 r=polar(r);weights=[w@r for w in base];sg=np.sign(sum(((w>0)*w*w).sum(0)-((w<0)*w*w).sum(0) for w in weights));sg[sg==0]=1;r*=sg
 meta=dict(steps=it,stop=stop,relative_gradient=float(rel),seconds=time.time()-start,orthogonality_error=float(abs(r.T@r-np.eye(256)).max()),sign='shared sign maximizing combined positive coefficient energy',objective=float(val))
 np.savez_compressed(dst,rotation=r,image=base[0]@r,text=base[1]@r);save(dst.with_suffix('.json'),meta);return r,meta

def retrieval(scores,parents):
 # Batched cosine matrix with identical lower-index tie rule; no GPU allocations.
 x,y=[(v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-30)).astype(np.float32) for v in scores];ni,nt=len(x),len(y);ir=np.empty(ni,int);tr=np.zeros(nt,int);pos=np.einsum('ij,ij->i',x[parents],y)
 # Compute positives with the same GEMM, avoiding dot-product rounding mismatch.
 for st in range(0,ni,256):
  sim=x[st:st+256]@y.T;ii=np.flatnonzero((parents>=st)&(parents<st+len(sim)));pos[ii]=sim[parents[ii]-st,ii]
 for st in range(0,ni,256):
  sim=x[st:st+256]@y.T
  tr+=(sim>pos[None,:]).sum(0)+((sim==pos[None,:])&(np.arange(st,st+len(sim))[:,None]<parents[None,:])).sum(0)
  for j,row in enumerate(sim):
   gt=np.flatnonzero(parents==st+j);b=row[gt].max();bi=gt[row[gt]==b].min();ir[st+j]=1+(row>b).sum()+((row==b)&(np.arange(nt)<bi)).sum()
 tr+=1
 return {d:dict(recall={str(k):float((rr<=k).mean()) for k in [1,5,10]}) for d,rr in [('image_to_text',ir),('text_to_image',tr)]}

def coeffstats(w):
 v=np.sort(w*w,axis=0)[::-1];nn=(np.cumsum(v,axis=0)<.9*v.sum(0)).sum(0)+1;return dict(n90_mean=float(nn.mean()),n90_median=float(np.median(nn)),n90_min=int(nn.min()),n90_max=int(nn.max()),mean=float((w!=0).sum(0).mean()))
def measure(name,seed,base,r,meta,data,parents,labels,splits,preret):
 dense=[w@r for w in base]
 for cut in ['dense','column64']:
  dst=OUT/'eval'/f'{name}-{cut}-s{seed}.json'
  if dst.exists():continue
  t=time.time();ws=[]
  for w in dense:
   if cut=='column64':ix=np.argsort(-abs(w),axis=0,kind='stable')[:64];v=np.zeros_like(w);np.put_along_axis(v,ix,np.take_along_axis(w,ix,0),0);w=v
   ws.append(w)
  sc=[np.asarray(data['raw'][s]@(w/data['scales'][s][:,None]))-data['means'][s]@(w/data['scales'][s][:,None]) for s,w in zip(SIDES,ws)];rows=[];fullauc=[[],[]]
  for split,pair in enumerate(splits):
   aa=[]
   for j,ix in enumerate(pair):
    z=sc[j][ix];lab=labels[j][ix];n=len(z);pos=lab.sum(0);ranks=stats.rankdata(z,axis=0,method='average');au=(ranks.T@lab-pos*(pos+1)/2)/(pos*(n-pos));au[np.ptp(z,axis=0)==0]=-np.inf;assert np.isfinite(au).all();aa.append(au);fullauc[j].append(au.T)
   idx=[q.argmax(0) for q in aa];best=[q[ii,np.arange(50)] for q,ii in zip(aa,idx)];dup=[np.bincount(ii,minlength=256)[ii]>1 for ii in idx];match=idx[0]==idx[1];solo=match&~dup[0]&~dup[1]
   rows.append(dict(split=split,match=int(match.sum()),solo=int(solo.sum()),overlap_image=int(dup[0].sum()),overlap_text=int(dup[1].sum()),responsive_image=float((aa[0]>=.8).sum(0).mean()),responsive_text=float((aa[1]>=.8).sum(0).mean()),image_axes=idx[0].tolist(),text_axes=idx[1].tolist(),image_auc=best[0].tolist(),text_auc=best[1].tolist()))
  ret=preret if cut=='dense' else retrieval(sc,parents);np.savez_compressed(OUT/'auroc'/f'{name}-{cut}-s{seed}.npz',image=np.array(fullauc[0]),text=np.array(fullauc[1]),category_ids=np.array([c['id'] for c in data['selected_concepts']]),axis_ids=np.arange(256),split_ids=np.arange(20));save(dst,dict(name=name,seed=seed,cut=cut,rows=rows,retrieval=ret,coefficients={s:coeffstats(w) for s,w in zip(SIDES,ws)},meta=meta,seconds=time.time()-t));log(stage='evaluated',name=name,seed=seed,cut=cut,seconds=time.time()-t)

def main():
 t=time.time();save(OUT/'cca-wide-protocol.json',dict(condition='CC3M SAE and CC3M mapping moments only',rank=256,statistics=['CCA-2','CCA-5','CCA-10','CCA-20','CCA-50'],coefficient_methods={'pullback':'M=Lx C Ly=U D Vt; A=Lx U, B=Ly V; Lx/Ly inverse square root, regularized CCA','direct':'A=U, B=V'},rotation='equal modality weighted row-normalized covariance loadings Varimax; shared orthogonal R; original SVD/polar with safeguarded Cayley updates',seeds=list(range(5)),splits=20,AUROC_shape='20 splits x50 categories x256 axes',F1=False,confidence='paired crossed initialization and split bootstrap',constant_features='omitted active coordinates, full4096 padded weights stored'))
 cfg=p.old.load_config(ROOT/p.CONFIG['cc3m-cc3m']);data,parents=p.eval_population(cfg,'test',OUT);eligible=np.isin([c['id'] for c in data['concepts']],data['metadata']['eligible_category_ids']);assert eligible.sum()==50;labels=[data['labels'][s][:,eligible].astype(float) for s in SIDES];data['selected_concepts']=[c for c,ok in zip(data['concepts'],eligible) if ok];save(OUT/'categories.json',data['selected_concepts']);ss=np.load(REF/'evaluation-splits.npz');splits=[(ss[f'image_{j}'],ss[f'text_{j}']) for j in range(20)];np.savez_compressed(OUT/'evaluation-splits.npz',**dict(ss))
 z=np.load(ROOT/'runs/mapping-suite-cc3m-2026-10-04/moments.npz');ids=[z['image_ids'],z['text_ids']];mu=z['fit_mean'];sec=z['fit_second'];sd=np.sqrt(sec.diagonal()-mu*mu);cv=(sec-np.outer(mu,mu))/sd[:,None]/sd[None,:];n=len(ids[0]);sx=cv[:n,:n].copy();sy=cv[n:,n:].copy();C=cv[:n,n:].copy();del sec,cv,z
 trainidx=np.load(REF/'split.npz');trainraw=[]
 for side,fi,key in zip(SIDES,ids,['fit_rows','fit_text_rows']):
  log(stage='orientation_samples',side=side);raw=sparse.load_npz(ROOT/'runs/cc3m-sae-2026-10-04/activations/train2017'/f'{side}.npz').tocsr();trainraw.append(raw[trainidx[key]][:,fi]);del raw
 log(stage='moments_eigh');ex,Qx=linalg.eigh(sx,check_finite=False);ey,Qy=linalg.eigh(sy,check_finite=False)
 def inv(Q,e,lam):return (Q/np.sqrt(e+lam))@Q.T
 for stat in ['CCA-2','CCA-5','CCA-10','CCA-20','CCA-50']:
  bp=OUT/'bases'/f'{stat}.npz';Lx=Ly=None
  if stat!='C':
   lam=float(stat.split('-')[1]);Lx=inv(Qx,ex,lam);Ly=inv(Qy,ey,lam) if stat.startswith('CCA') else None
  if bp.exists():b=np.load(bp);u,v,sv=b['U'],b['V'],b['singular_values']
  elif stat=='C':
   with np.load(ROOT/'runs/mapping-ablation-cc3m-2026-10-04/transforms/cross_svd_3221.npz') as b:u,v=b['image'][:,:256],b['text'][:,:256]
   sv=np.diag(u.T@C@v);np.savez_compressed(bp,U=u,V=v,singular_values=sv)
  else:
   log(stage='svd',statistic=stat);M=Lx@C;M=M@Ly if Ly is not None else M;uu,sv,vh=linalg.svd(M,full_matrices=False,check_finite=False);u=uu[:,:256];v=vh[:256].T;sv=sv[:256];res=float(np.linalg.norm(M@v-u*sv)/np.linalg.norm(u*sv));assert res<1e-8;np.savez_compressed(bp,U=u,V=v,singular_values=sv,residual=res);del M,uu,vh
  for mode in ['pullback','direct']:
   name=stat+'-'+mode;base=[Lx@u if Lx is not None and mode=='pullback' else u,Ly@v if Ly is not None and mode=='pullback' else v];wb={}
   for side,w,fi in zip(SIDES,base,ids):ww=np.zeros((4096,256));ww[fi]=w;wb[side]=ww
   np.savez_compressed(OUT/'bases'/f'{name}-coefficients.npz',**wb,image_active=base[0],text_active=base[1],image_ids=ids[0],text_ids=ids[1])
   rp=OUT/'bases'/f'{name}-retrieval.json'
   if rp.exists():preret=json.loads(rp.read_text())
   else:
    sc=[np.asarray(data['raw'][s]@(w/data['scales'][s][:,None]))-data['means'][s]@(w/data['scales'][s][:,None]) for s,w in zip(SIDES,base)];preret=retrieval(sc,parents);save(rp,preret);log(stage='prerotation_retrieval',name=name,metrics=preret)
   orient=[np.asarray(raw@(w/sd[sl,None]))-mu[sl]@(w/sd[sl,None]) for raw,w,sl in zip(trainraw,base,[slice(0,n),slice(n,None)])] if stat!='C' else None
   for seed in range(5):
    if stat=='C':
     r=np.load(REF/f'rank256/covariance-svd-{seed}-oriented.npz')['rotation'];meta=json.loads((REF/f'rank256/covariance-svd-{seed}-fit.json').read_text());meta['reused']=True;np.savez_compressed(OUT/'fits'/f'{name}-s{seed}.npz',rotation=r,image=base[0]@r,text=base[1]@r);save(OUT/'fits'/f'{name}-s{seed}.json',meta)
    else:
     r,meta=fit(base,[sx,sy],seed,name);skew=sum(((o@r)**3).mean(0) for o in orient);r*=np.where(skew<0,-1.,1.);meta['sign']='shared CC3M FIT third-moment sign using same32768 orientation sample as baseline';np.savez_compressed(OUT/'fits'/f'{name}-s{seed}.npz',rotation=r,image=base[0]@r,text=base[1]@r);save(OUT/'fits'/f'{name}-s{seed}.json',meta)
    measure(name,seed,base,r,meta,data,parents,labels,splits,preret)
 save(OUT/'cca-wide-complete.json',dict(seconds=time.time()-t,conditions=10,rotations=50,evaluations=100))
if __name__=='__main__':
 try:main()
 except Exception as e:save(OUT/'cca-wide-error.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
