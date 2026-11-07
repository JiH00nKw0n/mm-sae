import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import numpy as np
from scipy import sparse,linalg
import json,csv,time,traceback
ROOT=Path('/mnt/working/mm-sae');OLD=ROOT/'runs/category-activation-patterns-2026-10-12';OUT=ROOT/'runs/category-power-patterns-2026-10-12';OUT.mkdir(exist_ok=True)
SRC=ROOT/'runs/cc3m-followup-2026-10-05/cc3m-sae-coco-activations';IDX=SRC/'index/train2017'
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def log(**x):print(x,flush=True);save('progress.json',x)
def csvout(n,rs):
 with (OUT/n).open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-30)
def cut(x,k):
 if not k:return x.copy()
 ids=np.argsort(-abs(x),axis=1,kind='stable')[:,:k];v=np.zeros_like(x);np.put_along_axis(v,ids,np.take_along_axis(x,ids,1),1);return v

def main():
 start=time.time();old=np.load(OLD/'patterns.npz');keep=np.flatnonzero(old['eligible']);cats=json.loads((OLD/'categories.json').read_text());names=np.array([cats[i]['name'] for i in keep]);groups=np.array([cats[i]['group'] for i in keep]);ph=np.load(OLD/'phi.npy')[np.ix_(keep,keep)];spl=np.load(OLD/'split.npz')
 with np.load(ROOT/'runs/mapping-suite-cc3m-2026-10-04/moments.npz') as z:
  ids=[z['image_ids'].copy(),z['text_ids'].copy()];mu=z['fit_mean'];second=z['fit_second'];n=len(ids[0]);sd=np.sqrt(second.diagonal()-mu**2);cov=(second-np.outer(mu,mu))/sd[:,None]/sd[None,:];fit_n=int(z['fit_n'])
 C=cov[:n,n:].copy();sx=cov[:n,:n].copy();sy=cov[n:,n:].copy();del second,cov
 patterns={'mean':(old['p'][keep][:,ids[0]],old['q'][keep][:,ids[1]])};reg=[];diagnostics=[]
 for side,rr,labpath,sl,fi in [('image',spl['image_half_a_rows'],ROOT/'runs/elice-rq1-lexicon2/index/train2017/full_presence.npy',slice(0,n),ids[0]),('text',spl['text_half_b_rows'],IDX/'mentions.npy',slice(n,None),ids[1])]:
  log(stage='regression',side=side);lab=np.load(labpath)[rr][:,keep].astype(float);raw=sparse.load_npz(SRC/'activations/train2017'/f'{side}.npz').tocsr()[rr][:,fi].astype(float);m=lab.mean(0);mx=np.asarray(raw.mean(0)).ravel();G=lab.T@lab/len(rr)-np.outer(m,m);H=(sparse.csr_matrix(lab).T@raw).toarray()/len(rr)-np.outer(m,mx);H/=sd[sl][None,:];ev=np.linalg.eigvalsh(G);ridge=0. if ev[0]>ev[-1]*1e-10 else float(np.diag(G).mean()*1e-6);beta=linalg.solve(G+ridge*np.eye(len(keep)),H,assume_a='pos');reg.append(beta);intercept=(mx-mu[sl])/sd[sl]-m@beta;np.savez_compressed(OUT/f'regression-{side}.npz',coefficients=beta,feature_ids=fi,category_names=names,intercept=intercept);diagnostics.append(dict(side=side,samples=len(rr),ridge=ridge,min_eigenvalue=float(ev[0]),max_eigenvalue=float(ev[-1]),condition=float(ev[-1]/ev[0])));del raw,lab
 patterns['regression']=tuple(reg);log(stage='eigendecomposition');ex,Qx=linalg.eigh(sx,check_finite=False);ey,Qy=linalg.eigh(sy,check_finite=False);save('regression-diagnostics.json',diagnostics)
 def inv(x,Q,e,lam,power=1):return ((x@Q)/(e+lam)**power)@Q.T
 statnames=['C']+[f'CCA-{v}' for v in [.01,.1,1.]]+[f'W-{v}' for v in [.01,.1,1.,10.]]+[f'W2-{v}' for v in [.01,.1,1.,10.]];scores={};reverse={}
 for pattern,(p,q) in patterns.items():
  for k in [0,8,16,64]:
   pp,qq=cut(p,k),cut(q,k)
   for stat in statnames:
    if stat=='C':a=pp@C;b=qq@C.T
    else:
     lam=float(stat.split('-')[1]);power=.5 if stat.startswith('CCA') else 1.;a=inv(pp,Qx,ex,lam,power)@C
     if stat.startswith(('W2','CCA')):a=inv(a,Qy,ey,lam,power);b=inv(inv(qq,Qy,ey,lam,power)@C.T,Qx,ex,lam,power)
     else:b=inv(qq@C.T,Qx,ex,lam)
    s=norm(a)@norm(qq).T;t=norm(b)@norm(pp).T;scores[pattern,k,stat]=s;reverse[pattern,k,stat]=t;tag=f'{pattern}-{k or "full"}-{stat}';np.savez_compressed(OUT/f'scores-{tag}.npz',s=s,reverse_projection=t,category_names=names)
    with (OUT/f'scores-{tag}.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.writer(f);w.writerow(['image category / text category']+names.tolist());w.writerows([[name]+row.tolist() for name,row in zip(names,s)])
   log(stage='scores',pattern=pattern,features=k or 'full')
 # Fixed category resamples are shared across all methods for paired intervals.
 rows=[];per=[];rng=np.random.default_rng(120169);boot={}
 for group in ['all','object','stuff']:
  gm=np.ones(len(names),bool) if group=='all' else groups==group
  for th in [.5,.3,.2]:
   mask=(ph>=th)&~np.eye(len(names),dtype=bool);ix=np.flatnonzero(gm&mask.any(1));boot[group,th]=(ix,rng.integers(0,len(ix),(10000,len(ix))))
 for pattern in patterns:
  for k in [0,8,16,64]:
   for stat in statnames:
    s=scores[pattern,k,stat];base=scores[pattern,k,'C'];original=scores['mean',k,'C']
    for group in ['all','object','stuff']:
     gm=np.ones(len(names),bool) if group=='all' else groups==group;r=dict(pattern=pattern,features=k or 'full',statistic=stat,group=group,categories=int(gm.sum()),image_top1=int(((s.argmax(1)==np.arange(len(names)))&gm).sum()),text_top1=int(((s.argmax(0)==np.arange(len(names)))&gm).sum()),text_reverse_projection_top1=int(((reverse[pattern,k,stat].argmax(1)==np.arange(len(names)))&gm).sum()))
     for th in [.5,.3,.2]:
      mask=(ph>=th)&~np.eye(len(names),dtype=bool);other=(ph<th)&~np.eye(len(names),dtype=bool);ix,bs=boot[group,th]
      def gap(z):return np.array([z[i,i]-z[i,mask[i]].mean() for i in ix])
      g=gap(s);d=g-gap(base);d0=g-gap(original);ci=np.quantile(g[bs].mean(1),[.025,.975]);dc=np.quantile(d[bs].mean(1),[.025,.975]);d0c=np.quantile(d0[bs].mean(1),[.025,.975]);prefix=f'phi{th}_';vals=dict(n=len(ix),same=float(np.diag(s)[ix].mean()),cooccurring=float(np.mean([s[i,mask[i]].mean() for i in ix])),other=float(np.mean([s[i,other[i]].mean() for i in ix])),gap=float(g.mean()),gap_low=float(ci[0]),gap_high=float(ci[1]),same_higher=int((g>0).sum()),delta_vs_C=float(d.mean()),delta_vs_C_low=float(dc[0]),delta_vs_C_high=float(dc[1]),delta_vs_original=float(d0.mean()),delta_vs_original_low=float(d0c[0]),delta_vs_original_high=float(d0c[1]));r.update({prefix+a:b for a,b in vals.items()})
     rows.append(r)
    for th in [.5,.3,.2]:
     mask=(ph>=th)&~np.eye(len(names),dtype=bool);other=(ph<th)&~np.eye(len(names),dtype=bool)
     for i,name in enumerate(names):per.append(dict(pattern=pattern,features=k or 'full',statistic=stat,threshold=th,category=name,group=groups[i],same=float(s[i,i]),cooccurring=float(s[i,mask[i]].mean()) if mask[i].any() else None,other=float(s[i,other[i]].mean()),partners=int(mask[i].sum())))
 csvout('summary.csv',rows);csvout('category-scores.csv',per)
 errors={str(k or 'full'):float(np.max(abs(scores['mean',k,'C']-np.load(OLD/f'scores-{k or "full"}.npz')['s']))) for k in [0,8,16,64]};assert max(errors.values())<1e-10
 save('protocol.json',dict(cpu_only=True,threads=2,CC3M_fit_pairs=fit_n,statistics='CC3M FIT moments only; constant dimensions fixed zero',lambdas=[.01,.1,1.,10.],mean_active_image_variance=float(np.diag(sx).mean()),mean_active_text_variance=float(np.diag(sy).mean()),regression_categories=169,regression=diagnostics,split='same disjoint COCO train image halves as preceding analysis',bootstrap='10000 paired category resamples; no image or training resampling; pointwise intervals, no multiple-testing correction',reverse='primary text top1 is column argmax of the same s matrix; actual reverse projection is additional',baseline_max_error=errors,min_eigenvalues=[float(ex[0]),float(ey[0])],matrix_computation='exact inverse application via eigendecomposition, without constructing dense W'))
 save('complete.json',dict(seconds=time.time()-start,score_matrices=len(scores),summary_rows=len(rows)));log(stage='complete')
if __name__=='__main__':
 if (OUT/'complete.json').exists():raise SystemExit(0)
 try:main()
 except Exception as e:save('error.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
