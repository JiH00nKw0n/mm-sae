"""Post-hoc diagnostics, no test-label training. Original CCA rotations stay frozen."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json,time
import numpy as np
from scipy.stats import rankdata
import experiments.apbt_search_20261009 as prior
from mm_sae.analysis.semantic_evaluation import auc_matrix
ROOT=prior.ROOT; SRC=ROOT/'runs/cca-unsupervised-rotations-2026-10-09';OUT=ROOT/'runs/rotation-mechanisms-2026-10-09'
OUT.mkdir(exist_ok=True); start=time.time()
def save(p,x):
 def clean(v):
  if isinstance(v,dict):return {k:clean(a) for k,a in v.items()}
  if isinstance(v,(list,tuple)):return [clean(a) for a in v]
  if isinstance(v,np.ndarray):return clean(v.tolist())
  if isinstance(v,(np.integer,np.floating)):return clean(v.item())
  if isinstance(v,float) and not np.isfinite(v):return None
  return v
 p.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2))
def concentration(z,axis):
 q=np.sort(z*z,axis=axis)
 q=np.flip(q,axis=axis);tot=q.sum(axis=axis,keepdims=True);cum=q.cumsum(axis=axis)
 out={f'n{int(t*100)}':float(np.median((cum<t*tot).sum(axis=axis)+1)) for t in [.9,.95]}
 eff=np.abs(z).sum(axis=axis)**2/np.maximum((z*z).sum(axis=axis),1e-25)
 out.update(effective_mean=float(eff.mean()),effective_median=float(np.median(eff)))
 return out
def scorestats(z):
 out=concentration(z,1);v=z-z.mean(0);v/=np.maximum(np.sqrt((v*v).mean(0)),1e-12)
 q=np.sort(v*v,axis=0);n=max(1,int(len(z)*.01))
 out.update(tail1pct_energy=float(np.mean(q[-n:].sum(0)/np.maximum(q.sum(0),1e-20))),tail1pct_fourth=float(np.mean((q[-n:]**2).sum(0)/np.maximum((q*q).sum(0),1e-20))),kurtosis_median=float(np.median((v**4).mean(0)-3)))
 return out
def binary_auc(s,y):
 p=y.sum();n=len(y)-p
 if p<10 or n<10:return np.nan
 return float((rankdata(s)[y].sum()-p*(p+1)/2)/(p*n))
def paired_corr(x,y):
 x=x-x.mean(0);y=y-y.mean(0)
 return (x*y).sum(0)/np.maximum(np.linalg.norm(x,axis=0)*np.linalg.norm(y,axis=0),1e-15)
def main():
 for cond in ['cc3m-cc3m','cc3m-coco','coco-coco']:
  out=OUT/cond;out.mkdir(exist_ok=True)
  cfg=prior.old.load_config(ROOT/prior.CONFIG[cond]);data,parents=prior.eval_population(cfg,'test',out)
  ix=Path(cfg['source_run'])/'index/val2017';labs={'image':np.load(ix/'presence.npy').astype(bool),'text':np.load(ix/'mentions.npy').astype(bool)}
  names=json.loads((ROOT/'runs/cca-171-category-names-2026-10-09/protocol.json').read_text())['prototype_texts'];assert len(names)==171
  with np.load(next(m['path'] for m in cfg['models'] if m['key']=='cca_256')) as z:w={s:z[s].astype(float) for s in ['image','text']}
  base={s:np.asarray(data['raw'][s]@(w[s]/data['scales'][s][:,None]))-data['means'][s]@(w[s]/data['scales'][s][:,None]) for s in w}
  with np.load(Path(cfg['parent_run'])/'moments.npz') as z:
   ids={s:z[s+'_ids'] for s in w};ni=len(ids['image']);mu=z['fit_mean'];cv=z['fit_second']-np.outer(mu,mu);sd=np.sqrt(np.maximum(cv.diagonal(),0));sd[sd==0]=1;cv/=sd[:,None]*sd[None,:]
  cov={'image':cv[:ni,:ni],'text':cv[ni:,ni:]}
  # Confounders selected from image half A only; both B strata must have >=10 positive and negative A examples on selection half.
  ma=data['masks']['image_a'];mb=data['masks']['image_b'];la=labs['image'][ma];cen=la-la.mean(0);phi=cen.T@cen/np.maximum(np.linalg.norm(cen,axis=0)[:,None]*np.linalg.norm(cen,axis=0)[None,:],1e-15);np.fill_diagonal(phi,-np.inf)
  conf=[]
  for c in range(171):
   valid=[]
   for d in np.argsort(-phi[c]):
    if d==c or phi[c,d]<=0:continue
    if all(min(la[la[:,d]==b,c].sum(),(~la[la[:,d]==b,c]).sum())>=10 for b in [False,True]):valid.append(int(d));break
   conf.append(valid[0] if valid else None)
  save(out/'confounders.json',[dict(concept=names[c],other=names[d] if d is not None else None,phi=phi[c,d] if d is not None else None) for c,d in enumerate(conf)])
  oracle={};ow={}
  if cond!='cc3m-cc3m':
   op=ROOT/'runs/oracle-sets-2026-10-04' if cond=='coco-coco' else ROOT/'runs/cc3m-followup-2026-10-05/coco-fit/annotation-sets'
   for s,key in [('image','image'),('text','propagated_presence')]:
    z=np.load(op/'fits'/f'{key}_16.npz');assert np.array_equal(ids[s],z['feature_ids'])
    ow[s]=z['coefficients']/z['feature_scale'][:,None]*data['scales'][s][:,None]
    oracle[s]=np.asarray(data['raw'][s]@(ow[s]/data['scales'][s][:,None]))-data['means'][s]@(ow[s]/data['scales'][s][:,None])
  jobs=[(p.stem.replace('-result',''),p) for p in (SRC/cond).glob('*-result.json')]+[(p.stem.replace('-result',''),p) for p in (ROOT/'runs/rotation-logcosh-scale-2026-10-09/evaluated'/cond).glob('*-result.json')]+([('oracle_ridge',None)] if oracle else [])
  for name,p in jobs:
   dest=out/(name+'.json')
   if dest.exists():continue
   print(cond,name,round(time.time()-start),flush=True)
   if p:
    r=np.load(p.parent/(name+'-evaluated-rotation.npz'))['rotation'];ww={s:w[s]@r for s in w};scores={s:base[s]@r for s in w};orig=json.loads(p.read_text());matching=orig['matching50']['summary']['agree_at1_count']
   else:ww=ow;scores=oracle;matching=None
   res=dict(condition=cond,name=name,matching50=matching,sides={},categories=[])
   selected={};testauc={};aucs={}
   for s in w:
    selecthalf='a' if s=='image' else 'b';testhalf='b' if s=='image' else 'a';sel=data['masks'][s+'_'+selecthalf];test=data['masks'][s+'_'+testhalf]
    aucs[s]=auc_matrix(scores[s][sel],labs[s][sel],chunk_size=64);held=auc_matrix(scores[s][test],labs[s][test],chunk_size=64)
    selected[s]=np.nan_to_num(aucs[s],nan=-np.inf).argmax(0) if p else np.arange(171);testauc[s]=held[selected[s],np.arange(171)]
    valid=(labs[s][sel].sum(0)>=10)&((~labs[s][sel]).sum(0)>=10)&(labs[s][test].sum(0)>=10)&((~labs[s][test]).sum(0)>=10)
    tops=np.argsort(np.abs(ww[s]),axis=0)[-16:];indicator=np.zeros_like(ww[s],dtype=np.int16);np.put_along_axis(indicator,tops,1,axis=0);over=indicator.T@indicator
    intra=[]
    for k in range(ww[s].shape[1]):
     co=cov[s][np.ix_(tops[:,k],tops[:,k])];intra.append((co.sum()-np.trace(co))/240)
    res['sides'][s]=dict(coeff=concentration(ww[s],0),output=scorestats(scores[s]),heldout_mean_auc=float(np.nanmean(testauc[s][valid])),eligible_concepts=int(valid.sum()),distinct_selected=int(len(set(selected[s][valid]))),median_top16_internal_correlation=float(np.median(intra)),mean_top16_overlap=float((over.sum()-np.trace(over))/(len(over)*(len(over)-1))),negative_energy_fraction=float((ww[s][ww[s]<0]**2).sum()/(ww[s]**2).sum()))
    if oracle and p:
     matchw=ww[s][:,selected[s]];support=ow[s]!=0;top=np.argsort(abs(matchw),axis=0)[-16:];ind=np.zeros_like(support);np.put_along_axis(ind,top,True,axis=0)
     overlap=(ind&support).sum(0);mass=((matchw**2)*support).sum(0)/np.maximum((matchw**2).sum(0),1e-20)
     cor=paired_corr(scores[s][test][:,selected[s]],oracle[s][test]);res['sides'][s]['oracle']=dict(top16_overlap_mean=float(overlap[valid].mean()),oracle_support_energy=float(mass[valid].mean()),heldout_output_corr=float(cor[valid].mean()),per_concept_overlap=overlap,per_concept_mass=mass,per_concept_corr=cor)
   labels= labs['image'][mb]
   for c in range(171):
    k=selected['image'][c];s=scores['image'][mb,k];d=conf[c];cond_auc=[]
    if d is not None:
     for b in [False,True]:
      keep=labels[:,d]==b;cond_auc.append(binary_auc(s[keep],labels[keep,c]))
    res['categories'].append(dict(name=names[c],image_coordinate=int(k),text_coordinate=int(selected['text'][c]),same_coordinate=bool(k==selected['text'][c]),image_auc=testauc['image'][c],text_auc=testauc['text'][c],confounder=names[d] if d is not None else None,phi=phi[c,d] if d is not None else None,conditional_absent=cond_auc[0] if cond_auc else None,conditional_present=cond_auc[1] if cond_auc else None,image_frequency=float(labs['image'][:,c].mean()),text_frequency=float(labs['text'][:,c].mean())))
   save(dest,res)
  del data,base,cov,cv
 save(OUT/'complete.json',dict(completed=True,seconds=time.time()-start))
if __name__=='__main__':main()
