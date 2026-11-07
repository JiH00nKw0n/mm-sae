"""CPU-only category-level raw SAE patterns; no rotation learning or single-feature selection."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import json,time,hashlib,csv,traceback
import numpy as np
from scipy import sparse
ROOT=Path('/mnt/working/mm-sae');OUT=ROOT/'runs/category-activation-patterns-2026-10-12';OUT.mkdir(exist_ok=True)
SRC=ROOT/'runs/cc3m-followup-2026-10-05/cc3m-sae-coco-activations';IDX=SRC/'index/train2017';ANN=ROOT/'runs/elice-rq1-lexicon2/index/train2017'
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def log(**v):save(OUT/'progress.json',dict(time=time.time(),**v));print(v,flush=True)
def csvout(name,rows):
 if rows:
  with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def matrixcsv(name,m,names):
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:w=csv.writer(f);w.writerow(['image category / text category']+names);w.writerows([[c]+r.tolist() for c,r in zip(names,m)])
def norm(x):
 n=np.linalg.norm(x,axis=1);assert np.all(n>0);return x/n[:,None]
def cut(x,k):
 if k==0:return x.copy()
 ix=np.argsort(-abs(x),axis=1,kind='stable')[:,:k];y=np.zeros_like(x);np.put_along_axis(y,ix,np.take_along_axis(x,ix,axis=1),axis=1);return y

def main():
 start=time.time();ds=json.loads((SRC/'dataset.json').read_text());cs=ds['concepts'];ids=np.array(json.loads((IDX/'concept_ids.json').read_text()));assert len(cs)==171 and np.array_equal(ids,[c['id'] for c in cs]);names=[c['name'] for c in cs];group=np.array(['object' if c['id']<91 else 'stuff' for c in cs]);assert (group=='object').sum()==80
 save(OUT/'lexicon.json',cs)
 yi=np.load(ANN/'full_presence.npy').astype(bool);yt=np.load(IDX/'mentions.npy').astype(bool);parents=np.load(IDX/'parents.npy');images=json.loads((IDX/'images.json').read_text());imids=np.array([v['image_id'] for v in images]);N=len(imids)
 # Audit existing own-caption labels against the stored171-category lexical matcher.
 from mm_sae.data.text import CaptionEditor,Concept
 ed=CaptionEditor([Concept(c['id'],c['name'],tuple(c['aliases'])) for c in cs]);caps=json.loads((IDX/'captions.json').read_text());cols={c['id']:j for j,c in enumerate(cs)};sample=np.random.default_rng(120171).choice(len(caps),1000,replace=False);mismatch=[]
 for i in sample:
  present,*_=ed.analyze(caps[i]['caption_id'],caps[i]['text']);v=np.zeros(171,bool);v[[cols[c] for c in present]]=True
  if not np.array_equal(v,yt[i]):mismatch.append(int(i))
 label_action='Reused cached own-caption mentions; all1000 audit samples match full-caption lexical labels.'
 if mismatch:
  log(stage='recompute_full_caption_mentions',sample_mismatches=len(mismatch));yt=np.zeros_like(yt)
  for i,c in enumerate(caps):
   present,*_=ed.analyze(c['caption_id'],c['text']);yt[i,[cols[v] for v in present]]=True
  label_action='Cached mentions differed on audit; recomputed ALL full-caption labels with existing171-category lexicon, longest overlapping phrase wins.'
 save(OUT/'label-audit.json',dict(samples=1000,mismatch_rows=mismatch,action=label_action));del caps
 rng=np.random.default_rng(20261012171);perm=rng.permutation(N);half=np.zeros(N,bool);half[perm[:N//2]]=True;ir=np.flatnonzero(half);tr=np.flatnonzero(~half[parents]);assert not set(imids[ir])&set(imids[parents[tr]])
 np.savez_compressed(OUT/'split.npz',image_half_a_rows=ir,text_half_b_rows=tr,image_half_a_ids=imids[ir],image_half_b_ids=imids[~half])
 mompath=ROOT/'runs/mapping-suite-cc3m-2026-10-04/moments.npz'
 with np.load(mompath) as z:
  ni=len(z['image_ids']);fid={'image':z['image_ids'].copy(),'text':z['text_ids'].copy()};mu=z['fit_mean'];second=z['fit_second'];var=second.diagonal()-mu**2;assert np.all(var>0);sd=np.sqrt(var);cross=second[:ni,ni:]-np.outer(mu[:ni],mu[ni:]);cross/=sd[:ni,None]*sd[None,ni:];fit_n=int(z['fit_n'])
 C=np.zeros((4096,4096));C[np.ix_(fid['image'],fid['text'])]=cross;np.savez_compressed(OUT/'cc3m-cross-covariance.npz',C=C)
 ps={};counts={}
 for side,rr,lab,sl in [('image',ir,yi,slice(0,ni)),('text',tr,yt,slice(ni,None))]:
  log(stage='pattern_means',side=side,samples=len(rr));raw=sparse.load_npz(SRC/'activations/train2017'/f'{side}.npz').tocsr();assert raw.shape[1]==4096 and raw.shape[0]==len(lab)
  x=raw[rr][:,fid[side]].astype(np.float64);ll=lab[rr];pos=ll.sum(0);total=np.asarray(x.sum(0)).ravel();crosslab=(sparse.csr_matrix(ll.astype(np.float64)).T@x).toarray();delta=crosslab/np.maximum(pos[:,None],1)-(total[None,:]-crosslab)/np.maximum((len(rr)-pos)[:,None],1);delta/=sd[sl][None,:]
  p=np.zeros((171,4096));p[:,fid[side]]=delta;ps[side]=p;counts[side]=pos;del raw,x,ll
 eligible=(counts['image']>=20)&(counts['text']>=20)&(counts['image']<len(ir))&(counts['text']<len(tr));kept=np.flatnonzero(eligible)
 category=[]
 for j,c in enumerate(cs):category.append(dict(id=c['id'],name=c['name'],group=group[j],image_positives=int(counts['image'][j]),text_positives=int(counts['text'][j]),included=bool(eligible[j])))
 csvout('category-eligibility.csv',category);save(OUT/'categories.json',category)
 np.savez_compressed(OUT/'patterns.npz',p=ps['image'],q=ps['text'],eligible=eligible,ids=ids,image_positive=counts['image'],text_positive=counts['text'])
 # Phi from full original COCO train images, not captions or CLIP crops.
 lab=sparse.csr_matrix(yi.astype(float));freq=np.asarray(lab.mean(0)).ravel();phi=(lab.T@lab).toarray()/N-np.outer(freq,freq);den=np.sqrt(freq*(1-freq));phi/=den[:,None]*den[None,:];assert np.isfinite(phi).all();np.save(OUT/'phi.npy',phi)
 ph=phi[np.ix_(kept,kept)];nm=[names[j] for j in kept];gg=group[kept];patterns={};per=[];summaries=[];rankings=[];pairs=[];pair_counts=[];fail=[]
 for th in [.5,.3,.2]:
  mask=(ph>=th)&~np.eye(len(kept),dtype=bool)
  for i in range(len(kept)):
   for j in range(i+1,len(kept)):
    if mask[i,j]:pairs.append(dict(threshold=th,first=nm[i],second=nm[j],phi=float(ph[i,j]),first_group=gg[i],second_group=gg[j]))
  pair_counts.append(dict(threshold=th,eligible_categories=len(kept),categories_with_partner=int(mask.any(1).sum()),undirected_pairs=int(mask.sum()//2),partners={nm[i]:[nm[j] for j in np.flatnonzero(mask[i])] for i in range(len(nm)) if mask[i].any()}))
 for k in [0,8,16,64]:
  pp=cut(ps['image'][kept],k);qq=cut(ps['text'][kept],k);s=norm(pp@C)@norm(qq).T;t=norm(qq@C.T)@norm(pp).T
  np.savez_compressed(OUT/f'scores-{k or "full"}.npz',s=s,reverse_projection=t,category_ids=ids[kept],category_names=np.array(nm));matrixcsv(f'scores-{k or "full"}.csv',s,nm)
  # reverse_projection CSV rows=text, columns=image; separate explicit labels file.
  matrixcsv(f'reverse-projection-{k or "full"}.csv',t,nm)
  for i,c in enumerate(nm):
   rankings.append(dict(pattern_features=k or 'full',category=c,group=gg[i],image_selects_text=nm[int(s[i].argmax())],image_top1=bool(s[i].argmax()==i),text_selects_image_same_s=nm[int(s[:,i].argmax())],text_top1_same_s=bool(s[:,i].argmax()==i),text_selects_image_reverse_projection=nm[int(t[i].argmax())],text_top1_reverse_projection=bool(t[i].argmax()==i),same_score=float(s[i,i])))
  for th in [.5,.3,.2]:
   pos=(ph>=th)&~np.eye(len(nm),dtype=bool);other=(ph<th)&~np.eye(len(nm),dtype=bool)
   for i,c in enumerate(nm):
    co=float(s[i,pos[i]].mean()) if pos[i].any() else None;ot=float(s[i,other[i]].mean()) if other[i].any() else None
    row=dict(pattern_features=k or 'full',threshold=th,category=c,group=gg[i],same=float(s[i,i]),cooccurring=co,other=ot,cooccurring_categories=int(pos[i].sum()),other_categories=int(other[i].sum()));per.append(row)
    if co is not None and s[i,i]<=co:fail.append({**row,'partners':' | '.join(nm[j] for j in np.flatnonzero(pos[i]))})
   for grp in ['all','object','stuff']:
    gmask=np.ones(len(nm),bool) if grp=='all' else gg==grp
    for cohort in ['all_eligible','has_cooccurring_partner']:
     mask=gmask&(pos.any(1) if cohort=='has_cooccurring_partner' else True);rr=[v for v in per if v['pattern_features']==(k or 'full') and v['threshold']==th and v['category'] in set(np.array(nm)[mask])]
     for typ in ['same','cooccurring','other']:
      values=[v[typ] for v in rr if v[typ] is not None];summaries.append(dict(pattern_features=k or 'full',threshold=th,group=grp,cohort=cohort,comparison=typ,categories=len(values),mean=float(np.mean(values)) if values else None,median=float(np.median(values)) if values else None))
  log(stage='condition_complete',features=k or 'full',categories=len(kept))
 csvout('category-scores.csv',per);csvout('summary.csv',summaries);csvout('top1-by-category.csv',rankings);csvout('cooccurring-pairs.csv',pairs);csvout('same-not-higher.csv',fail);save(OUT/'partner-counts.json',pair_counts)
 top=[]
 for k in ['full',8,16,64]:
  for grp in ['all','object','stuff']:
   rr=[v for v in rankings if v['pattern_features']==k and (grp=='all' or v['group']==grp)];top.append(dict(pattern_features=k,group=grp,categories=len(rr),image_top1=sum(v['image_top1'] for v in rr),text_top1_same_s=sum(v['text_top1_same_s'] for v in rr),text_top1_reverse_projection=sum(v['text_top1_reverse_projection'] for v in rr)))
 csvout('top1-summary.csv',top)
 save(OUT/'protocol.json',dict(cpu_only=True,threads=2,sae='frozen CC3M SAE',normalization='CC3M mapping FIT means and population SD; constant features set0, both vectors4096',C_dataset='all original CC3M mapping FIT pairs, not20만 sample and no COCO',C_pairs=fit_n,C_source=str(mompath),C_source_sha256=hashlib.sha256(mompath.read_bytes()).hexdigest(),image_nonconstant=len(fid['image']),text_nonconstant=len(fid['text']),image_source=str(SRC/'activations/train2017/image.npz'),text_source=str(SRC/'activations/train2017/text.npz'),presence_source=str(ANN/'full_presence.npy'),phi='binary-label Pearson correlation over all118287 original train images; group depends on query category, candidates include all eligible groups',lexicon_source=str(SRC/'dataset.json'),label_action=label_action,missing_inputs=[],train_images=N,image_pattern_images=len(ir),text_pattern_captions=len(tr),split_seed=20261012171,eligible=len(kept),excluded=[c for c in category if not c['included']],cohorts='Report all available categories and separately same query-category cohort with a cooccurring partner for fair three-value comparison',scores='s(c,cprime)=cos(C.T p_c,q_cprime)',reverse='Report both column argmax of SAME s matrix and separately cos(C q_c,p_cprime); these differ due to cosine denominator',feature_cuts='top absolute8,16,64 components of EACH p and q, signed values kept; no AUROC filter, no single-feature experiment'))
 save(OUT/'complete.json',dict(seconds=time.time()-start,categories=len(kept),conditions=4));log(stage='complete')
if __name__=='__main__':
 try:main()
 except Exception as ex:save(OUT/'error.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
