"""Matched CCA rank and caption-concept count, FIT-only shared orthogonal rotation."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import json,time,traceback
import numpy as np
from scipy import sparse,linalg
import experiments.apbt_search_20261009 as prior
from mm_sae.analysis.mapping_evaluation import paired_retrieval
from mm_sae.analysis.semantic_evaluation import auc_matrix
from mm_sae.analysis.representative_agreement import variable_coordinates
ROOT=prior.ROOT;OUT=ROOT/'runs/cc3m-caption-rank-sweep-2026-10-09';save=prior.save
START=time.time()
def status(**v):
 save(OUT/'progress.json',{'elapsed':time.time()-START,**v});print(json.dumps(v),flush=True)
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 p=ROOT/'runs/cc3m-caption-concepts-2026-10-09/extraction';rows=np.load(p/'activation-rows.npy');labels=sparse.load_npz(p/'top4096-labels.npz').astype(np.float64);vocab=json.loads((p/'top4096.json').read_text())
 cfg=prior.old.load_config(ROOT/prior.CONFIG['cc3m-cc3m']);parent=Path(cfg['parent_run']);source=ROOT/'runs/cc3m-sae-2026-10-04'
 with np.load(parent/'moments.npz') as z:
  ids={s:z[s+'_ids'].copy() for s in ['image','text']};ni=len(ids['image']);mean=z['fit_mean'];scale=np.sqrt(np.maximum(z['fit_second'].diagonal()-mean**2,0));scale[scale==0]=1;assert int(z['fit_n'])==len(rows)
 means={'image':mean[:ni],'text':mean[ni:]};scales={'image':scale[:ni],'text':scale[ni:]}
 with np.load(ROOT/'runs/mapping-ablation-cc3m-2026-10-04/transforms/cca_3221.npz') as z:w={s:z[s].copy() for s in ids}
 maxrank=min(len(q) for q in ids.values());grid=[256,512,1024,2048,maxrank];assert maxrank==3221
 save(OUT/'protocol.json',dict(sae_training='CC3M',mapping_training='CC3M FIT',fit_captions=len(rows),requested=[256,512,1024,2048,4096],evaluated=grid,unavailable_4096='Existing FIT feature dimensions are3695 image and3221 text; effective common rank cannot exceed3221.',concepts='Nested frequency top r noun/noun-phrase labels from FIT captions only.',labels='Same paired caption mention labels supervise both modalities. Not verified image presence. No COCO annotations used for fitting.',rotation='min_R 0.5 E||U R-T||^2+0.5 E||V R-T||^2 subject R.T R=I, standardized binary T; r CCA coordinates and r concept targets.',cca_ridge=.01,cca_source='Cached full3221 ridgeCCA, leading r columns. Same FIT normalization and no singular-value reweighting.',evaluation='COCO val2017 retrieval R1/5/10 both directions. Category names evaluation-only. Shared orthogonal rotation preserves cosine, not assumed to improve retrieval.'))
 mu=np.asarray(labels.mean(0)).ravel();sd=np.sqrt(mu*(1-mu));assert np.all(sd>0)
 momentpath=OUT/'cca-label-cross.npz'
 if not momentpath.exists():
  crosses={};audit={}
  for side in ids:
   status(stage='fit_caption_cross_moment',side=side)
   raw=sparse.load_npz(source/'activations/train2017'/(side+'.npz')).tocsr()[rows].astype(np.float64)
   sums=np.asarray(raw.sum(0)).ravel();seconds=np.asarray(raw.power(2).sum(0)).ravel()/len(rows);vary=seconds-(sums/len(rows))**2
   audit[side]=dict(total_coordinates=raw.shape[1],nonconstant_coordinates=int((vary>1e-16).sum()),used_coordinates=len(ids[side]))
   cross=(raw[:,ids[side]].T@labels[:,:maxrank]).toarray()/len(rows)
   cross-=means[side][:,None]*mu[None,:maxrank];cross/=scales[side][:,None]*sd[None,:maxrank]
   crosses[side]=w[side].T@cross;del raw,cross
  np.savez_compressed(momentpath,**crosses);save(OUT/'feature-audit.json',audit)
 with np.load(momentpath) as z:cross=(z['image']+z['text'])/2
 status(stage='load_evaluation');data,parents=prior.eval_population(cfg,'test',OUT);test={s:np.asarray(data['raw'][s]@(w[s]/scales[s][:,None]))-means[s]@(w[s]/scales[s][:,None]) for s in w}
 vx=Path(cfg['source_run'])/'index/val2017';presence=np.load(vx/'presence.npy').astype(bool);assert presence.shape==(5000,171)
 names=json.loads((ROOT/'runs/cca-171-category-names-2026-10-09/protocol.json').read_text())['prototype_texts'];rawname=np.load(ROOT/'runs/cca-171-category-names-2026-10-09/category-name-activations.npz')['activation'];proto=rawname[:,ids['text']]@(w['text']/scales['text'][:,None])-means['text']@(w['text']/scales['text'][:,None])
 fixed=np.isin([c['id'] for c in data['concepts']],data['metadata']['eligible_category_ids']);assert fixed.sum()==50
 for r in grid:
  if (OUT/f'result-{r}.json').exists():continue
  status(stage='rotation_svd',rank=r)
  u,sv,vt=linalg.svd(cross[:r,:r],full_matrices=False);rot=u@vt;err=float(np.max(abs(rot.T@rot-np.eye(r))));assert err<1e-10
  np.savez_compressed(OUT/f'model-{r}.npz',image=w['image'][:,:r]@rot,text=w['text'][:,:r]@rot,rotation=rot)
  methods={}
  for name,rr in [('cca',np.eye(r)),('caption_rotation',rot)]:
   status(stage='evaluation',rank=r,method=name)
   scores={s:test[s][:,:r]@rr for s in w};ret=paired_retrieval(scores['image'],scores['text'],parents,device='cuda',chunk_size=128)
   auc=auc_matrix(scores['image'],presence,chunk_size=32);pt=proto[:,:r]@rr;ki=auc.argmax(0);kt=pt.argmax(1);match=ki==kt
   arr={}
   for side in scores:
    for half in ['a','b']:
     key=side+'_'+half;mask=data['masks'][key];arr[key]=auc_matrix(scores[side][mask],data['labels'][side][mask],chunk_size=32);arr[key+'_variable']=variable_coordinates(scores[side][mask])
   arr['eligible']=fixed;arr['category_ids']=np.array([c['id'] for c in data['concepts']]);m50=prior.old.evaluation(arr,{**cfg,'n_null':0},signed=False)
   methods[name]=dict(retrieval={d:{k:v for k,v in val.items() if k!='hits'} for d,val in ret.items()},matching50=m50,matching171=dict(matched=int(match.sum()),total=171,distinct_matched=len(set(ki[match].tolist())),categories=[dict(name=n,image_coordinate=int(i),text_coordinate=int(t),matched=bool(i==t),image_auc_at_text=float(auc[t,j])) for j,(n,i,t) in enumerate(zip(names,ki,kt))]))
  rankchanges={d:int(np.count_nonzero(np.array(methods['cca']['retrieval'][d]['ranks'])!=np.array(methods['caption_rotation']['retrieval'][d]['ranks']))) for d in methods['cca']['retrieval']}
  save(OUT/f'result-{r}.json',dict(rank=r,concepts=r,orthogonality_error=err,retrieval_rank_changes=rankchanges,methods=methods,elapsed=time.time()-START));status(stage='rank_complete',rank=r,elapsed=time.time()-START)
 save(OUT/'complete.json',dict(completed=True,elapsed=time.time()-START));status(stage='complete')
if __name__=='__main__':
 try:main()
 except Exception as e:
  OUT.mkdir(parents=True,exist_ok=True);save(OUT/'error.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
