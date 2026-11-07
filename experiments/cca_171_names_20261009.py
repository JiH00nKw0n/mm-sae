"""171-label supervised common rotation, category-name-only text evaluation."""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
from pathlib import Path
from types import SimpleNamespace
import json,time,traceback
import numpy as np
from scipy import sparse,linalg
import torch
import experiments.apbt_search_20261009 as prior
from mm_sae.analysis.semantic_evaluation import auc_matrix
from mm_sae.analysis.mapping_evaluation import paired_retrieval
from mm_sae.models.encoder import CLIPEncoder
from mm_sae.models.sae import TopKSAE
ROOT=prior.ROOT;OUT=ROOT/'runs/cca-171-category-names-2026-10-09';PREV=ROOT/'runs/cca-concept-rotation-2026-10-09';save=prior.save

def status(**v):save(OUT/'progress.json',dict(time=time.time(),**v));print(json.dumps(v),flush=True)
def main():
 OUT.mkdir(parents=True,exist_ok=True);cfg=prior.old.load_config(ROOT/prior.CONFIG['cc3m-coco']);source=Path(cfg['source_run']);parent=Path(cfg['parent_run']);pop=json.loads((parent/'population.json').read_text());ix=source/'index/train2017';concept_ids=np.array(json.loads((ix/'concept_ids.json').read_text()));assert len(concept_ids)==171
 lookup={c['id']:c['name'] for c in json.loads((source/'dataset.json').read_text())['concepts']};names=[lookup[int(i)] for i in concept_ids]
 model=Path(next(m['path'] for m in cfg['models'] if m['key']=='cca_256'))
 with np.load(model) as z:w={s:z[s].copy() for s in ['image','text']}
 with np.load(parent/'moments.npz') as z:
  ids={s:z[s+'_ids'].copy() for s in w};ni=len(ids['image']);mean=z['fit_mean'];scale=np.sqrt(np.maximum(z['fit_second'].diagonal()-mean**2,0));scale[scale==0]=1
 means={'image':mean[:ni],'text':mean[ni:]};scales={'image':scale[:ni],'text':scale[ni:]}
 with np.load(PREV/'split.npz') as z:fit=z['image_rows'];texts=z['text_rows'];fitids=z['image_ids'];assert not set(fitids)&set(pop['test_image_ids'])
 labels={'image':np.load(ix/'presence.npy')[fit].astype(bool),'text':np.load(ix/'mentions.npy')[texts].astype(bool)};cross=np.zeros((256,256));trainstats={}
 for side,rows in [('image',fit),('text',texts)]:
  status(state='fit_rotation',side=side);raw=sparse.load_npz(source/'activations/train2017'/(side+'.npz')).tocsr()[rows][:,ids[side]];scaled=w[side]/scales[side][:,None];score=np.asarray(raw@scaled)-means[side]@scaled;del raw
  y=labels[side].astype(float);mu=y.mean(0);sd=y.std(0);valid=sd>0;sd[~valid]=1;target=(y-mu)/sd;cross[:,:171]+=score.T@target/len(target)/2;trainstats[side]=dict(positive=y.sum(0),valid=valid,mean=mu,scale=sd);del score
 left,sing,right=linalg.svd(cross,full_matrices=True);r=left@right;np.testing.assert_allclose(r.T@r,np.eye(256),atol=1e-10);np.savez_compressed(OUT/'rotation171.npz',rotation=r,singular_values=sing);np.savez_compressed(OUT/'rotated171.npz',image=w['image']@r,text=w['text']@r)
 save(OUT/'protocol.json',dict(condition='CC3M SAE / COCO2017 mapping',fit_images=len(fit),fit_captions=len(texts),text_training='Own-caption mentions for171 concepts, never propagated image labels; zero-count concept targets become zero.',trainstats=trainstats,concepts=[dict(id=int(i),name=n) for i,n in zip(concept_ids,names)],rotation='Same orthogonal Procrustes as prior80 experiment, now first171 columns supervised and remaining85 zero.',prototype_texts=names,prototype_template='Exact dataset category name only. No a-photo-of template, no prompt ensemble.',prototypes_used_for_training=False,evaluation='COCO val2017 image AUROC over all5000 images selects image representative. Highest signed score across256 coordinates for a single category-name text selects text representative. No sign fitting or forced unique assignment.',limitations='A single name input has no AUROC; this evaluation differs from own-caption AUROC matching. Class names may be ambiguous, especially stuff labels.'))
 # Encode exact category names only once, with the frozen original CLIP and text SAE.
 status(state='encode_names');encfg=json.loads((ROOT/'runs/cc3m-sae-2026-10-04/resolved-config.json').read_text())['encoder'];encoder=CLIPEncoder(SimpleNamespace(**encfg));emb=np.concatenate([encoder.texts(names[i:i+64]) for i in range(0,len(names),64)]);del encoder;torch.cuda.empty_cache();sae=TopKSAE.from_pretrained(source/'models/text').to('cuda').eval()
 with torch.inference_mode():
  val,ind=sae.encode(torch.tensor(emb,device='cuda'));raw=torch.zeros((len(names),sae.config.latent_size),device='cuda').scatter(1,ind,val).cpu().numpy()
 np.savez_compressed(OUT/'category-name-activations.npz',embedding=emb,activation=raw,category_ids=concept_ids);del sae;torch.cuda.empty_cache();scaled=w['text']/scales['text'][:,None];proto=raw[:,ids['text']]@scaled-means['text']@scaled
 status(state='evaluate');data,parents=prior.eval_population(cfg,'test',OUT);vx=source/'index/val2017';allids=np.array([x['image_id'] for x in json.loads((vx/'images.json').read_text())]);keep=np.isin(allids,pop['test_image_ids']);y=np.load(vx/'presence.npy')[keep].astype(bool);assert y.shape==(5000,171);np.testing.assert_array_equal(concept_ids,np.array(json.loads((vx/'concept_ids.json').read_text())))
 scores={s:np.asarray(data['raw'][s]@(w[s]/scales[s][:,None]))-means[s]@(w[s]/scales[s][:,None]) for s in w};rots={'cca':np.eye(256),'rotation80':np.load(PREV/'rotation.npz')['rotation'],'rotation171':r};allresults={}
 for name,rr in rots.items():
  si=scores['image']@rr;st=scores['text']@rr;pt=proto@rr;aucs=auc_matrix(si,y,chunk_size=32);counts=y.sum(0);eligible=(counts>0)&(counts<len(y));rows=[]
  for j,(cid,cname) in enumerate(zip(concept_ids,names)):
   if not eligible[j]:rows.append(dict(id=int(cid),name=cname,eligible=False,image_positive=int(counts[j])));continue
   k=int(np.argmax(aucs[:,j]));l=int(np.argmax(pt[j]));ordered=np.sort(pt[j]);rows.append(dict(id=int(cid),name=cname,group='object' if cid<91 else 'background',eligible=True,image_positive=int(counts[j]),image_coordinate=k,text_coordinate=l,matched=k==l,image_auc=float(aucs[k,j]),image_auc_at_text_coordinate=float(aucs[l,j]),text_score=float(pt[j,l]),text_top_gap=float(ordered[-1]-ordered[-2]),intended_coordinate=j,match_intended=(k==j and l==j) if name=='rotation171' else None))
  summary={}
  for group in ['all','object','background']:
   selected=[q for q in rows if q['eligible'] and (group=='all' or q['group']==group)];matched=[q for q in selected if q['matched']]
   summary[group]=dict(total=171 if group=='all' else 80 if group=='object' else 91,evaluable=len(selected),matched=len(matched),distinct_matched=len({q['image_coordinate'] for q in matched}),distinct_image=len({q['image_coordinate'] for q in selected}),distinct_text=len({q['text_coordinate'] for q in selected}),intended_matches=sum(bool(q['match_intended']) for q in selected),mean_image_auc_at_text_coordinate=float(np.mean([q['image_auc_at_text_coordinate'] for q in selected])))
  ret=paired_retrieval(si,st,parents,device='cuda',chunk_size=128);result=dict(summary=summary,categories=rows,retrieval={d:{k:v for k,v in v.items() if k!='hits'} for d,v in ret.items()});save(OUT/(name+'.json'),result);allresults[name]=result;np.savez_compressed(OUT/(name+'-scores.npz'),image_auc=aucs,text_name_scores=pt,category_ids=concept_ids);print(json.dumps(dict(method=name,summary=summary)),flush=True)
 # Caption evaluation under old50 alongside name171 isolates the changed text protocol.
 from mm_sae.analysis.representative_agreement import variable_coordinates
 fixed=np.isin([c['id'] for c in data['concepts']],data['metadata']['eligible_category_ids']);arr={}
 for side in w:
  score=scores[side]@r
  for half in ['a','b']:
   key=side+'_'+half;mask=data['masks'][key];arr[key]=auc_matrix(score[mask],data['labels'][side][mask],chunk_size=32);arr[key+'_variable']=variable_coordinates(score[mask])
 arr['eligible']=fixed;arr['category_ids']=np.array([c['id'] for c in data['concepts']]);save(OUT/'rotation171-caption50.json',prior.old.evaluation(arr,{**cfg,'n_null':0},signed=False))
 for name in rots:
  for d in allresults['cca']['retrieval']:np.testing.assert_array_equal(allresults[name]['retrieval'][d]['ranks'],allresults['cca']['retrieval'][d]['ranks'])
 save(OUT/'verification.json',dict(no_train_test_overlap=True,prototypes_not_used_for_rotation=True,all_retrieval_ranks_equal=True,rotation_orthogonality_error=float(np.max(np.abs(r.T@r-np.eye(256))))));save(OUT/'complete.json',dict(completed=True));status(state='complete')
if __name__=='__main__':
 try:main()
 except Exception as e:save(OUT/'error.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
