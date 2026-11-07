"""Supervised diagnostic common rotation, never optimized on validation annotations."""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
from pathlib import Path
import json,time,traceback
import numpy as np
from scipy import sparse,linalg
from scipy.optimize import minimize
from scipy.special import expit
import experiments.apbt_search_20261009 as prior
from mm_sae.analysis.semantic_evaluation import auc_matrix
from mm_sae.analysis.representative_agreement import variable_coordinates
from mm_sae.analysis.mapping_evaluation import paired_retrieval
ROOT=prior.ROOT;OUT=ROOT/'runs/cca-concept-rotation-2026-10-09';save=prior.save

def roc_auc_score(y,score):
 return float(auc_matrix(np.asarray(score)[:,None],np.asarray(y)[:,None],chunk_size=1)[0,0])
def logistic_scores(x,y,v):
 x=np.c_[x,np.ones(len(x))];v=np.c_[v,np.ones(len(v))];y=y.astype(float)
 def fun(w):
  z=x@w;reg=w.copy();reg[-1]=0
  return np.logaddexp(0,z).sum()-y@z+.5*(reg@reg), x.T@(expit(z)-y)+reg
 fit=minimize(fun,np.zeros(x.shape[1]),jac=True,method="L-BFGS-B",options={"maxiter":2000,"ftol":1e-12,"gtol":1e-7})
 assert fit.success,fit.message
 return v@fit.x

def status(**v):save(OUT/'progress.json',dict(time=time.time(),**v));print(json.dumps(v),flush=True)
def main():
 OUT.mkdir(parents=True,exist_ok=True);cfg=prior.old.load_config(ROOT/prior.CONFIG['cc3m-coco']);source=Path(cfg['source_run']);parent=Path(cfg['parent_run']);pop=json.loads((parent/'population.json').read_text())
 model=Path(next(m['path'] for m in cfg['models'] if m['key']=='cca_256'))
 with np.load(model) as z:w={s:z[s].copy() for s in ['image','text']}
 with np.load(parent/'moments.npz') as z:
  ids={s:z[s+'_ids'].copy() for s in w};ni=len(ids['image']);mean=z['fit_mean'];scale=np.sqrt(np.maximum(z['fit_second'].diagonal()-mean**2,0));scale[scale==0]=1
 means={'image':mean[:ni],'text':mean[ni:]};scales={'image':scale[:ni],'text':scale[ni:]}
 ix=source/'index/train2017';imids=np.array([r['image_id'] for r in json.loads((ix/'images.json').read_text())]);fit=np.flatnonzero(np.isin(imids,pop['fit_image_ids']));rng=np.random.default_rng(9109);fit=np.sort(rng.choice(fit,min(32768,len(fit)),replace=False))
 par=np.load(ix/'parents.npy');order=np.argsort(par,kind='stable');counts=np.bincount(par,minlength=len(imids));starts=np.r_[0,counts.cumsum()[:-1]];texts=order[starts[fit]+(rng.random(len(fit))*counts[fit]).astype(int)]
 assert not set(imids[fit])&set(pop['test_image_ids']);np.savez_compressed(OUT/'split.npz',image_rows=fit,text_rows=texts,image_ids=imids[fit])
 concepts=np.array(json.loads((ix/'concept_ids.json').read_text()));obj=np.flatnonzero(concepts<91);assert len(obj)==80
 trainlabels={'image':np.load(ix/'presence.npy')[fit][:,obj].astype(bool),'text':np.load(ix/'mentions.npy')[texts][:,obj].astype(bool)};tr={}
 for side,rows in [('image',fit),('text',texts)]:
  status(state='loading_train',side=side);raw=sparse.load_npz(source/'activations/train2017'/(side+'.npz')).tocsr()[rows][:,ids[side]];scaled=w[side]/scales[side][:,None];tr[side]=np.asarray(raw@scaled)-means[side]@scaled;del raw
 # Unit-variance concept labels balance frequent and rare targets, separately per modality.
 cross=np.zeros((256,256));labelstats={}
 for side in w:
  y=trainlabels[side].astype(float);mu=y.mean(0);sd=y.std(0);valid=sd>0;sd[~valid]=1;target=(y-mu)/sd;cross[:,:80]+=tr[side].T@target/len(target)/2
  labelstats[side]=dict(mean=mu,scale=sd,positive=y.sum(0),valid=valid)
 left,sing,right=linalg.svd(cross,full_matrices=True);rot=left@right
 np.testing.assert_allclose(rot.T@rot,np.eye(256),atol=1e-10)
 np.savez_compressed(OUT/'rotation.npz',rotation=rot,singular_values=sing,category_ids=concepts[obj]);np.savez_compressed(OUT/'rotated.npz',image=w['image']@rot,text=w['text']@rot)
 save(OUT/'protocol.json',dict(condition='CC3M SAE / COCO2017 mapping',train_images=len(fit),train_captions=len(texts),sampling_seed=9109,train_source='COCO mapping FIT images only; one caption per image',rotation_objective='min_R 0.5 mean||U R - T_image||^2 +0.5 mean||V R - T_text||^2 subject R.T R=I. First80 target columns own-modality standardized binary labels, remaining176 zero. Closed-form SVD of average U.T T and V.T T.',label_stats=labelstats,labels_text='Own-caption dictionary mentions',evaluation='COCO val2017 disjoint from training',limitations='Supervised diagnostic, no sparsity/nonnegativity constraints; unused complement rotation not unique; one training sample seed.'))
 status(state='evaluating');data,parents=prior.eval_population(cfg,'test',OUT);test={s:np.asarray(data['raw'][s]@(w[s]/data['scales'][s][:,None]))-data['means'][s]@(w[s]/data['scales'][s][:,None]) for s in w};rotated={s:t@rot for s,t in test.items()}
 fixed=np.isin([c['id'] for c in data['concepts']],data['metadata']['eligible_category_ids']);assert fixed.sum()==50;data['eligible']=fixed
 # Algebraic full cosine invariance; explicit numerical check on fixed random submatrix.
 norm=lambda v:v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
 q=np.random.default_rng(42).choice(len(test['image']),128,False);t=np.random.default_rng(43).choice(len(test['text']),256,False)
 delta=np.max(np.abs(norm(test['image'][q])@norm(test['text'][t]).T-norm(rotated['image'][q])@norm(rotated['text'][t]).T))
 results={};concept_results={}
 for name,scores in [('cca',test),('common_rotation',rotated)]:
  ret=paired_retrieval(scores['image'],scores['text'],parents,device='cuda',chunk_size=128);arr={}
  for side in scores:
   for half in ['a','b']:
    key=side+'_'+half;mask=data['masks'][key];arr[key]=auc_matrix(scores[side][mask],data['labels'][side][mask],chunk_size=32);arr[key+'_variable']=variable_coordinates(scores[side][mask])
  arr['eligible']=fixed;arr['category_ids']=np.array([c['id'] for c in data['concepts']]);match=prior.old.evaluation(arr,{**cfg,'n_null':0},signed=False)
  results[name]=dict(retrieval={d:{k:v for k,v in z.items() if k!='hits'} for d,z in ret.items()},matching50=match);np.savez_compressed(OUT/(name+'-auc.npz'),**arr)
  save(OUT/(name+'.json'),results[name])
 # Train-selected baseline coordinate vs predetermined annotated coordinate after rotation.
 for j,c in enumerate(data['concepts']):
  if not fixed[j]:continue
  row=dict(**c)
  for side in w:
   trainauc=auc_matrix(tr[side],trainlabels[side][:,j:j+1],chunk_size=1)[:,0];k=int(np.argmax(trainauc));y=data['labels'][side][:,j]
   row[side]=dict(before_train_selected_coordinate=k,before_test_auc=float(roc_auc_score(y,test[side][:,k])),after_annotated_coordinate=j,after_test_auc=float(roc_auc_score(y,rotated[side][:,j])))
  concept_results[str(c['id'])]=row
 save(OUT/'assigned-concepts.json',concept_results)
 names=[c['name'] for c in data['concepts']];pairs=[]
 for cname in ['cow','horse']:
  gi=names.index('giraffe');ci=names.index(cname)
  for side in w:
   tl=trainlabels[side];vl=data['labels'][side];fm=tl[:,gi]^tl[:,ci];vm=vl[:,gi]^vl[:,ci];yt=tl[fm,gi];yv=vl[vm,gi];row=dict(side=side,concept_a='giraffe',concept_b=cname,train_a=int(yt.sum()),train_b=int((~yt).sum()),test_a=int(yv.sum()),test_b=int((~yv).sum()))
   for key,features in [('first_coordinate',[0]),('all_256',np.arange(256))]:
    x=tr[side][fm][:,features];v=test[side][vm][:,features];mu=x.mean(0);sd=x.std(0);sd[sd==0]=1;pred=logistic_scores((x-mu)/sd,yt,(v-mu)/sd);row[key]=float(roc_auc_score(yv,pred))
   row['rotated_giraffe_coordinate']=float(roc_auc_score(yv,rotated[side][vm,gi]));row['rotated_giraffe_minus_other']=float(roc_auc_score(yv,rotated[side][vm,gi]-rotated[side][vm,ci]));pairs.append(row)
 save(OUT/'pairwise.json',pairs)
 # Exact retrieval rank equality may be perturbed by float32 GPU rounding; record, never assume.
 agreement={d:dict(rank_changes=int(np.count_nonzero(np.array(results['cca']['retrieval'][d]['ranks'])!=np.array(results['common_rotation']['retrieval'][d]['ranks'])))) for d in results['cca']['retrieval']}
 save(OUT/'verification.json',dict(orthogonality_max=float(np.max(np.abs(rot.T@rot-np.eye(256)))),sample_cosine_max_difference=float(delta),retrieval_rank_checks=agreement,no_train_test_overlap=True))
 save(OUT/'complete.json',dict(completed=True));status(state='complete',pairwise=pairs,matching={k:v['matching50']['summary']['agree_at1_count'] for k,v in results.items()})
if __name__=='__main__':
 try:main()
 except Exception as e:save(OUT/'error.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
