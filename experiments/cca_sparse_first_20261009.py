"""First paired coordinate only, dense vs sparse CCA, heldout category selectivity."""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
from pathlib import Path
import numpy as np,json,time,traceback
import experiments.apbt_search_20261009 as prior
from mm_sae.analysis.semantic_evaluation import auc_matrix
ROOT=prior.ROOT;OUT=ROOT/'runs/cca-sparse-first-2026-10-09';save=prior.save

def main():
 OUT.mkdir(parents=True,exist_ok=True);cfg=prior.old.load_config(ROOT/prior.CONFIG['cc3m-coco']);models=[m for m in cfg['models'] if m['key'] in ['cca_256','sparse_cca_4','sparse_cca_8','sparse_cca_16','sparse_cca_32']];cols={}
 for m in models:
  with np.load(m['path']) as z:cols[m['key']]=[z[s][:,0].copy() for s in ['image','text']]
 with np.load(Path(cfg['parent_run'])/'moments.npz') as z:
  n=cols['cca_256'][0].size;scale=np.sqrt(np.maximum(z['fit_second'].diagonal()-z['fit_mean']**2,0));scale[scale==0]=1
  cov=z['fit_second']-np.outer(z['fit_mean'],z['fit_mean']);cov/=np.outer(scale,scale);xx,xy,yy=cov[:n,:n],cov[:n,n:],cov[n:,n:];a0,b0=cols['cca_256'];orient={}
  for name,(a,b) in cols.items():
   sign=1 if a@xx@a0+b@yy@b0>=0 else -1;cols[name]=[a*sign,b*sign];orient[name]=sign
  corr={name:{} for name in cols}
  for part in ['fit','tune','test']:
   cov=z[part+'_second']-np.outer(z[part+'_mean'],z[part+'_mean']);cov/=np.outer(scale,scale);xx,xy,yy=cov[:n,:n],cov[:n,n:],cov[n:,n:]
   for name,(a,b) in cols.items():corr[name][part]=float((a@xy@b)/np.sqrt((a@xx@a)*(b@yy@b)))
 del cov,xx,xy,yy
 data,parents=prior.eval_population(cfg,'test',OUT);names=[c['name'] for c in data['concepts']];fixed=np.isin([c['id'] for c in data['concepts']],data['metadata']['eligible_category_ids']);assert fixed.sum()==50;gi=names.index('giraffe');allrows=[]
 for spec in models:
  name=spec['key'];aa,bb=cols[name];scores={s:np.asarray(data['raw'][s]@(w/data['scales'][s])).ravel()-data['means'][s]@(w/data['scales'][s]) for s,w in [('image',aa),('text',bb)]};aucs={};pair={};tops={};repeat={}
  for side in scores:
   key=side+('_a' if side=='image' else '_b');mask=data['masks'][key];v=auc_matrix(scores[side][mask,None],data['labels'][side][mask],chunk_size=32)[0];aucs[side]=v
   eligible_indices=np.flatnonzero(fixed);order=eligible_indices[np.argsort(v[eligible_indices])[::-1]];tops[side]=[dict(**data['concepts'][j],auc=float(v[j])) for j in order[:8]]
   y=data['labels'][side]
   for other in ['cow','horse']:
    j=names.index(other);selected=y[:,gi]^y[:,j];pair[side+'_'+other]=dict(auc=float(auc_matrix(scores[side][selected,None],y[selected,gi:gi+1],chunk_size=1)[0,0]),giraffe_n=int(y[selected,gi].sum()),other_n=int(y[selected,j].sum()))
   mask=data['masks'][side+('_b' if side=='image' else '_a')];repeat[side]=auc_matrix(scores[side][mask,None],data['labels'][side][mask],chunk_size=32)[0]
  categories=[dict(**c,in_fixed50=bool(fixed[j]),image_auc=float(aucs['image'][j]),text_auc=float(aucs['text'][j]),image_repeat_auc=float(repeat['image'][j]),text_repeat_auc=float(repeat['text'][j])) for j,c in enumerate(data['concepts'])]
  row=dict(method=name,source=spec['path'],correlation=corr[name],joint_sign_relative_dense=orient[name],support=[int(np.count_nonzero(a)) for a in [aa,bb]],negative_weights=[int((a<0).sum()) for a in [aa,bb]],top_categories=tops,same_top_category=tops['image'][0]['id']==tops['text'][0]['id'],giraffe_vs_other=pair,categories=categories)
  allrows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='categories'}),flush=True)
 save(OUT/'summary.json',allrows);save(OUT/'protocol.json',dict(condition='CC3M SAE / COCO2017 mapping',scope='First column only of existing sequential rank-one sparse CCA. No deflation before first component. Exact dense optimum for regularized problem, sparse local coordinate/support search not global certificate.',evaluation='50 historical eligible COCO categories, image presence halfA vs own-caption mentions halfB on val2017. Cross-modal correlation all val pairs. Pair-exclusive AUROC all val samples per modality.',orientation='Simultaneous sign to positively align with dense first pair in FIT covariance, no annotation-based sign choice.',not_matching_count='With just one output, choosing representative index among one gives tautological 50/50. Compare independently best category and exclusive-category discrimination instead.',no_new_training=True));save(OUT/'complete.json',dict(completed=True))
if __name__=='__main__':
 try:main()
 except Exception as e:save(OUT/'error.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
