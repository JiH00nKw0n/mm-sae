import os
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import json,time,traceback
import numpy as np
from scipy import linalg
import experiments.representative_agreement.run as old
from mm_sae.metrics.regression import Moments
from mm_sae.io import sha256
ROOT=Path('/mnt/working/mm-sae');OUT=ROOT/'runs/caption-matching-2026-10-09'
base_load=old.load_population

def load_population(cfg,kind):
 data=base_load(cfg,kind);idx=Path(cfg['source_run'])/'index/val2017'
 ids=np.array([r['image_id'] for r in json.loads((idx/'images.json').read_text())]);par=np.load(idx/'parents.npy')
 keep=np.isin(ids,data['metadata']['half_a_image_ids']+data['metadata']['half_b_image_ids'])
 objects=np.flatnonzero(np.array(json.loads((idx/'concept_ids.json').read_text()))<91)
 data['labels']['text']=np.load(idx/'mentions.npy')[keep[par]][:,objects].astype(bool)
 data['counts']={k:data['labels'][k.split('_')[0]][m].sum(0) for k,m in data['masks'].items()}
 data['eligible']=(data['counts']['image_a']>=50)&(data['counts']['text_b']>=50)
 for k in ['image_a','text_b']:data['eligible'] &= data['counts'][k]<data['masks'][k].sum()
 md=data['metadata'];md['n_eligible_categories']=int(data['eligible'].sum());md['positive_counts']={k:v.tolist() for k,v in data['counts'].items()};md['eligible_category_ids']=[c['id'] for c,ok in zip(data['concepts'],data['eligible']) if ok]
 md['source_hashes'][str(idx/'mentions.npy')]=sha256(idx/'mentions.npy')
 md['text_annotation']='Own-caption dictionary mentions; not inherited image labels; no claim of human semantic gold labels.'
 return data
old.load_population=load_population

def projection(dest,key,a,b):
 p=dest/'weights'/f'{key}.npz';p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():np.savez_compressed(p,image=a,text=b)
 return dict(key=key,label=key,kind='projection',path=str(p))

def main():
 total=0
 for cond,config in [('coco-coco','configs/representative-agreement-local.yaml'),('cc3m-cc3m','runs/cc3m-followup-2026-10-05/configs/cc3m-fit-agreement.yaml'),('cc3m-coco','runs/cc3m-followup-2026-10-05/configs/coco-fit-agreement.yaml')]:
  cfg=old.load_config(ROOT/config);dest=OUT/cond;dest.mkdir(parents=True,exist_ok=True)
  parent=Path(cfg['parent_run']);follow=ROOT/'runs/cc3m-followup-2026-10-05'/('cc3m-fit' if cond=='cc3m-cc3m' else 'coco-fit')
  abl=ROOT/'runs/mapping-ablation-2026-10-04' if cond=='coco-coco' else ROOT/'runs/mapping-ablation-cc3m-2026-10-04' if cond=='cc3m-cc3m' else follow/'ablation'
  prune=ROOT/'runs/mapping-pruning-2026-10-04' if cond=='coco-coco' else follow/'pruning'
  with np.load(parent/'moments.npz') as z:
   fit=Moments(int(z['fit_n']),z['fit_mean'],z['fit_second']);ni=len(z['image_ids']);nt=len(z['text_ids']);ids={s:z[s+'_ids'].copy() for s in ['image','text']}
  specs=[m for m in cfg['models'] if m['key'] in ['hungarian','cca_256','sparse_cca_4','sparse_cca_8','sparse_cca_16','sparse_cca_32']]
  specs.append(dict(key='cross_svd_256',label='PLS-SVD 256',kind='projection',path=str(abl/'transforms/cross_svd_256.npz')))
  pp=dest/'weights/procrustes.npz'
  if pp.exists():specs.append(dict(key='procrustes',label='procrustes',kind='projection',path=str(pp)))
  else:
   cv=fit.standardized(fit);u,_,vt=linalg.svd(cv[:ni,ni:],full_matrices=True);del cv
   width=max(ni,nt);specs.append(projection(dest,'procrustes',np.pad(u,((0,0),(0,width-ni))),np.pad(vt.T,((0,0),(0,width-nt)))))
  selected=json.loads((parent/'selected.json').read_text());sel=next(v for v in selected if v['family']=='sinkhorn')
  for sparse in [False,True]:
   path=prune/'sinkhorn/transforms/sinkhorn_e0.05_k16.npz' if sparse else parent/'candidates'/(sel['key']+'.npz')
   if not path.exists() and cond=='coco-coco':path=OUT/'inputs/coco-sinkhorn16.npz'
   with np.load(path) as z:m=z['mapping'].copy()
   rs=m.sum(1);cs=m.sum(0)
   f=np.divide(m,rs[:,None],out=np.zeros_like(m),where=rs[:,None]>0);b=np.divide(m,cs[None,:],out=np.zeros_like(m),where=cs[None,:]>0)
   for direction in ['text','image']:
    key=('sinkhorn_'+direction+'_16') if sparse else 'sinkhorn__standardized__'+('text_projected_to_image' if direction=='text' else 'image_projected_to_text')
    specs.append(projection(dest,key,np.eye(ni) if direction=='text' else b,f.T if direction=='text' else np.eye(nt)))
  specs.append(dict(key='pls_16',label='Sparse PLS-SVD 16',kind='projection',path=str(ROOT/'runs/sparse-pls-2026-10-06'/cond/'transform.npz')))
  if cond!='cc3m-cc3m':
   oracle=ROOT/'runs/oracle-sets-2026-10-04' if cond=='coco-coco' else follow/'annotation-sets';ww=[]
   for side,key,sl in [('image','image',slice(0,ni)),('text','propagated_presence',slice(ni,None))]:
    with np.load(oracle/'fits'/f'{key}_16.npz') as z:
     np.testing.assert_array_equal(ids[side],z['feature_ids']);ww.append(z['coefficients']/z['feature_scale'][:,None]*fit.scale[sl,None])
   specs.append(projection(dest,'oracle_ridge_16',*ww))
  if cond=='cc3m-coco':
   for seed in range(3):
    folder=ROOT/'runs'/('contrastive-concepts-2026-10-09'+('' if seed==0 else f'-seed{seed}'))
    for name in ['nce_centered','nce_uncentered','nce_nonnegative']+(['sparse_cca_unitnorm'] if seed==0 else []):
     p=folder/(name+'.npz');specs.append(dict(key=name+f'_seed{seed}',label=name,kind='projection',path=str(p)))
  cfg.update(models=specs,populations=['test'],output=str(dest))
  cfg['protocol']={'annotation':'Image object presence and OWN caption dictionary mentions evaluated separately.','training':cond,'scope':'Independent representative identity agreement; no refitting or label-based rematching.','min_positive':50,'coordinate_definition':'CCA/PLS use stored paired columns. Sinkhorn compares actual retrieval-transformed coordinates. Procrustes uses the SVD common basis used for retrieval, not unique semantic concepts.','differences_from_rebuttal':['Own-caption dictionary labels. 50-positive threshold retained; denominator recomputed.'],'text_labels':'Own caption mentions.npy; not image labels.'}
  old.save_json(dest/'configuration.json',cfg);old.run(cfg);total+=len(specs)
 old.save_json(OUT/'complete.json',dict(completed=True,models=total))
if __name__=='__main__':
 try:main()
 except Exception as e:
  old.save_json(OUT/'error.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
