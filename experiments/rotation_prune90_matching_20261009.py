"""Repeat unchanged annotation representative agreement on frozen 90%-mass pruned matrices."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json,time
import numpy as np
import experiments.apbt_search_20261009 as p
from mm_sae.analysis.semantic_evaluation import auc_matrix
from mm_sae.analysis.representative_agreement import variable_coordinates
ROOT=p.ROOT;SRC=ROOT/'runs/rotation-convergence-2026-10-09';OUT=SRC/'prune90-matching';OUT.mkdir(exist_ok=True)

def main():
 start=time.time();cfg=p.old.load_config(ROOT/p.CONFIG['cc3m-cc3m']);data,_=p.eval_population(cfg,'test',OUT)
 jobs=sorted((SRC/'prune90').glob('*-weights.npz'));assert len(jobs)==22
 for f in jobs:
  name=f.name.replace('-weights.npz','');dest=OUT/(name+'.json')
  if dest.exists():continue
  beforepath=ROOT/'runs/cca-unsupervised-rotations-2026-10-09/cc3m-cc3m/cca_original-result.json' if name=='cca_original' else SRC/f'{name}-final-result.json'
  old=json.loads(beforepath.read_text());arr={}
  with np.load(f) as z:
   for s in ['image','text']:
    scaled=z[s]/data['scales'][s][:,None];scores=np.asarray(data['raw'][s]@scaled)-data['means'][s]@scaled
    for half in ['a','b']:
     key=s+'_'+half;mask=data['masks'][key]
     arr[key]=auc_matrix(scores[mask],data['labels'][s][mask],chunk_size=32)
     arr[key+'_variable']=variable_coordinates(scores[mask])
  arr['eligible']=np.isin([c['id'] for c in data['concepts']],data['metadata']['eligible_category_ids']);assert arr['eligible'].sum()==50
  arr['category_ids']=np.array([c['id'] for c in data['concepts']])
  unsigned=p.old.evaluation(arr,{**cfg,'n_null':0},signed=False)
  signed=p.old.evaluation(arr,{**cfg,'n_null':0},signed=True)
  previous={c['category_id']:c for c in old['matching50']['per_category'] if c['status']=='ok'}
  current={c['category_id']:c for c in unsigned['per_category'] if c['status']=='ok'}
  assert previous.keys()==current.keys() and len(current)==50
  gained=[c for c in current if current[c]['agree_at1'] and not previous[c]['agree_at1']]
  lost=[c for c in current if previous[c]['agree_at1'] and not current[c]['agree_at1']]
  before=old['matching50']['summary']['agree_at1_count'];after=unsigned['summary']['agree_at1_count']
  assert after-before==len(gained)-len(lost)
  changes=dict(before_count=before,after_count=after,delta=after-before,gained_category_ids=gained,lost_category_ids=lost,image_representative_changed=sum(current[c]['image_coordinate']!=previous[c]['image_coordinate'] for c in current),text_representative_changed=sum(current[c]['text_coordinate']!=previous[c]['text_coordinate'] for c in current),distinct_matched_coordinates=len(set(c['image_coordinate'] for c in current.values() if c['agree_at1'])))
  p.save(dest,dict(name=name,condition='CC3M SAE / CC3M mapping',matching50=unsigned,signed_matching50=signed,changes=changes))
  print(name,before,after,round(time.time()-start,1),flush=True)
 p.save(OUT/'protocol.json',dict(condition='CC3M SAE / CC3M mapping',evaluation='Exact same fixed50 categories, masks, native coordinate index tie policy, and positive-polarity representative AUROC criterion as convergence experiment. Image object labels and own-caption mentions, not propagated image labels for text.',pruning='Use saved A/B from prune90 experiment without additional operations.',signs='Keep signs fixed from before pruning. No annotation-based sign choice for primary metric. Signed metric supplementary only.',training_performed=False,category_denominator_changed=False,models=22,seconds=time.time()-start))

if __name__=='__main__':main()
