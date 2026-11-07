"""Evaluate uncentered nonnegative concept sums, with fixed original 50 categories."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import json,time,traceback
import numpy as np
import experiments.apbt_search_20261009 as prior
from mm_sae.analysis.mapping_evaluation import paired_retrieval
ROOT=prior.ROOT; OUT=ROOT/'runs/raw-positive-2026-10-09'; OLD=ROOT/'runs/nonnegative-barlow-2026-10-09'
NAMES=['cca_256','sparse_cca_16','positive_initialization','nonnegative_correlation','nonnegative_diagonal','nonnegative_barlow','nonnegative_barlow_16']
save=prior.save

def main():
 for cond in prior.CONFIG:
  dest=OUT/cond;dest.mkdir(parents=True,exist_ok=True)
  cfg=prior.old.load_config(ROOT/prior.CONFIG[cond]);data,parents=prior.eval_population(cfg,'test',dest)
  ids=np.array([c['id'] for c in data['concepts']]);fixed=np.isin(ids,data['metadata']['eligible_category_ids']);assert fixed.sum()==50
  counts={k:data['labels'][k.split('_')[0]][m].sum(0) for k,m in data['masks'].items()}
  audit=[]
  for c,keep,eligible,ni,nt in zip(data['concepts'],fixed,data['eligible'],counts['image_a'],counts['text_b']):
   if keep:
    audit.append(dict(**c,image_positive=int(ni),text_mention_positive=int(nt),previously_in_42=bool(eligible)))
  assert all(r['image_positive']>0 and r['text_mention_positive']>0 for r in audit),audit
  save(dest/'category-audit.json',audit)
  for name in NAMES:
   p=Path(next(z['path'] for z in cfg['models'] if z['key']==name)) if name in ['cca_256','sparse_cca_16'] else OLD/cond/(name+'.npz')
   with np.load(p) as z: weights={s:z[s].copy() for s in ['image','text']}
   with np.load(OLD/cond/(name+'-test-auc.npz')) as z: arr={k:z[k].copy() for k in z.files}
   arr['eligible']=fixed
   matching=prior.old.evaluation(arr,{**cfg,'n_null':0},signed=False)
   assert matching['n_eligible_categories']==50
   scores={s:np.asarray(data['raw'][s]@(weights[s]/data['scales'][s][:,None])) for s in weights}
   is_positive=name not in ['cca_256','sparse_cca_16']
   if is_positive:
    assert all(w.min()>=0 for w in weights.values());assert all(s.min()>=-1e-9 for s in scores.values())
   common=dict(condition=cond,method=name,matching50=matching,nonnegative_weights=is_positive,score='raw activation divided by FIT std, then weighted sum; no intercept',source=str(p),matching_note='Same 50 historical image-eligible categories; own-caption mentions. Eight have fewer than 50 positive text examples. AUROC representative index invariant to constant centering; reuse saved AUROCs.')
   for centered in [False,True]:
    key='centered' if centered else 'uncentered';target=dest/(name+'-'+key+'.json')
    if target.exists():continue
    ss={s:v-data['means'][s]@(weights[s]/data['scales'][s][:,None]) if centered else v for s,v in scores.items()}
    ret=paired_retrieval(ss['image'],ss['text'],parents,device='cuda',chunk_size=128)
    result=dict(**common,centered_for_search=centered,retrieval={d:{k:v for k,v in z.items() if k!='hits'} for d,z in ret.items()})
    save(target,result)
    print(json.dumps(dict(condition=cond,method=name,centering=key,match50=matching['summary']['agree_at1_count'],recall={d:z['recall'] for d,z in ret.items()})),flush=True)
   del scores
  save(dest/'complete.json',dict(completed=True))
 save(OUT/'complete.json',dict(completed=True,time=time.time()))
if __name__=='__main__':
 try:main()
 except Exception as e:save(OUT/'error.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
