"""Frozen rank models; label-based candidate restriction is diagnostic only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json,time
import numpy as np
from experiments import apbt_search_20261009 as p
from mm_sae.analysis.semantic_evaluation import auc_matrix
ROOT=p.ROOT;OUT=ROOT/'runs/rank-candidate-masking-2026-10-10';OUT.mkdir(exist_ok=True)
cfg=p.old.load_config(ROOT/p.CONFIG['cc3m-cc3m']);train,_=p.eval_population(cfg,'tune',OUT)
ref=ROOT/'runs/rank-concept-redundancy-2026-10-10'
orig=json.loads((ref/'512-0.json').read_text());cats=sorted({r['category_index'] for r in orig['records']});assert len(cats)==50
p.save(OUT/'candidate-protocol.json',dict(selection='5000 COCO train2017 heldout-tune images, disjoint from COCO val2017. Both modalities use own labels. Frozen CC3M SAE / CC3M mapping.',evaluation='Same original image-A/text-B 50 categories and repeat image-B/text-A; frozen rank and rotations. No coordinate rematching.',rules='Union of image and text top q AUROC coordinates on train-tune, q=1,2,4,8,16. Same candidate IDs retained on both modalities. Full and random same-size candidates are controls. No claim of improved label-free matching.',selection_metadata=train['metadata'],warning='More agreement can be forced by fewer candidates; also record heldout AUROC and both>=.8. Random same-size controls are averaged over100 repeats.'))
start=time.time();rows=[]
for rank in [128,256,512]:
 for seed in range(3):
  if rank==128:path=ROOT/f'runs/matched-statistic-rotation128-2026-10-10/pls-128/{seed}-transforms.npz'
  elif rank==256:path=ROOT/f'runs/pls-cca-gap-2026-10-10/pls-covariance-{seed}-transforms.npz'
  else:path=ROOT/f'runs/matched-statistic-rotation-2026-10-10/pls-512/{seed}-transforms.npz'
  with np.load(path) as z:w={s:z[s].copy() for s in ['image','text']}
  ac={}
  for s in w:
   x=np.asarray(train['raw'][s]@(w[s]/train['scales'][s][:,None]))-train['means'][s]@(w[s]/train['scales'][s][:,None])
   ac[s]=auc_matrix(x,train['labels'][s],chunk_size=32)
  np.savez_compressed(OUT/f'select-aucs-{rank}-{seed}.npz',**ac)
  with np.load(ref/f'{rank}-{seed}-aucs.npz') as z:ev={k:z[k].copy() for k in z.files}
  for c in cats:
   orders={s:np.argsort(-ac[s][:,c],kind='stable') for s in w}
   for fold,ih,th in [('original','a','b'),('reversed','b','a')]:
    ai=ev['image_'+ih][:,c];at=ev['text_'+th][:,c];fulli=int(ai.argmax());fullt=int(at.argmax())
    def measure(ids):
     ii=int(ids[np.argmax(ai[ids])]);tt=int(ids[np.argmax(at[ids])]);agree=ii==tt
     return dict(match=int(agree),match_both_auc08=int(agree and ai[ii]>=.8 and at[tt]>=.8),image_rep=ii,text_rep=tt,image_auc=float(ai[ii]),text_auc=float(at[tt]),candidate_count=len(ids),original_image_kept=int(fulli in ids),original_text_kept=int(fullt in ids))
    for q in [0,1,2,4,8,16]:
     ids=np.arange(rank) if q==0 else np.union1d(orders['image'][:q],orders['text'][:q])
     row=dict(rank=rank,seed=seed,category_index=c,concept=train['concepts'][c],fold=fold,q=q,**measure(ids))
     if q:
      rng=np.random.default_rng(12910+rank*100+seed*10000+c*71+q)
      controls=[measure(rng.choice(rank,len(ids),replace=False)) for _ in range(100)]
      row['random']={k:float(np.mean([v[k] for v in controls])) for k in ['match','match_both_auc08','image_auc','text_auc']}
     rows.append(row)
  p.save(OUT/'candidate-records.json',rows);print(rank,seed,'complete',time.time()-start,flush=True)
p.save(OUT/'candidate-complete.json',dict(seconds=time.time()-start))
