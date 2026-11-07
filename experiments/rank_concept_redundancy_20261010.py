"""Frozen PLS covariance rotations: cross-fit AUROC multiplicity and response overlap."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json,time
import numpy as np
from experiments import apbt_search_20261009 as p
from mm_sae.analysis.semantic_evaluation import auc_matrix
ROOT=p.ROOT;OUT=ROOT/'runs/rank-concept-redundancy-2026-10-10';OUT.mkdir(exist_ok=True)
cfg=p.old.load_config(ROOT/p.CONFIG['cc3m-cc3m']);data,parents=p.eval_population(cfg,'test',OUT);sides=['image','text'];start=time.time()
eligible=np.flatnonzero(np.isin([c['id'] for c in data['concepts']],data['metadata']['eligible_category_ids']));assert len(eligible)==50
p.save(OUT/'protocol.json',dict(condition='CC3M SAE / CC3M mapping',method='PLS-SVD + covariance-loading common rotation',ranks=[128,256,512],seeds=[0,1,2],thresholds=[.7,.8,.9],selection='Choose high-AUROC coordinates on half A, measure AUROC and response overlap only on half B; repeat B to A. Positive polarity only, fixed50 categories. Validated count requires threshold on BOTH halves.',overlap='Coordinates selected with AUROC>=.8 in source half. On independent target half measure mean pairwise Pearson correlation on all samples and within category-positive samples, plus Jaccard of top10% positive samples per coordinate. Require >=20 category-positive samples for overlap. At least2 selected coordinates. No outcome-based pair selection.',matching='Reuse exactly existing per-category cross-half top1/5/10 ranks, both directions. Also compute representative stability within modality across halves. Repeated seeds and folds are not independent dataset replicates.',limits='Rank changes the entire fitted rotation. Counts are raw counts plus fractions of available coordinates. Co-activation does not prove identical semantics. No retraining or new objective. PLS512 previous6000-step fit was budget-limited.'))
for rank in [128,256,512]:
 for seed in range(3):
  dest=OUT/f'{rank}-{seed}.json'
  if dest.exists():continue
  if rank==128:
   folder=ROOT/'runs/matched-statistic-rotation128-2026-10-10/pls-128';weights=folder/f'{seed}-transforms.npz';oldfile=folder/f'{seed}-result.json'
  elif rank==256:
   weights=ROOT/f'runs/pls-cca-gap-2026-10-10/pls-covariance-{seed}-transforms.npz';oldfile=ROOT/f'runs/matched-statistic-rotation-2026-10-10/pls-256/{seed}-result.json'
  else:
   folder=ROOT/'runs/matched-statistic-rotation-2026-10-10/pls-512';weights=folder/f'{seed}-transforms.npz';oldfile=folder/f'{seed}-result.json'
  with np.load(weights) as z:w={s:z[s].astype(float) for s in sides}
  old=json.loads(oldfile.read_text());scores={s:np.asarray(data['raw'][s]@(w[s]/data['scales'][s][:,None]))-data['means'][s]@(w[s]/data['scales'][s][:,None]) for s in sides}
  auc={}
  for s in sides:
   for half in ['a','b']:
    mask=data['masks'][s+'_'+half];auc[s,half]=auc_matrix(scores[s][mask],data['labels'][s][mask],chunk_size=32)
  np.savez_compressed(OUT/f'{rank}-{seed}-aucs.npz',**{s+'_'+h:a for (s,h),a in auc.items()})
  records=[]
  for s in sides:
   for select,test in [('a','b'),('b','a')]:
    x=scores[s][data['masks'][s+'_'+test]];labels=data['labels'][s][data['masks'][s+'_'+test]]
    for c in eligible:
     a=auc[s,select][:,c];b=auc[s,test][:,c];order=np.argsort(-a,kind='stable');pick=int(order[0]);targetpick=int(np.argsort(-b,kind='stable')[0]);chosen=np.flatnonzero(a>=.8)
     rec=dict(rank=rank,seed=seed,side=s,selection_half=select,category_index=int(c),concept=data['concepts'][c],selected_coordinate=pick,heldout_selected_auc=float(b[pick]),selection_best_auc=float(a[pick]),selection_margin=float(a[order[0]]-a[order[1]]),same_representative=pick==targetpick,heldout_rank_selected=int(np.flatnonzero(np.argsort(-b,kind='stable')==pick)[0]+1),counts={str(t):dict(selected=int((a>=t).sum()),confirmed=int(((a>=t)&(b>=t)).sum())) for t in [.7,.8,.9]})
     positives=labels[:,c].astype(bool);n=int(positives.sum());rec['positive_count']=n;rec['overlap']=None
     if len(chosen)>=2 and n>=20:
      def normcorr(z):
       z=z-z.mean(0);den=np.sqrt((z*z).sum(0));valid=den>1e-12
       if valid.sum()<2:return None
       z=z[:,valid]/den[valid];cc=z.T@z;return float(cc[np.triu_indices(len(cc),1)].mean())
      pos=x[positives][:,chosen];take=max(1,int(np.ceil(.1*n)));top=np.zeros(pos.shape,dtype=float)
      ix=np.argsort(-pos,axis=0,kind='stable')[:take];top[ix,np.arange(len(chosen))[None,:]]=1
      intersection=top.T@top;jac=intersection/(2*take-intersection);tri=np.triu_indices(len(chosen),1)
      rec['overlap']=dict(coordinates=chosen.tolist(),all_sample_correlation=normcorr(x[:,chosen]),within_positive_correlation=normcorr(pos),positive_top10_jaccard=float(jac[tri].mean()),positive_top10_count=take)
      if seed==0 and select=='a':
       coords=[int(k) for k in order if a[k]>=.8][:2];examples=[]
       for coord in coords:
        local=np.argsort(-x[positives,coord],kind='stable')[:take];context=labels[positives][local].mean(0)-labels[positives].mean(0);context[c]=-np.inf
        context_ids=np.argsort(-context)[:3]
        examples.append(dict(coordinate=coord,selection_auc=float(a[coord]),heldout_auc=float(b[coord]),heldout_positive_row_indices=np.flatnonzero(data['masks'][s+'_'+test])[np.flatnonzero(positives)[local[:5]]].tolist(),other_category_enrichment=[dict(concept=data['concepts'][int(j)],difference=float(context[j])) for j in context_ids]))
       rec['examples']=examples
     records.append(rec)
  # Audit existing cross-half top1 against recomputed AUROCs.
  for label,ih,th in [('matching50','a','b'),('repeat_matching50','b','a')]:
   computed=sum(int(np.argmax(auc['image',ih][:,c])==np.argmax(auc['text',th][:,c])) for c in eligible)
   assert computed==old[label]['summary']['agree_at1_count'],(rank,seed,label,computed)
  p.save(dest,dict(rank=rank,seed=seed,records=records,matching=old['matching50'],repeat_matching=old['repeat_matching50']))
  p.save(OUT/'progress.json',dict(rank=rank,seed=seed,seconds=time.time()-start));print(rank,seed,'done',flush=True)
p.save(OUT/'complete.json',dict(complete=True,seconds=time.time()-start))
