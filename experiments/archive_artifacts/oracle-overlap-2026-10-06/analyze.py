import json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2];out=Path(__file__).parent
all_results={}
for condition,cca in [('coco-coco',root/'mm-sae/runs/mapping-semantics-2026-10-04/sparse-fit/sparse_cca_k16.npz'),('cc3m-coco',out/'cc3m-sparse-cca16.npz')]:
 z=np.load(cca); overlaps={};weights={}
 for side in ['image','text']:
  p=root/f'artifacts/oracle-pursuit-2026-10-06/warm-start-backup/results/warmstart-{condition}/mse_l2_0.01_{side}.npz'
  q=np.load(p);np.testing.assert_array_equal(q['feature_ids'],z[side+'_ids'])
  oracle=q['raw'];assert np.all(np.count_nonzero(oracle,axis=0)==16)
  assert np.all(np.count_nonzero(z[side],axis=0)==16)
  overlaps[side]=(oracle!=0).astype(np.int16).T@(z[side]!=0).astype(np.int16)
 for n in [171,256]:
  oi,ot=[overlaps[s][:,:n] for s in ['image','text']]
  joint=oi+ot;best=joint.argmax(1);row=np.arange(171)
  im=oi.max(1);tm=ot.max(1)
  tied=((oi==im[:,None])&(ot==tm[:,None])).any(1)
  def stats(x):return dict(mean=float(x.mean()),median=float(np.median(x)),min=int(x.min()),max=int(x.max()),q25=float(np.quantile(x,.25)),q75=float(np.quantile(x,.75)))
  result=dict(image_best=stats(im),text_best=stats(tm),joint_image=stats(oi[row,best]),joint_text=stats(ot[row,best]),joint_total=stats(joint[row,best]),independent_best_share_any_component=int(tied.sum()),exact_image=int((im==16).sum()),exact_text=int((tm==16).sum()),exact_pair=int((joint.max(1)==32).sum()),per_concept=[dict(index=int(c),best_joint_component=int(best[c]),image_overlap=int(oi[c,best[c]]),text_overlap=int(ot[c,best[c]]),image_best=int(im[c]),text_best=int(tm[c])) for c in row])
  all_results[f'{condition}_{n}']=result
  print(condition,n,{k:v for k,v in result.items() if k!='per_concept'})
(out/'overlap.json').write_text(json.dumps(all_results,indent=2))
