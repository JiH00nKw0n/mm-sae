import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import numpy as np,torch,json,time,traceback
from pathlib import Path
from experiments.axis_overlap_audit_20261011 import Evaluator,ROOT,SRC,OUT,a,p,S,save,prune
try:
 e=Evaluator();base=e.matrices(ROOT/'runs/mapping-ablation-cc3m-2026-10-04/transforms/cross_svd_3221.npz',256)
 with np.load(Path(e.cfg['parent_run'])/'moments.npz') as zz:ids={s:zz[s+'_ids'] for s in S}
 rawauc={};held={}
 for j,s in enumerate(S):
  sel=e.splits[0][j];allix=torch.arange(len(e.x[s]),device='cuda');held[s]=allix[~torch.isin(allix,sel)];rawauc[s]=a.gpu_auc(e.x[s][sel],e.labels[s][sel])
 for method in ['covariance-svd','correlation-cayley']:
  data=json.loads((OUT/(method+'.json')).read_text());r=a.tensor(np.load(SRC/f'rank256/{method}-0-oriented.npz')['rotation']);w={s:base[s]@r for s in S}
  for s in S:
   x=e.x[s][held[s]];pw,_=prune(w[s],.9)
   for c,cat in enumerate(data[s]['categories']):
    axes=[q['axis'] for q in cat['axes']]
    if not axes:continue
    cols=torch.tensor(axes,device='cuda');t=e.labels[s][held[s],c:c+1];positive=t[:,0]>0;delta=x[positive].mean(0)-x[~positive].mean(0)
    good=rawauc[s][:,c]>=.8;ww=w[s][:,cols];removed=ww.clone();removed[good]=0
    removedauc=a.gpu_auc(x@removed,t)[:,0].cpu().numpy();prunedauc=a.gpu_auc(x@pw[:,cols],t)[:,0].cpu().numpy()
    contrib=(delta[:,None]*ww).cpu().numpy()
    for q,j in zip(cat['axes'],range(len(axes))):
     order=np.argsort(-contrib[:,j])[:5];q['positive_class_mean_contributions']=[dict(sae_id=int(ids[s][f]),mean_difference_contribution=float(contrib[f,j]),coefficient=float(ww[f,j]),selection_raw_auc=float(rawauc[s][f,c]) if torch.isfinite(rawauc[s][f,c]) else None) for f in order]
     q['heldout_auc_after_remove_raw_auc080']=float(removedauc[j]) if np.isfinite(removedauc[j]) else None;q['heldout_auc_after_energy90_prune']=float(prunedauc[j]) if np.isfinite(prunedauc[j]) else None;q['removed_raw_features']=int(good.sum())
  save(OUT/(method+'.json'),data)
 save(OUT/'contribution-complete.json',dict(time=time.time()))
except Exception as ex:save(OUT/'error-contribution.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
