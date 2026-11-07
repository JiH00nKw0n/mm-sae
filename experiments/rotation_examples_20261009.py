import os
os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import numpy as np,json
import experiments.apbt_search_20261009 as p
ROOT=p.ROOT;OUT=ROOT/'runs/rotation-mechanisms-2026-10-09'
for cond in ['cc3m-cc3m','cc3m-coco']:
 cfg=p.old.load_config(ROOT/p.CONFIG[cond]);out=OUT/cond;data,_=p.eval_population(cfg,'test',out)
 z=np.load(next(m['path'] for m in cfg['models'] if m['key']=='cca_256'));w={s:z[s] for s in ['image','text']}
 names=json.loads((ROOT/'runs/cca-171-category-names-2026-10-09/protocol.json').read_text())['prototype_texts'];examples={};arrays={}
 methods=['cca_original','coefficient_l1-0','loading_varimax-0','output_l1-0','output_kurtosis-0','output_logcosh-0']+(['oracle_ridge'] if cond=='cc3m-coco' else [])
 with np.load(Path(cfg['parent_run'])/'moments.npz') as z:ids={s:z[s+'_ids'] for s in w}
 for method in methods:
  f=out/(method+'.json')
  if not f.exists():continue
  diag=json.loads(f.read_text());examples[method]={};r=None
  if method!='oracle_ridge':r=np.load(ROOT/'runs/cca-unsupervised-rotations-2026-10-09'/cond/(method+'-evaluated-rotation.npz'))['rotation']
  for s in w:
   if r is not None:ww=w[s]@r
   else:
    key='image' if s=='image' else 'propagated_presence';z=np.load(ROOT/'runs/cc3m-followup-2026-10-05/coco-fit/annotation-sets/fits'/f'{key}_16.npz');ww=z['coefficients']/z['feature_scale'][:,None]*data['scales'][s][:,None]
   examples[method][s]={}
   for cname in ['car','dog','person','road','sky-other','tree']:
    if cname not in names:continue
    c=names.index(cname);k=diag['categories'][c][s+'_coordinate'] if method!='oracle_ridge' else c;col=ww[:,k];top=np.argsort(abs(col))[-32:][::-1]
    examples[method][s][cname]=dict(coordinate=k,top_activation_ids=ids[s][top].tolist(),top_coefficients=col[top].tolist(),coefficient_norm=float(np.linalg.norm(col)))
    arrays[f'{method}__{s}__{cname}']=col.astype('float32')
   arrays['ids__'+s]=ids[s]
 (out/'examples.json').write_text(json.dumps(examples,indent=2));np.savez_compressed(out/'example-columns.npz',**arrays)
 print(cond,'examples saved',flush=True)
