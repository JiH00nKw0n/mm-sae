from pathlib import Path
import numpy as np,json,hashlib
R=Path(__file__).resolve().parent;D=R/'results';checks=[]
for cond in ['coco-coco','cc3m-coco','cc3m-cc3m']:
 with np.load(D/cond/'split.npz') as z:assert not np.intersect1d(z['fit_rows'],z['tune_rows']).size
 for key in ['original_sparse_pls','pls_hsic_0','pls_hsic_0.1','pls_hsic_1']:
  v=json.loads((D/cond/(key+'-evaluation.json')).read_text());assert v['denominator']==42
  if key=='original_sparse_pls':
   old=json.loads((R.parent/'caption-matching-2026-10-09/results'/cond/'results/test__pls_16.json').read_text())
   assert v['signed_agreement']['summary']['agree_at1_count']==old['summary']['agree_at1_count']
   continue
  with np.load(D/cond/(key+'.npz')) as z:
   for side in ['image','text']:
    w=z[side];assert np.isfinite(w).all();assert w.shape[1]==256
    assert (w!=0).sum(0).max()<=16
    np.testing.assert_allclose(np.linalg.norm(w,axis=0),1,atol=2e-6)
  checks.append(cond+'/'+key)
v={'trained_models_verified':len(checks),'checks':checks,'evaluations':len(list(D.glob('*/*-evaluation.json'))),'sha256':{str(p.relative_to(D)):hashlib.sha256(p.read_bytes()).hexdigest() for p in D.rglob('*') if p.is_file()}}
assert v['evaluations']==12
(R/'verification.json').write_text(json.dumps(v,indent=2));print('Verified',len(checks),'models and 12 evaluations')
