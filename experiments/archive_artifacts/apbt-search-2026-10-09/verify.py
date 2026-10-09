from pathlib import Path
import json,numpy as np,hashlib
R=Path(__file__).resolve().parent;D=R/'results';model_checks=[];evals=[]
for p in D.glob('*/*/seed*/*.npz'):
 if p.stem=='split':
  with np.load(p) as z:assert not np.intersect1d(z['fit_rows'],z['tune_rows']).size
  continue
 if p.stem.endswith('-auc'):continue
 with np.load(p) as z:
  assert set(['image','text','permutation']).issubset(z.files)
  k=z['image'].shape[1];assert k==256
  assert sorted(z['permutation'].tolist())==list(range(k))
  for s in ['image','text']:
   w=z[s];assert np.isfinite(w).all();assert (w!=0).sum(0).max()<=16
   np.testing.assert_allclose(np.linalg.norm(w,axis=0),1,atol=3e-6)
   if p.stem!='signed_pair':assert w.min()>=0
  # Stored original columns with explicit P must agree with reordered-coordinate scores.
  rng=np.random.default_rng(9);x=rng.normal(size=(2,z['image'].shape[0]));y=rng.normal(size=(2,z['text'].shape[0]));P=np.zeros((k,k));P[np.arange(k),z['permutation']]=1
  # Explicit contractions avoid host BLAS floating-point status warnings.
  xa=np.einsum('ni,ik->nk',x,z['image']);yb=np.einsum('ni,ik->nk',y,z['text'])
  lhs=np.einsum('nk,kl,ml->nm',xa,P,yb)
  rhs=np.einsum('nk,mk->nm',xa,yb[:,z['permutation']])
  assert np.isfinite(lhs).all() and np.isfinite(rhs).all()
  np.testing.assert_allclose(lhs,rhs,atol=1e-10,equal_nan=False)
 model_checks.append(str(p.relative_to(D)))
for p in D.glob('*/*/seed*/*-test.json'):
 v=json.loads(p.read_text());assert v['denominator']==42
 assert v['retrieval']['image_to_text']['query_count']==5000
 assert v['retrieval']['text_to_image']['query_count']==25014
 for metric in ['positive','signed']:
  rows=[x for x in v[metric]['per_category'] if x.get('status')=='ok']
  assert sum(r['agree_at1'] for r in rows)==v[metric]['summary']['agree_at1_count']
 evals.append(str(p.relative_to(D)))
assert len(evals)==15
v=dict(models=len(model_checks),test_evaluations=len(evals),models_verified=model_checks,test_evaluations_verified=evals)
(R/'verification.json').write_text(json.dumps(v,indent=2));print(v['models'],'models;',v['test_evaluations'],'test evaluations verified')
