from pathlib import Path
import numpy as np,json
from scipy.optimize import linear_sum_assignment
root=Path(__file__).resolve().parent
folders=sorted((root/'results').glob('seed*-rho*'))
assert len(folders)==12
count=0
for p in folders:
 assert (p/'complete.json').exists()
 g=np.load(p/'generator.npz')
 for f in p.glob('*-evaluation.json'):
  e=json.loads(f.read_text());w=np.load(p/(f.name.replace('-evaluation.json','.npz')))
  responses=[]
  for side in ['image','text']:
   a=w[side];assert np.isfinite(a).all() and (np.count_nonzero(a,axis=0)<=16).all()
   if f.name.startswith('nce_nonnegative'):assert (a>=0).all()
   d=g[side].astype(float);s=g['scale_'+side].astype(float)
   response=np.einsum('ij,jk->ik',d/s,a)
   np.testing.assert_allclose(response,e['selectivity']['response_'+side],atol=2e-6,rtol=2e-5)
   x=np.ones(128);before=np.einsum('j,jk->k',x/s,a);after=np.einsum('j,jk->k',(x+d[0])/s,a)
   np.testing.assert_allclose(after-before,response[0],atol=1e-10)
   responses.append(response)
  ri,rt=responses
  score=np.minimum(ri**2/np.maximum((ri**2).sum(0),1e-15),rt**2/np.maximum((rt**2).sum(0),1e-15))*(ri*rt>0)
  r,c=linear_sum_assignment(-score[:2]);assert abs(score[r,c].mean()-e['selectivity']['target_mean'])<1e-5, (str(f),score[r,c].mean(),e['selectivity']['target_mean'])
  if f.name.startswith('known_anchor'):assert abs(e['selectivity']['target_mean']-1)<1e-6
  for mode in e['retrieval'].values():
   for direction in ['image_to_text','text_to_image']:
    v=mode[direction];ranks=np.array(v['ranks'])
    for k in [1,5,10]:assert abs(np.mean(ranks<=k)-v['recall'][str(k)])<1e-10
  count+=1
out=dict(conditions=len(folders),evaluations=count,verified=['completion','finite weights and support <=16','nonnegative coefficients','analytic response against finite difference','selectivity recomputation','Recall from saved ranks','known generator reference selectivity=1'])
(root/'verification.json').write_text(json.dumps(out,indent=2));print(out)
