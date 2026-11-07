import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import numpy as np,json
from scipy import sparse,stats
P=Path('/mnt/working/mm-sae/runs/cc3m-assumption-audit-2026-10-09');result={}
for side in ['image','text']:
 x=sparse.load_npz(P/(side+'-audit-standardized-raw.npz')).tocsc();rows=[]
 for k in range(x.shape[1]):
  a=x.data[x.indptr[k]:x.indptr[k+1]];ix=x.indices[x.indptr[k]:x.indptr[k+1]];train=a[(ix<8192)&(a>0)];test=a[(ix>=8192)&(a>0)]
  if min(len(train),len(test))<40:continue
  mean=train.mean();shape,_,scale=stats.gamma.fit(train,floc=0);gain=(stats.gamma.logpdf(test,shape,scale=scale)-stats.expon.logpdf(test,scale=mean)).mean();ks=stats.kstest(test,'expon',args=(0,mean)).statistic
  rows.append(dict(coordinate=k,fit_n=len(train),test_n=len(test),positive_cv=float(test.std()/test.mean()),gamma_shape=float(shape),gamma_minus_exp_nats=float(gain),exponential_ks=float(ks)))
 def summary(k):
  a=np.array([r[k] for r in rows]);return dict(median=float(np.median(a)),p10=float(np.quantile(a,.1)),p90=float(np.quantile(a,.9)))
 result[side]=dict(eligible_coordinates=len(rows),cv=summary('positive_cv'),gamma_shape=summary('gamma_shape'),gamma_gain=summary('gamma_minus_exp_nats'),ks=summary('exponential_ks'),gamma_better_fraction=float(np.mean([r['gamma_minus_exp_nats']>0 for r in rows])),coordinates=rows)
result['protocol']='Descriptive distribution check within audit population. Fit marginal density on first8192 rows and score on last8192. Requires40 positive values per half. Raw coordinates scaled by positive training SD only, so CV and KS are invariant to that scaling. This does not refit CCA/rotation. Fitted exponential assumes no shift and positive tail not truncated; TopK makes this a restrictive candidate.'
(P/'positive-marginals.json').write_text(json.dumps(result,indent=2))
