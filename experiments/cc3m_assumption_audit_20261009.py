"""Frozen-model CC3M internal holdout audit. No fitting or model selection on audit rows."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import numpy as np,json,time,yaml
from scipy import sparse,stats,linalg
ROOT=Path('/mnt/working/mm-sae');OUT=ROOT/'runs/cc3m-assumption-audit-2026-10-09';OUT.mkdir(exist_ok=True)
START=time.time()
def save(name,x):
 (OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda z:z.tolist() if isinstance(z,np.ndarray) else z.item()))
def status(**kw):
 d=dict(seconds=time.time()-START,**kw);save('progress.json',d);print(json.dumps(d),flush=True)
def summary(x):
 x=np.asarray(x);return dict(mean=float(x.mean()),median=float(np.median(x)),p10=float(np.quantile(x,.1)),p90=float(np.quantile(x,.9)))
def standard(x):
 x=x-x.mean(0);return x/np.maximum(np.sqrt(np.mean(x*x,0)),1e-12)
def support(x,mass=.9):
 e=np.sort(x*x,axis=1)[:,::-1];return (e.cumsum(1)<mass*e.sum(1,keepdims=True)).sum(1)+1
rng=np.random.default_rng(20261010)
rank=256
pairs=np.array([(i,j) for i in range(rank) for j in range(i+1,rank)])
pairs=pairs[rng.choice(len(pairs),512,replace=False)]

def dependence(x):
 x=x[:4096];outputs={}
 for name,z in [('linear',x),('squared',x*x),('rank',stats.rankdata(x,axis=0))]:
  z=standard(z);a=z[:,pairs[:,0]];b=z[:,pairs[:,1]];observed=(a*b).mean(0)
  null=[];prng=np.random.default_rng(9981)
  for _ in range(19):null.append((a*b[prng.permutation(len(b))]).mean(0))
  null=np.abs(np.concatenate(null));q=float(np.quantile(null,.95));outputs[name]=dict(abs_correlation=summary(np.abs(observed)),shuffled_abs_p95=q,fraction_exceeding_shuffled_p95=float((np.abs(observed)>q).mean()))
 return outputs

def describe(x):
 z=standard(x);m4=np.mean(z**4,0);tail=np.sort(z**4,axis=0)[-max(1,len(z)//100):].sum(0)/np.maximum((z**4).sum(0),1e-12)
 clipped=np.clip(z,np.quantile(z,.005,axis=0),np.quantile(z,.995,axis=0));clipped=standard(clipped)
 nr=np.sqrt((x*x).sum(1));l1=np.abs(x).sum(1)/np.maximum(nr,1e-12)
 blocks=[]
 for ix in np.array_split(np.arange(len(x)),8):blocks.append(float(np.median(np.mean(standard(x[ix])**4,0)-3)))
 return dict(l1_l2=summary(l1),effective_support=summary(l1*l1),support90=summary(support(x)),support95=summary(support(x,.95)),excess_kurtosis=summary(m4-3),fraction_positive_excess=float((m4>3).mean()),absolute_skew=summary(np.abs(np.mean(z**3,0))),fourth_moment_top1pct_share=summary(tail),winsorized_excess=summary(np.mean(clipped**4,0)-3),block_median_excess=blocks,logcosh_contrast=summary((np.mean(np.logaddexp(z,-z)-np.log(2),0)-.37456720749)**2),dependence=dependence(z))

def shared(x,y):
 a,b=standard(x),standard(y)
 def cum(aa,bb):return (aa*aa*bb*bb).mean(0)-1-2*(aa*bb).mean(0)**2
 true=cum(a,b);prng=np.random.default_rng(913);null=np.array([cum(a,b[prng.permutation(len(b))]) for _ in range(19)])
 # winsorization is diagnostic only, never used for model fitting or search.
 wa=standard(np.clip(a,np.quantile(a,.005,axis=0),np.quantile(a,.995,axis=0)));wb=standard(np.clip(b,np.quantile(b,.005,axis=0),np.quantile(b,.995,axis=0)))
 return dict(cokurtosis=summary(true),shuffled_cokurtosis=summary(null.ravel()),fraction_above_coordinate_shuffled_p95=float((true>np.quantile(null,.95,axis=0)).mean()),winsorized_cokurtosis=summary(cum(wa,wb)),paired_correlation=summary((a*b).mean(0)))

def main():
 cfg=yaml.safe_load((ROOT/'runs/cc3m-followup-2026-10-05/configs/cc3m-fit-agreement.yaml').read_text());source=ROOT/'runs/cc3m-sae-2026-10-04';parent=Path(cfg['parent_run']);rotroot=ROOT/'runs/rotation-convergence-2026-10-09'
 with np.load(next(m['path'] for m in cfg['models'] if m['key']=='cca_256')) as z:weights={s:z[s].astype(float) for s in ['image','text']}
 assert weights['image'].shape[1]==rank
 cache=OUT/'audit-input.npz'
 if not cache.exists():
  status(stage='select_holdout')
  pop=json.loads((parent/'population.json').read_text());names=np.array([r['image_id'] for r in json.loads((source/'index/train2017/images.json').read_text())]);candidates=np.flatnonzero(np.isin(names,pop['tune_image_ids']));excluded=[]
  for path in [rotroot/'split.npz',ROOT/'runs/cca-unsupervised-rotations-2026-10-09/cc3m-cc3m/split.npz']:
   if path.exists():
    with np.load(path) as z:excluded.extend(z['tune_rows'].tolist())
  # Exclude additional known CC3M experiment tune subsets as a conservative safeguard.
  for path in (ROOT/'runs').glob('*/**/cc3m-cc3m/**/split.npz'):
   try:
    with np.load(path) as z:excluded.extend(z['tune_rows'].tolist())
   except (KeyError,ValueError):pass
  candidates=np.setdiff1d(candidates,np.unique(excluded));rows=rng.choice(candidates,16384,replace=False);assert not np.isin(names[rows],pop['fit_image_ids']).any();assert not np.isin(rows,excluded).any()
  parents=np.load(source/'index/train2017/parents.npy');assert len(parents)==len(names) and np.array_equal(parents,np.arange(len(names)))
  np.savez(OUT/'audit-split.npz',image_rows=rows,text_rows=rows,image_ids=names[rows],excluded_tune_rows=np.unique(excluded))
  with np.load(parent/'moments.npz') as z:
   ids={s:z[s+'_ids'] for s in weights};ni=len(ids['image']);mu=z['fit_mean'];cov=z['fit_second']-np.outer(mu,mu);sd=np.sqrt(np.maximum(cov.diagonal(),0));sd[sd==0]=1;cov/=sd[:,None]*sd[None,:]
  rawstats={};saved={}
  for s,sl in [('image',slice(0,ni)),('text',slice(ni,None))]:
   status(stage='load_activation',modality=s)
   whole=sparse.load_npz(source/'activations/train2017'/f'{s}.npz').tocsr();raw=whole[rows];del whole
   vals=raw.data;vals=vals[vals>0];rrng=np.random.default_rng(852);vals=rrng.choice(vals,min(150000,len(vals)),replace=False)
   half=len(vals)//2;mean=vals[:half].mean();ks=stats.kstest(vals[half:],'expon',args=(0,mean));cvv=float(vals.std()/vals.mean())
   shape,loc,scale=stats.gamma.fit(vals[:half],floc=0)
   rawstats[s]=dict(shape=raw.shape,nonzeros_per_row=summary(np.diff(raw.indptr)),zero_fraction=float(1-raw.count_nonzero()/np.prod(raw.shape)),positive_cv=cvv,positive_exponential_ks=float(ks.statistic),positive_exp_test_loglik=float(stats.expon.logpdf(vals[half:],scale=mean).mean()),positive_gamma_shape=float(shape),positive_gamma_test_loglik=float(stats.gamma.logpdf(vals[half:],shape,scale=scale).mean()),warning='Positive values pooled across coordinates. Distribution heterogeneity and TopK selection preclude a coordinatewise generative conclusion.')
   raw=raw[:,ids[s]].multiply(1/sd[sl]).tocsr();u=np.asarray(raw@weights[s])- (mu[sl]/sd[sl])@weights[s]
   val_load=np.asarray(raw.T@(u-u.mean(0)))/len(rows);fit_load=cov[sl,sl]@weights[s]
   saved[s]=u;saved[s+'_fit_load']=fit_load;saved[s+'_val_load']=val_load
   sparse.save_npz(OUT/(s+'-audit-standardized-raw.npz'),raw)
  np.savez(cache,**saved);save('raw-activation.json',rawstats)
  save('population.json',dict(dataset='CC3M train internal mapping holdout, NOT official CC3M validation',samples=16384,seed=20261010,cca_fit_overlap=0,known_rotation_tune_overlap=0,excluded_known_tune_rows=len(np.unique(excluded)),sae_exposure='SAE was trained on CC3M training source, including these internal holdout samples. This is held out from mapping fit and known rotation tuning, not from SAE.',official_validation_cache='No official validation shard in existing cache manifest; val2017 activations are COCO.'))
 with np.load(cache) as z:data={k:z[k] for k in z.files}
 rotations=[('cca_original',np.eye(rank))]
 for seed in range(3):
  rr=np.random.default_rng(471+seed).normal(size=(rank,rank));q,r=linalg.qr(rr);q*=np.sign(r.diagonal());rotations.append((f'random-{seed}',q))
 for path in sorted(rotroot.glob('*-final.npz')):rotations.append((path.stem[:-6],np.load(path)['rotation']))
 for name,R in rotations:
  if (OUT/(name+'.json')).exists():continue
  status(stage='diagnose',name=name)
  result=dict(name=name,modalities={});uv=[]
  for s in ['image','text']:
   x=data[s]@R;uv.append(x);d=describe(x)
   fit=data[s+'_fit_load']@R;val=data[s+'_val_load']@R
   # Per-column L2 normalized comparison of loading directions; support is squared covariance, not attribution.
   denom=np.linalg.norm(fit,axis=0)*np.linalg.norm(val,axis=0)
   d['loading_validation']=dict(column_cosine=summary((fit*val).sum(0)/np.maximum(denom,1e-12)),fit_support95=summary(support(fit.T,.95)),val_support95=summary(support(val.T,.95)))
   result['modalities'][s]=d
  result['shared']=shared(*uv);save(name+'.json',result)
 # Covariance-matched Gaussian controls are a diagnostic distribution, not semantic data.
 if not (OUT/'gaussian-control.json').exists():
  g={}
  for s in ['image','text']:
   x=data[s];rrng=np.random.default_rng(77);cov=np.cov(x,rowvar=False,bias=True);ga=rrng.normal(size=x.shape)@linalg.cholesky(cov+1e-12*np.eye(rank),lower=True).T+x.mean(0)
   g[s]=describe(ga)
  save('gaussian-control.json',g)
 save('protocol.json',dict(frozen_models=True,rank=rank,rotations=len(rotations),pair_count=512,dependence_rows=4096,shuffle_repeats=19,permutation_tail='Descriptive shuffled 95th-percentile exceedance; NOT a multiple-testing-corrected independence test.',scope='Marginal distributions and selected dependence diagnostics cannot prove latent semantic recovery or full independence. No objectives refit and no best seed selected on this holdout.',tail_sensitivity='Clip each coordinate to its own validation .5/99.5 percentiles, recenter and rescale for descriptive sensitivity only.',normalization='Frozen CCA input training means/scales. Marginal moment and dependence diagnostics use within-audit centering/scaling and do not alter retrieval.'))
 status(stage='complete',models=len(rotations));save('complete.json',dict(seconds=time.time()-START))
if __name__=='__main__':main()
