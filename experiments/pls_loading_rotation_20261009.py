import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import json,time
from pathlib import Path
import numpy as np
import torch
import experiments.rotation_convergence_20261009 as conv
import experiments.multiview_likelihood_20261009 as multi
p=conv.p;ROOT=p.ROOT;OUT=ROOT/'runs/pls-loading-rotation-2026-10-09';OUT.mkdir(exist_ok=True);conv.OUT=OUT
cfg=p.old.load_config(ROOT/p.CONFIG['cc3m-cc3m']);path=ROOT/'runs/mapping-ablation-cc3m-2026-10-04/transforms/cross_svd_256.npz';z=np.load(path);w={s:z[s].astype(float) for s in ['image','text']}
assert all(a.shape[1]==256 for a in w.values())
cache=OUT/'projected.npz'
if not cache.exists():
 cfg,arr,_=p.prepare('cc3m-cc3m',OUT,32768,0);a={}
 for kind in ['fit','tune']:
  for s in w:a[kind+'_'+s]=((arr[kind,s]-arr['mean',s])@torch.tensor(w[s],device='cuda',dtype=torch.float32)).cpu().numpy()
 np.savez(cache,**a);del arr;torch.cuda.empty_cache()
z=np.load(cache);orientation=[z['fit_'+s] for s in w]
with np.load(Path(cfg['parent_run'])/'moments.npz') as z:
 ni=len(z['image_ids']);mu=z['fit_mean'];cv=z['fit_second']-np.outer(mu,mu);sd=np.sqrt(np.maximum(cv.diagonal(),0));sd[sd==0]=1;cv/=sd[:,None]*sd[None,:]
ts={'load':[]}
for s,c in [('image',cv[:ni,:ni]),('text',cv[ni:,ni:])]:
 ll=c@w[s];ll/=np.maximum(np.linalg.norm(ll,axis=1,keepdims=True),1e-12);ts['load'].append(torch.tensor(ll,device='cuda',dtype=torch.float64))
base_trace=float(np.trace(w['image'].T@cv[:ni,ni:]@w['text']));orth={s:float(np.max(abs(q.T@q-np.eye(256)))) for s,q in w.items()}
for seed in range(3):conv.fit('loading_varimax',seed,ts,type('Args',(),{'max_steps':6000})())
del ts;torch.cuda.empty_cache();data,parents=p.eval_population(cfg,'test',OUT)
base={s:np.asarray(data['raw'][s]@(w[s]/data['scales'][s][:,None]))-data['means'][s]@(w[s]/data['scales'][s][:,None]) for s in w}
jobs=[('pls_original',np.eye(256),False),('pls_oriented',np.eye(256),True)]+[(f'pls_rotated_{seed}',np.load(OUT/f'loading_varimax-{seed}-final.npz')['rotation'],True) for seed in range(3)]
for name,r,orient in jobs:
 dest=OUT/(name+'-result.json')
 if dest.exists():continue
 if orient:
  skew=sum(np.mean((x@r)**3,0) for x in orientation);r=r*np.where(skew<0,-1.,1.)
 scores={s:x@r for s,x in base.items()};result=multi.metrics(scores,data,parents,cfg)
 norm=lambda x:x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-15)
 err=float(np.max(abs(norm(base['image'][:128])@norm(base['text'][:256]).T-norm(scores['image'][:128])@norm(scores['text'][:256]).T)))
 new_trace=float(np.trace((w['image']@r).T@cv[:ni,ni:]@(w['text']@r)));assert err<1e-10;assert abs(new_trace-base_trace)<1e-7*abs(base_trace)
 result.update(name=name,training_covariance_sum=base_trace,rotated_training_covariance_sum=new_trace,sample_cosine_error=err)
 p.save(dest,result);np.savez_compressed(OUT/(name+'-transforms.npz'),image=w['image']@r,text=w['text']@r,rotation=r);print(name,result['matching50']['summary']['agree_at1_count'],flush=True)
# Reuse exactly the existing CCA results, same population / input statistics / rank / optimizer.
sources={'cca_original':ROOT/'runs/multiview-likelihood-2026-10-09/cc3m-cc3m/cca_original-result.json'}
for name,src in sources.items():p.save(OUT/(name+'-result.json'),json.loads(src.read_text())['primary'])
for seed in range(3):
 src=ROOT/f'runs/rotation-convergence-2026-10-09/loading_varimax-{seed}-final-result.json';p.save(OUT/f'cca_rotated_{seed}-result.json',json.loads(src.read_text()))
p.save(OUT/'protocol.json',dict(condition='CC3M SAE / CC3M mapping',rank=256,pls_source=str(path),input='Same training-standardized SAE coordinates as CCA. PLS uses left/right singular vectors without whitening or singular-value weights.',weight_orthogonality=orth,rotation='Same normalized loading Varimax objective, equal modality weight, same three initializations and Cayley/Armijo optimizer, max6000 with confirmed plateau stopping. Shared orthogonal R. Final iterate, no annotation checkpoint selection.',evaluation='Same frozen COCO val2017 fixed50 own-caption mentions agreement and bidirectional Recall1/5/10. No labels in fitting or sign orientation.',signs='Same FIT32768 skew-based common sign rule as prior CCA rotation. Unmodified and sign-only PLS controls included.',limits='No global optimum guarantee; 3 initialization repeats not independent data replications.'))
p.save(OUT/'complete.json',dict(complete=True))
