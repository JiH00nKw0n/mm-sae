"""Frozen CC3M/CC3M subspace and output-statistic ablation. No label fitting."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import json
from pathlib import Path
import numpy as np
import torch
from experiments import rotation_convergence_20261009 as conv
from experiments import multiview_likelihood_20261009 as multi
from mm_sae.analysis.semantic_evaluation import auc_matrix
from mm_sae.analysis.representative_agreement import variable_coordinates

p=conv.p; ROOT=p.ROOT
OUT=ROOT/'runs/pls-cca-gap-2026-10-10'; OUT.mkdir(exist_ok=True)
cfg=p.old.load_config(ROOT/p.CONFIG['cc3m-cc3m'])
sides=['image','text']
sources={'cca':Path(next(m['path'] for m in cfg['models'] if m['key']=='cca_256')),
         'pls':ROOT/'runs/mapping-ablation-cc3m-2026-10-04/transforms/cross_svd_256.npz'}
weights={m:{s:np.load(path)[s].astype(float) for s in sides} for m,path in sources.items()}
with np.load(Path(cfg['parent_run'])/'moments.npz') as z:
    ni=len(z['image_ids']);mu=z['fit_mean'];cv=z['fit_second']-np.outer(mu,mu)
sd=np.sqrt(np.maximum(cv.diagonal(),0));sd[sd==0]=1;cv/=sd[:,None]*sd[None,:]
sigmas={'image':cv[:ni,:ni],'text':cv[ni:,ni:]}
loads={m:{s:sigmas[s]@weights[m][s] for s in sides} for m in weights}
grams={m:{s:weights[m][s].T@loads[m][s] for s in sides} for m in weights}
trans={};diagnostics={}
for m in weights:
    trans[m]={k:{} for k in ['original','standardized','whitened']};diagnostics[m]={}
    for s in sides:
        g=grams[m][s];e,q=np.linalg.eigh((g+g.T)/2);assert e.min()>0
        trans[m]['original'][s]=np.eye(256)
        trans[m]['standardized'][s]=np.diag(1/np.sqrt(np.diag(g)))
        trans[m]['whitened'][s]=(q/np.sqrt(e))@q.T
        diagnostics[m][s]={'output_variance_quantiles':np.quantile(np.diag(g),[0,.25,.5,.75,1]).tolist(),
                           'covariance_eigenvalue_minmax':[float(e.min()),float(e.max())]}
p.save(OUT/'diagnostics.json',diagnostics)

# Actual correlations after EACH proposed rotation, not just rotated initial correlations.
def correlation_objective(r,kind,ts,tune=False):
    values=[]
    for load,g in zip(ts['load'],ts['gram']):
        out_sd=(r*(g@r)).sum(0).clamp_min(1e-20).sqrt()
        corr=(load@r)/out_sd[None,:]
        corr=corr/corr.norm(dim=1,keepdim=True).clamp_min(1e-12)
        values.append((corr.pow(4).mean(0)-corr.square().mean(0).square()).sum())
    return -sum(values)/2
conv.objective=correlation_objective
for m in weights:
    conv.OUT=OUT/m;conv.OUT.mkdir(exist_ok=True)
    ts={'load':[torch.tensor(loads[m][s],device='cuda',dtype=torch.float64) for s in sides],
        'gram':[torch.tensor(grams[m][s],device='cuda',dtype=torch.float64) for s in sides]}
    for seed in range(3):conv.fit('correlation_varimax',seed,ts,type('Args',(),{'max_steps':6000})())
    del ts
torch.cuda.empty_cache()
data,parents=p.eval_population(cfg,'test',OUT)
base={m:{s:(np.asarray(data['raw'][s]@(weights[m][s]/data['scales'][s][:,None]))-
               data['means'][s]@(weights[m][s]/data['scales'][s][:,None])) for s in sides} for m in weights}
fit={}
for m,folder in [('cca','rotation-convergence-2026-10-09'),('pls','pls-loading-rotation-2026-10-09')]:
    with np.load(ROOT/'runs'/folder/'projected.npz') as z:fit[m]={s:z['fit_'+s].astype(float) for s in sides}

def evaluate_matching(scores):
    arr={}
    for s in sides:
        for half in ['a','b']:
            key=s+'_'+half;mask=data['masks'][key]
            arr[key]=auc_matrix(scores[s][mask],data['labels'][s][mask],chunk_size=32)
            arr[key+'_variable']=variable_coordinates(scores[s][mask])
    arr['eligible']=np.isin([c['id'] for c in data['concepts']],data['metadata']['eligible_category_ids'])
    assert arr['eligible'].sum()==50
    arr['category_ids']=np.array([c['id'] for c in data['concepts']])
    normal=p.old.evaluation(arr,{**cfg,'n_null':0},signed=False)
    swapped={k:v for k,v in arr.items()}
    for s in sides:
        for suff in ['', '_variable']:
            swapped[s+'_a'+suff]=arr[s+'_b'+suff];swapped[s+'_b'+suff]=arr[s+'_a'+suff]
    repeat=p.old.evaluation(swapped,{**cfg,'n_null':0},signed=False)
    return normal,repeat

for m in weights:
    for mode,ds in trans[m].items():
        dest=OUT/f'{m}-{mode}-result.json'
        if dest.exists():continue
        result=multi.metrics({s:base[m][s]@ds[s] for s in sides},data,parents,cfg)
        p.save(dest,result);print(m,mode,result['retrieval'],flush=True)
    jobs=[]
    for seed in range(3):
        old=ROOT/'runs'/('rotation-convergence-2026-10-09' if m=='cca' else 'pls-loading-rotation-2026-10-09')
        jobs.append((f'covariance-{seed}',np.load(old/f'loading_varimax-{seed}-final.npz')['rotation']))
        jobs.append((f'correlation-{seed}',np.load(OUT/m/f'correlation_varimax-{seed}-final.npz')['rotation']))
    for name,r in jobs:
        dest=OUT/f'{m}-{name}-result.json'
        if dest.exists():continue
        skew=sum(np.mean((fit[m][s]@r)**3,0) for s in sides);r=r*np.where(skew<0,-1.,1.)
        scores={s:base[m][s]@r for s in sides};ma,repeat=evaluate_matching(scores)
        norm=lambda x:x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-15)
        err=float(np.max(abs(norm(base[m]['image'][:128])@norm(base[m]['text'][:256]).T-
                             norm(scores['image'][:128])@norm(scores['text'][:256]).T)))
        assert err<1e-10
        p.save(dest,dict(matching50=ma,repeat_matching50=repeat,sample_cosine_error=err))
        np.savez_compressed(OUT/f'{m}-{name}-transforms.npz',image=weights[m]['image']@r,text=weights[m]['text']@r,rotation=r)
        print(m,name,ma['summary']['agree_at1_count'],repeat['summary']['agree_at1_count'],flush=True)

p.save(OUT/'protocol.json',dict(condition='CC3M SAE / CC3M mapping',rank=256,
    statistics='Full existing mapping-fit training moments. Evaluation statistics are never used for transforms.',
    retrieval='Same fixed COCO val2017. Original vs per-output standard deviation vs symmetric within-subspace whitening. Whitening preserves each selected subspace but changes cosine geometry; it does not optimize cross-modal orientation anew.',
    rotation='3 same initializations, same final-iterate optimizer and stopping; dynamically recompute corr(X,UR) using training Cov(U) for every rotation; normalize each row then maximize Varimax. Shared orthogonal R preserves cosine.',
    matching='Same fixed50 positive-polarity own-caption mention protocol, primary split plus reversed halves for representative-selection stability. Shared skew sign determined from FIT32768 only.',
    limits='Exploratory fixed-data ablation; whitening changes scale and correlations jointly. Remaining gaps do not alone prove subspace causation. Initialization repeats are not independent dataset samples.'))
p.save(OUT/'complete.json',{'complete':True})
