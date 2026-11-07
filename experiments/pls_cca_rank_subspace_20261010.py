"""Rank sweep and fixed-subspace objective crossover on frozen CC3M SAE."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
import time,json
from pathlib import Path
import numpy as np
import torch
from scipy import linalg
from experiments import apbt_search_20261009 as p
from experiments import multiview_likelihood_20261009 as multi
from mm_sae.analysis.mapping_ablation import fit_common_projection
from mm_sae.analysis.mapping_evaluation import paired_retrieval
ROOT=p.ROOT;OUT=ROOT/'runs/pls-cca-rank-subspace-2026-10-10';OUT.mkdir(exist_ok=True)
start=time.time()
def status(**d):p.save(OUT/'progress.json',dict(elapsed=time.time()-start,**d));print(json.dumps(d),flush=True)
cfg=p.old.load_config(ROOT/p.CONFIG['cc3m-cc3m']);sides=['image','text']
with np.load(Path(cfg['parent_run'])/'moments.npz') as z:
    ni=len(z['image_ids']);nt=len(z['text_ids']);mu=z['fit_mean'];cv=z['fit_second']-np.outer(mu,mu)
sd=np.sqrt(np.maximum(cv.diagonal(),0));sd[sd==0]=1;cv/=sd[:,None]*sd[None,:]
sigmas={'image':cv[:ni,:ni],'text':cv[ni:,ni:]};cross=cv[:ni,ni:]
maxrank=min(ni,nt);grid=[128,256,512,1024,2048,maxrank]
weights={}
for m,stem in [('cca','cca'),('pls','cross_svd')]:
    with np.load(ROOT/f'runs/mapping-ablation-cc3m-2026-10-04/transforms/{stem}_{maxrank}.npz') as z:weights[m]={s:z[s].astype(float) for s in sides}
p.save(OUT/'protocol.json',dict(sae_training='CC3M',mapping_training='CC3M',ranks=grid,requested4096='Not possible with existing retained3695 image and3221 text coordinates. Common independent rank max3221. No duplicated dimensions.',
    fitting='Existing full SVD/CCA leading columns, same training-standardized input and lambda=.01. No validation labels or search evaluation in fitting.',
    crossover='For each rank, QR gives Euclidean-orthonormal bases of each existing selected subspace. Solve PLS or regularized CCA inside that SAME pair of subspaces. Ridge=.01 remains same under orthonormal bases. No dimensions discarded in within-subspace fit.',
    matching='Original CCA/PLS only, fixed50 own-caption mention protocol, native cached coordinate signs, no interpretation rotation. Not concept count.',
    evaluation='Same COCO val2017 retrieval R1/5/10 both directions. Image-cluster paired bootstrap at r256 using stored ranks.',
    limits='Dimension is representation capacity, not the number of recovered concepts. Crossover order-dependent effects are not universal causal percentages. Subspace overlap is coefficient-space geometry in standardized SAE coordinates.'))
status(stage='load_evaluation');data,parents=p.eval_population(cfg,'test',OUT)
base={m:{s:np.asarray(data['raw'][s]@(weights[m][s]/data['scales'][s][:,None]))-data['means'][s]@(weights[m][s]/data['scales'][s][:,None]) for s in sides} for m in weights}
def evaluate(scores,name,matching):
    dest=OUT/(name+'-result.json')
    if dest.exists():return
    status(stage='evaluate',name=name)
    ret=paired_retrieval(scores['image'],scores['text'],parents,device='cuda',chunk_size=128)
    result={'retrieval':{d:{k:v for k,v in z.items() if k not in ['hits','ranks']} for d,z in ret.items()}}
    np.savez_compressed(OUT/(name+'-ranks.npz'),image=ret['image_to_text']['ranks'],text=ret['text_to_image']['ranks'],parents=parents)
    if matching:result['matching50'],_=multi.matching(scores,data,cfg)
    p.save(dest,result)
    print(name,[ret[d]['recall'][5] for d in ret],result.get('matching50',{}).get('summary',{}).get('agree_at1_count'),flush=True)
geometry={}
for rank in grid:
    status(stage='rank_start',rank=rank)
    qs={};rs={}
    for m in weights:
        w={s:weights[m][s][:,:rank] for s in sides};scores={s:base[m][s][:,:rank] for s in sides}
        evaluate(scores,f'{m}-native-{rank}',True)
        qs[m]={};rs[m]={}
        for s in sides:qs[m][s],rs[m][s]=linalg.qr(w[s],mode='economic')
        q=qs[m];xx=q['image'].T@sigmas['image']@q['image'];yy=q['text'].T@sigmas['text']@q['text'];xy=q['image'].T@cross@q['text']
        objective='cca' if m=='pls' else 'pls'
        name=f'{m}-space-{objective}-objective-{rank}'
        if not (OUT/(name+'-result.json')).exists():
            status(stage='fit_fixed_subspace',rank=rank,space=m,objective=objective)
            projection=fit_common_projection(xx,xy,yy,rank,whiten=objective=='cca',ridge=.01)
            ds={'image':projection.image,'text':projection.text};maps={s:linalg.solve_triangular(rs[m][s],ds[s]) for s in sides}
            result_scores={}
            for s in sides:
                with torch.no_grad():result_scores[s]=(torch.tensor(scores[s],device='cuda',dtype=torch.float32)@torch.tensor(maps[s],device='cuda',dtype=torch.float32)).cpu().numpy()
            evaluate(result_scores,name,False);del result_scores
            np.savez_compressed(OUT/(name+'-within-space-transforms.npz'),**maps)
    geometry[str(rank)]={s:float(np.sum((qs['cca'][s].T@qs['pls'][s])**2)/rank) for s in sides}
    p.save(OUT/'subspace-overlap.json',geometry)
    status(stage='rank_complete',rank=rank)
p.save(OUT/'complete.json',dict(complete=True,seconds=time.time()-start));status(stage='complete')
