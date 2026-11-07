"""Fit-only Ising communities, then support-preserving CCA or PLS-SVD."""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
from pathlib import Path
import json,time,random,hashlib,traceback
import numpy as np
from scipy import sparse,linalg
from scipy.optimize import linear_sum_assignment
import igraph as ig
import experiments.set_cca_20261009 as base
import experiments.apbt_search_20261009 as prior
from mm_sae.analysis.mapping_ablation import fit_common_projection
ROOT=base.ROOT
BASE=ROOT/'runs/ising-leiden-2026-10-09'
prior.BASE=BASE
save=prior.save

def status(**kw):
    save(BASE/'progress.json',dict(time_unix=time.time(),**kw));print(json.dumps(kw),flush=True)

def groups(condition,side,out):
    path=out/f'{side}-groups.json'
    if path.exists():return json.loads(path.read_text())['groups']
    interaction,eligible=base.relation(condition,side,'conditional')
    ids=np.flatnonzero(eligible)
    w=np.abs(interaction[np.ix_(ids,ids)]).astype(float);w=(w+w.T)/2;np.fill_diagonal(w,0)
    ig.set_random_number_generator(random.Random(42))
    graph=ig.Graph.Weighted_Adjacency(sparse.coo_matrix(np.triu(w,1)),mode='upper',loops=False)
    partition=graph.community_leiden(objective_function='modularity',weights='weight',resolution=1.,n_iterations=-1)
    result=sorted([sorted(ids[np.asarray(c,dtype=int)].tolist()) for c in partition],key=lambda c:c[0])
    flat=[j for c in result for j in c]
    assert len(flat)==len(set(flat))==len(ids)
    sizes=[len(c) for c in result]
    info=dict(groups=result,sizes=sizes,n_groups=len(result),min_size=min(sizes),median_size=float(np.median(sizes)),max_size=max(sizes),eligible_coordinates=len(ids),full_coordinates=len(interaction),modularity=float(partition.modularity),resolution=1.,seed=42,graph='Absolute symmetric 0/1 Ising interactions, no degree normalization or edge pruning. Singleton groups retained.',source_sha256=hashlib.sha256((base.SOURCE/condition/f'{side}-conditional.npz').read_bytes()).hexdigest())
    save(path,info);status(state='groups_ready',condition=condition,side=side,**{k:v for k,v in info.items() if k not in ['groups','sizes']})
    return result

def fit(xx,xy,yy,gi,gt,out,method):
    name='ising_leiden_'+method;path=out/(name+'.npz')
    if path.exists():return np.load(path)['image'].shape[1]
    t=time.monotonic();cca=method=='cca'
    def whiten(cov,g):
        return linalg.solve_triangular(linalg.cholesky(cov[np.ix_(g,g)]+.01*np.eye(len(g)),lower=True),np.eye(len(g)),lower=True)
    wi=[whiten(xx,g) for g in gi] if cca else None
    wt=[whiten(yy,g) for g in gt] if cca else None
    score=np.zeros((len(gi),len(gt)));coeff={}
    for i,si in enumerate(gi):
        for j,tj in enumerate(gt):
            cross=xy[np.ix_(si,tj)]
            if cca:cross=wi[i]@cross@wt[j].T
            u,s,vt=linalg.svd(cross,full_matrices=False,check_finite=False)
            a,b=(wi[i].T@u[:,0],wt[j].T@vt[0]) if cca else (u[:,0],vt[0])
            cx=xx[np.ix_(si,si)]+.01*np.eye(len(si)) if cca else np.eye(len(si))
            cy=yy[np.ix_(tj,tj)]+.01*np.eye(len(tj)) if cca else np.eye(len(tj))
            assert np.allclose([a@cx@a,b@cy@b],[1,1],atol=1e-6)
            assert np.allclose(a@xy[np.ix_(si,tj)]@b,s[0],atol=1e-6)
            score[i,j]=s[0];coeff[i,j]=(a,b)
        status(state='fitting',method=name,image_group=i+1,total_image_groups=len(gi),seconds=time.monotonic()-t)
    ii,jj=linear_sum_assignment(-score);r=len(ii)
    a=np.zeros((xy.shape[0],r));b=np.zeros((xy.shape[1],r));p=np.zeros(score.shape,dtype=np.uint8);p[ii,jj]=1
    pairs=[]
    for k,(i,j) in enumerate(zip(ii,jj)):
        a[gi[i],k],b[gt[j],k]=coeff[i,j]
        pairs.append(dict(image_group=int(i),text_group=int(j),image_size=len(gi[i]),text_size=len(gt[j]),fit_score=float(score[i,j])))
    assert np.all((a!=0).sum(1)<=1) and np.all((b!=0).sum(1)<=1)
    np.savez_compressed(path,image=a,text=b,centered=True,set_mapping=p,score=score,permutation=np.arange(r))
    save(out/(name+'-fit.json'),dict(outputs=r,pairs=pairs,seconds=time.monotonic()-t,ridge=.01 if cca else 0,selection='Hungarian maximum sum of leading local regularized canonical correlations or covariances, fit data only',sign='Unmodified SVD sign; positive-only matching is sign dependent, so signed matching is also reported.'))
    return r

def main(condition):
    out=BASE/condition;out.mkdir(parents=True,exist_ok=True)
    cfg=prior.old.load_config(ROOT/prior.CONFIG[condition])
    save(out/'protocol.json',dict(condition=condition,grouping='Leiden modularity resolution1 on absolute within-modality Ising coefficients, independent modalities, variable group sizes. No concept labels.',ising='Reuse conditional pseudo-likelihood fit on 32768 fit samples, L2 .01, tune NLL stopping. Eligibility at least32 active and32 inactive samples.',readout='One leading CCA or PLS-SVD component per matched group pair. No mixing across groups. Hungarian partial permutation, every group on smaller side used.',data='Full original mapping-fit moments for CCA/PLS. Existing Ising feature IDs must match.',evaluation='COCO val2017 retrieval R1/5/10 both directions; independent representative coordinate agreement using image annotations and own-caption mentions. No test-based selection.',controls='Global CCA and PLS-SVD at256 and at exactly discovered group output count; existing Sparse CCA16.',code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    with np.load(base.SOURCE/condition/'feature-ids.npz') as source,np.load(Path(cfg['parent_run'])/'moments.npz') as moment:
        for side in ['image','text']:assert np.array_equal(source[side],moment[side+'_ids'])
    gi=groups(condition,'image',out);gt=groups(condition,'text',out)
    xx,xy,yy=base.moments(cfg);jobs=[]
    for method in ['cca','pls']:
        r=fit(xx,xy,yy,gi,gt,out,method);name='ising_leiden_'+method;jobs.append((name,out/(name+'.npz')))
        # Global controls are fitted once, then truncated in descending singular-value order.
        full=out/(method+'_256.npz')
        if not full.exists():
            if method=='cca':
                src=next(m['path'] for m in cfg['models'] if m['key']=='cca_256')
                with np.load(src) as z:a,b=z['image'].copy(),z['text'].copy()
            else:
                projection=fit_common_projection(xx,xy,yy,max(256,r),whiten=False)
                a,b=projection.image,projection.text
            np.savez_compressed(full,image=a,text=b,centered=True)
        with np.load(full) as z:a,b=z['image'].copy(),z['text'].copy()
        assert r<=a.shape[1]
        matched=out/(method+f'_dimension_{r}.npz');np.savez_compressed(matched,image=a[:,:r],text=b[:,:r],centered=True)
        jobs.extend([(full.stem,full),(matched.stem,matched)])
    jobs.append(('sparse_cca_16',Path(next(m['path'] for m in cfg['models'] if m['key']=='sparse_cca_16'))))
    del xx,xy,yy
    data,parents=prior.eval_population(cfg,'test',out)
    save(out/'evaluation.json',dict(metadata=data['metadata'],eligible_ids=[c['id'] for c,e in zip(data['concepts'],data['eligible']) if e]))
    for name,path in jobs:
        result=prior.evaluate(cfg,data,parents,name,path,out,'test')
        diagnostics={}
        for mode in ['positive','signed']:
            rows=[q for q in result[mode]['per_category'] if q.get('status')=='ok'];matched=[q for q in rows if q['agree_at1']]
            diagnostics[mode]=dict(distinct_matched_representatives=len({q['image_coordinate'] for q in matched}),matched_both_auc_ge_07=sum(min(q['image_auc'],q['text_auc'])>=.7 for q in matched))
        save(out/(name+'-diagnostics.json'),diagnostics)
    save(out/'complete.json',dict(completed=True,time_unix=time.time()));status(state='condition_complete',condition=condition)

if __name__=='__main__':
    try:
        for condition in ['coco-coco','cc3m-coco','cc3m-cc3m']:main(condition)
        save(BASE/'complete.json',dict(completed=True,time_unix=time.time()))
    except Exception as e:
        status(state='failed',error=repr(e),traceback=traceback.format_exc());raise
