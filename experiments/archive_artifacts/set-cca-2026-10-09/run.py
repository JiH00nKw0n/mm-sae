"""Within-modal activation SETS, followed by CCA restricted to each set pair.

No InfoNCE. No gradient training after set discovery. No 16-coordinate cap.
One leading canonical pair per matched set pair. Standard CCA with the same
number of output coordinates is an explicit dimensionality control.
"""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
from pathlib import Path
import json,time,random,argparse,hashlib,traceback
import numpy as np
from scipy import sparse,linalg
from scipy.optimize import linear_sum_assignment
import igraph as ig
import experiments.apbt_search_20261009 as old
from mm_sae.analysis.mapping_ablation import fit_common_projection
from mm_sae.metrics.regression import Moments

ROOT=Path('/mnt/working/mm-sae')
BASE=ROOT/'runs/set-cca-2026-10-09'
SOURCE=ROOT/'runs/within-modal-groups-2026-10-09'
old.BASE=BASE
save=old.save
METHODS=['correlation','binary_correlation','conditional']

def status(**kw):
    save(BASE/'progress.json',dict(time_unix=time.time(),**kw));print(json.dumps(kw),flush=True)

def relation(condition,side,method):
    folder=SOURCE/condition
    if method=='conditional':
        with np.load(folder/f'{side}-conditional.npz') as z:r=z['interaction'].copy()
    else:
        with np.load(folder/f'{side}-{method.replace("_","-")}.npz') as z:r=z['correlation'].copy()
    with np.load(folder/f'{side}-correlation.npz') as z:eligible=z['eligible'].copy()
    return r,eligible

def prepare_binary(condition,out):
    """Build the additional input-type control only; no contrastive training."""
    need=[s for s in ['image','text'] if not (SOURCE/condition/f'{s}-binary-correlation.npz').exists()]
    if not need:return
    import torch
    import torch.nn.functional as F
    cfg,arr,_=old.prepare(condition,out,32768,0)
    for side in need:
        z=(arr['fit',side]>0).float();z=F.normalize(z-z.mean(0),dim=0)
        r=z.T@z
        np.savez_compressed(SOURCE/condition/f'{side}-binary-correlation.npz',correlation=r.cpu().numpy())
    del arr;torch.cuda.empty_cache()

def groups(condition,side,method,out):
    path=out/f'{side}-{method}-sets.json'
    if path.exists():return json.loads(path.read_text())['groups']
    t0=time.monotonic();r,eligible=relation(condition,side,method);ids=np.flatnonzero(eligible)
    g=np.abs(r[np.ix_(ids,ids)]).astype('float64');np.fill_diagonal(g,0)
    degree=g.sum(1).clip(1e-12)
    g/=np.sqrt(degree[:,None]*degree[None,:])
    # Full weighted graph. There is no nearest-neighbor count or edge cutoff.
    ig.set_random_number_generator(random.Random(0))
    graph=ig.Graph.Weighted_Adjacency(sparse.coo_matrix(np.triu(g,1)),mode='upper',loops=False)
    levels=graph.community_multilevel(weights='weight',resolution=1.,return_levels=True)
    # Finest nontrivial partition returned by Louvain; no test-based selection.
    cluster=levels[0]
    result=sorted([sorted(ids[np.array(m,dtype=int)].tolist()) for m in cluster],key=lambda x:x[0])
    flat=[i for s in result for i in s]
    assert len(flat)==len(set(flat))==len(ids)
    sizes=[len(x) for x in result]
    info=dict(condition=condition,side=side,method=method,groups=result,n_groups=len(result),
        min_size=min(sizes),median_size=float(np.median(sizes)),max_size=max(sizes),sizes=sizes,
        eligible_coordinates=len(ids),full_coordinates=len(r),resolution=1.,seed=0,
        selected_level=0,level_counts=[len(x) for x in levels],modularity=float(cluster.modularity),
        overlap=False,seconds=time.monotonic()-t0)
    save(path,info);status(state='sets_ready',**{k:v for k,v in info.items() if k not in ['groups','sizes']})
    return result

def moments(cfg):
    with np.load(Path(cfg['parent_run'])/'moments.npz') as z:
        ni=len(z['image_ids'])
        fit=Moments(int(z['fit_n']),z['fit_mean'],z['fit_second'])
    c=fit.standardized(fit)
    return c[:ni,:ni],c[:ni,ni:],c[ni:,ni:]

def fit_sets(xx,xy,yy,gi,gt,out,name):
    path=out/(name+'.npz')
    if path.exists():return np.load(path)['image'].shape[1]
    ni,nt=xy.shape;t0=time.monotonic()
    # Precompute within-set inverse Cholesky factors.
    whiten=lambda cov,indices:linalg.solve_triangular(
        linalg.cholesky(cov[np.ix_(indices,indices)]+.01*np.eye(len(indices)),lower=True),
        np.eye(len(indices)),lower=True)
    wi=[whiten(xx,s) for s in gi];wt=[whiten(yy,s) for s in gt]
    score=np.zeros((len(gi),len(gt)));coeff={}
    for i,si in enumerate(gi):
        for j,tj in enumerate(gt):
            cross=wi[i]@xy[np.ix_(si,tj)]@wt[j].T
            u,s,vt=linalg.svd(cross,full_matrices=False,check_finite=False)
            ai=wi[i].T@u[:,0];bt=wt[j].T@vt[0]
            score[i,j]=s[0];coeff[i,j]=(ai,bt)
            # The same regularized CCA constraints hold in every set pair.
            assert abs(ai@(xx[np.ix_(si,si)]+.01*np.eye(len(si)))@ai-1)<1e-6
            assert abs(bt@(yy[np.ix_(tj,tj)]+.01*np.eye(len(tj)))@bt-1)<1e-6
        status(state='pairwise_cca',method=name,image_set=i+1,image_sets=len(gi),
               text_sets=len(gt),seconds=time.monotonic()-t0)
    ii,jj=linear_sum_assignment(-score)
    r=len(ii);a=np.zeros((ni,r));b=np.zeros((nt,r));pairs=[]
    p=np.zeros((len(gi),len(gt)),dtype='uint8');p[ii,jj]=1
    for k,(i,j) in enumerate(zip(ii,jj)):
        ai,bt=coeff[i,j];a[gi[i],k]=ai;b[gt[j],k]=bt
        pairs.append(dict(coordinate=k,image_set=int(i),text_set=int(j),
             image_size=len(gi[i]),text_size=len(gt[j]),fit_canonical_correlation=float(score[i,j])))
    np.savez_compressed(path,image=a,text=b,centered=True,score=score,set_mapping=p,
        image_set_indices=ii,text_set_indices=jj,permutation=np.arange(r))
    save(out/(name+'-fit.json'),dict(method=name,outputs=r,image_sets=len(gi),text_sets=len(gt),
         unmatched_image_sets=len(gi)-r,unmatched_text_sets=len(gt)-r,pairs=pairs,
         ridge=.01,seconds=time.monotonic()-t0,
         sign='Unmodified SciPy SVD signs; no annotation-based reorientation. Report both positive-only and sign-permitted matching.',
         score='Leading regularized canonical correlation in each fixed set pair.',
         pairing='Maximum total training canonical correlation with one-to-one Hungarian matching of SETS. All sets on smaller side matched.',
         objective='CCA only; no contrastive retrieval optimization.'))
    return r

def main(condition):
    out=BASE/condition;out.mkdir(parents=True,exist_ok=True)
    save(out/'protocol.json',dict(condition=condition,time_unix=time.time(),
         code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         self_relations='Reuse fit-only raw Pearson and fitted Ising coefficients; add Pearson of binary activations as input-type control.',
         grouping='Full absolute weighted graph, degree normalization, Louvain finest partition at resolution1 seed0. No set-size cap and no fixed set count. Disjoint sets.',
         pairing='CCA first component fitted within EACH candidate pair of sets, ridge .01, Hungarian matching on the resulting canonical correlations.',
         representation='One leading canonical scalar per matched set pair. Signed weights inside fixed sets. No across-set rotation.',
         data='Same full mapping FIT moments as existing CCA. Within-set discovery uses earlier 32768 fit samples and frequency>=32 eligibility.',
         no_objective='No InfoNCE, no retrieval gradients, no label-based selection, no L1 sweep.',
         evaluation='Same existing COCO val2017 retrieval and independent own-caption/image representative matching, 42 eligible categories.',
         controls='Original CCA256 and Sparse CCA16; ordinary CCA truncated to exactly each discovered output dimension.',
         important='Output dimension is variable; representative agreement is mechanically easier when fewer candidate coordinates exist. Report same-dimensional CCA and matched representative diversity.',
         packages=dict(igraph=ig.__version__,numpy=np.__version__)))
    status(state='loading',condition=condition)
    cfg=old.old.load_config(ROOT/old.CONFIG[condition]);prepare_binary(condition,out)
    with np.load(SOURCE/condition/'feature-ids.npz') as z:
        np.savez_compressed(out/'feature-ids.npz',image=z['image'],text=z['text'])
    xx,xy,yy=moments(cfg);jobs=[]
    for method in METHODS:
        gi=groups(condition,'image',method,out);gt=groups(condition,'text',method,out)
        name='set_cca_'+method
        r=fit_sets(xx,xy,yy,gi,gt,out,name)
        jobs.append((name,out/(name+'.npz')))
        control=out/f'cca_dimension_{r}.npz'
        if not control.exists():
            source=next(m['path'] for m in cfg['models'] if m['key']=='cca_256')
            with np.load(source) as z:
                if r<=z['image'].shape[1]:a,b=z['image'][:,:r].copy(),z['text'][:,:r].copy()
                else:
                    fitted=fit_common_projection(xx,xy,yy,r,whiten=True,ridge=.01)
                    a,b=fitted.image,fitted.text
            np.savez_compressed(control,image=a,text=b,centered=True)
        if not any(n==control.stem for n,_ in jobs):jobs.append((control.stem,control))
    del xx,xy,yy
    data,parents=old.eval_population(cfg,'test',out)
    save(out/'evaluation.json',dict(eligible_ids=[c['id'] for c,e in zip(data['concepts'],data['eligible']) if e],
          concepts=data['concepts'],metadata=data['metadata'],text_labels='Own-caption dictionary mentions.'))
    refs=[(k,Path(next(m['path'] for m in cfg['models'] if m['key']==k))) for k in ['cca_256','sparse_cca_16']]
    for name,path in refs+jobs:
        result=old.evaluate(cfg,data,parents,name,path,out,'test')
        with np.load(path) as z:r=z['image'].shape[1]
        rows=[q for q in result['positive']['per_category'] if q.get('status')=='ok']
        matched=[q for q in rows if q['agree_at1']]
        save(out/(name+'-diagnostics.json'),dict(outputs=r,
            distinct_image_representatives=len({q['image_coordinate'] for q in rows}),
            distinct_text_representatives=len({q['text_coordinate'] for q in rows}),
            distinct_matched_representatives=len({q['image_coordinate'] for q in matched}),
            matched_with_both_auroc_above_07=sum(min(q['image_auc'],q['text_auc'])>=.7 for q in matched),
            uniform_random_coordinate_agreement=42/r))
    save(out/'complete.json',dict(completed=True,time_unix=time.time()));status(state='condition_complete',condition=condition)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--condition',choices=list(old.CONFIG),required=True)
    args=ap.parse_args()
    try:main(args.condition)
    except Exception as e:
        status(state='failed',condition=args.condition,error=repr(e),traceback=traceback.format_exc());raise
