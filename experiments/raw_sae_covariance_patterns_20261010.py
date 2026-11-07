"""Raw SAE covariance/correlation geometry, without any learned cross-modal map."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import json
import numpy as np
from experiments import apbt_search_20261009 as p

ROOT=p.ROOT;OUT=ROOT/'runs/raw-sae-covariance-patterns-2026-10-10';OUT.mkdir(exist_ok=True)
SRC=ROOT/'runs/loading-structure-audit-2026-10-09'
cfg=p.old.load_config(ROOT/p.CONFIG['cc3m-cc3m'])
with np.load(Path(cfg['parent_run'])/'moments.npz') as z:
    ni=len(z['image_ids']);ids={s:z[s+'_ids'].copy() for s in ['image','text']}
    n=int(z['fit_n']);mu=z['fit_mean'];cov=z['fit_second']-np.outer(mu,mu)
sd=np.sqrt(np.maximum(cov.diagonal(),0));assert np.all(sd>0)
corr=cov/sd[:,None]/sd[None,:]
assert np.max(abs(np.diag(corr)-1))<1e-10
data,_=p.eval_population(cfg,'test',OUT)
eligible=np.flatnonzero(np.isin([c['id'] for c in data['concepts']],data['metadata']['eligible_category_ids']))
assert len(eligible)==50
names=[data['concepts'][i]['name'] for i in eligible]
del data
feat={}
for s in ['image','text']:
    z=np.load(SRC/(s+'-feature-aucs.npz'));a=z['cal'];b=z['test'];active=z['active']
    assert len(a)==len(ids[s])
    ix=np.flatnonzero((a.max(1)>=.60)&(active>=20))
    feat[s]={'local_index':ix,'global_index':ix+(ni if s=='text' else 0),'best_auc':a[ix].max(1),
             'cal_label':a[ix].argmax(1),'test_label':b[ix].argmax(1),
             'test_auc_for_cal_label':b[ix,a[ix].argmax(1)]}
    p.save(OUT/(s+'-selected-features.json'),[dict(sae_coordinate=int(ids[s][j]),moment_coordinate=int(j),
        calibration_category=names[int(a[j].argmax())],repeat_category=names[int(b[j].argmax())],
        calibration_auc=float(a[j].max()),repeat_auc_for_calibration_category=float(b[j,a[j].argmax()])) for j in ix])

def cosine(a,b):
    aa=np.linalg.norm(a,axis=1);bb=np.linalg.norm(b,axis=1)
    assert np.all(aa>0) and np.all(bb>0)
    return (a/aa[:,None])@(b/bb[:,None]).T

def shared_anchor_cosine(matrix,ii,jj):
    # Both raw modalities form the SAME anchor list. Exclude both endpoints
    # separately for each pair so own variance/direct edge cannot inflate similarity.
    a=matrix[ii];b=matrix[jj];dot=a@b.T
    ai=matrix[ii,ii];bj=matrix[jj,jj];ab=matrix[np.ix_(ii,jj)]
    dot-=ai[:,None]*ab+ab*bj[None,:]
    an=np.sum(a*a,1)[:,None]-ai[:,None]**2-ab**2
    bn=np.sum(b*b,1)[None,:]-bj[None,:]**2-ab**2
    return dot/np.sqrt(np.maximum(an*bn,1e-30))

pairs=json.loads((ROOT/'runs/rotation-selectivity-2026-10-09/pairs.json').read_text())
partners={i:set() for i in range(50)}
for pair in pairs:
    if pair['first'] in names and pair['second'] in names:
        a=names.index(pair['first']);b=names.index(pair['second']);partners[a].add(b);partners[b].add(a)

def summarize(mat,la,lb,same_modality=False,cooccurring=False):
    rows=[]
    for c in range(50):
        aa=np.flatnonzero(la==c);bb=np.flatnonzero(lb==c)
        dd=np.flatnonzero(np.isin(lb,list(partners[c]))) if cooccurring else np.flatnonzero(lb!=c)
        if not len(aa) or not len(bb) or not len(dd):continue
        same=mat[np.ix_(aa,bb)].ravel()
        if same_modality:
            keep=(aa[:,None]!=bb[None,:]).ravel();same=same[keep]
        if not len(same):continue
        other=mat[np.ix_(aa,dd)].ravel()
        rows.append(dict(category=names[c],same=float(same.mean()),different=float(other.mean()),
                         same_pairs=len(same),different_pairs=len(other)))
    if not rows:return {'categories':0}
    a=np.array([r['same'] for r in rows]);b=np.array([r['different'] for r in rows]);delta=a-b
    rng=np.random.default_rng(1010);resampled=delta[rng.integers(0,len(rows),(10000,len(rows)))].mean(1)
    return dict(categories=len(rows),same=float(a.mean()),different=float(b.mean()),difference=float(delta.mean()),
                categories_same_larger=int(np.sum(delta>0)),category_resample_difference95=np.quantile(resampled,[.025,.975]).tolist(),per_category=rows)

results={};ii=feat['image']['global_index'];jj=feat['text']['global_index']
for stat,matrix in [('covariance',cov),('correlation',corr)]:
    matrices={
        'direct_image_text':matrix[np.ix_(ii,jj)],
        'image_profiles_over_text':cosine(matrix[ii,ni:],matrix[ii,ni:]),
        'text_profiles_over_image':cosine(matrix[jj,:ni],matrix[jj,:ni]),
        'cross_modal_profiles_over_shared_raw_anchors':shared_anchor_cosine(matrix,ii,jj)}
    results[stat]={}
    for threshold in [.60,.65,.70]:
        results[stat][str(threshold)]={}
        keep={s:np.flatnonzero(v['best_auc']>=threshold) for s,v in feat.items()}
        for labelsource in ['cal_label','test_label']:
            group={}
            for key,mat in matrices.items():
                sa,sb=('image','image') if key.startswith('image_profiles') else ('text','text') if key.startswith('text_profiles') else ('image','text')
                sub=mat[np.ix_(keep[sa],keep[sb])];la=feat[sa][labelsource][keep[sa]];lb=feat[sb][labelsource][keep[sb]]
                group[key]=dict(all_other=summarize(sub,la,lb,sa==sb),
                                cooccurring_other=summarize(sub,la,lb,sa==sb,True))
            results[stat][str(threshold)][labelsource]=group
    p.save(OUT/'results.json',results);print('finished',stat,flush=True)
selection={str(th):{s:dict(features=int(np.sum(v['best_auc']>=th)),
    represented_categories=len(set(v['cal_label'][v['best_auc']>=th].tolist())),
    repeat_label_agreement=float(np.mean(v['cal_label'][v['best_auc']>=th]==v['test_label'][v['best_auc']>=th])),
    repeat_auc_for_fixed_label=float(v['test_auc_for_cal_label'][v['best_auc']>=th].mean())) for s,v in feat.items()} for th in [.60,.65,.70]}
p.save(OUT/'selection.json',selection)
p.save(OUT/'protocol.json',dict(sae_training='CC3M',statistics_dataset='Existing full CC3M paired mapping-fit population, but no mapping used',
    paired_sample_count=n,image_coordinates=ni,text_coordinates=len(sd)-ni,coordinate_scope='Same previously retained nonconstant SAE coordinates. No CCA/PLS/Hungarian, projections, rotations, or dimension alignment.',
    covariance='Raw SAE E[zzT]-E[z]E[z]T, no standard deviation scaling.',correlation='Raw covariance divided by BOTH raw-coordinate standard deviations.',
    feature_labels='Existing raw SAE AUROC cache. Primary labels and feature selection use COCO val half A. Text uses own-caption category mentions. Repeat labels use half B while selected feature IDs stay fixed.',
    main_threshold=.65,sensitivity_thresholds=[.60,.65,.70],feature_selection='Best positive AUROC>=threshold and >=20 positive raw activations on half A. Among previously fixed50 objects.',
    direct='Cross-modal scalar covariance/correlation, same-category vs different-category feature pairs, category macro average.',
    within_profiles='Compare TWO IMAGE coordinates over ALL TEXT anchors, or TWO TEXT coordinates over ALL IMAGE anchors. No counterpart feature matching is learned.',
    cross_profiles='Compare image and text features over common concatenated raw image+text anchors. Remove both compared coordinates from each pair of profiles, including self variance and direct edge.',
    cooccurrence='Additional negatives drawn from previously fixed12 cooccurring concept pairs where qualifying features exist.',
    limits='Feature semantics operationally defined by highest object AUROC, not pure concepts. Only selected category-responsive coordinates assessed. Profiles may reflect shared contexts. Statistical relation alone does not identify causal concepts or prove matching method superiority. Repeat uses same validation dataset disjoint halves. Bootstrap resamples categories, not independent datasets.'))
p.save(OUT/'complete.json',{'complete':True})
