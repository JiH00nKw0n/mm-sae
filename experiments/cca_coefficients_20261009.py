"""Frozen dense CCA coefficient concentration, pruning, and pair correlations."""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
from pathlib import Path
import numpy as np,json,time,argparse,hashlib,traceback
import experiments.apbt_search_20261009 as old
from mm_sae.metrics.regression import Moments
from mm_sae.analysis.mapping_pruning import prune_coefficients
ROOT=Path('/mnt/working/mm-sae');BASE=ROOT/'runs/cca-coefficients-2026-10-09';old.BASE=BASE
save=old.save
KS=[8,16,32,64,128,256];MASSES=[.9,.95,.99]
def stats(v):
    v=np.asarray(v,dtype=float)
    assert np.isfinite(v).all()
    return dict(mean=float(v.mean()),median=float(np.median(v)),minimum=float(v.min()),maximum=float(v.max()),
                p10=float(np.quantile(v,.1)),p90=float(np.quantile(v,.9)))
def concentration(w):
    square=np.sort(w*w,axis=0)[::-1];cum=np.cumsum(square,axis=0)/np.sum(square,axis=0)
    counts={str(p):np.minimum((cum<p).sum(0)+1,w.shape[0]) for p in [.5,.9,.95,.99]}
    return dict(input_coordinates=w.shape[0],outputs=w.shape[1],
                counts_for_squared_coefficient_mass={p:stats(c) for p,c in counts.items()},
                counts_per_component={p:c.tolist() for p,c in counts.items()},
                topk_squared_coefficient_mass={str(k):stats(cum[min(k,w.shape[0])-1]) for k in KS},
                negative_fraction=float((w<0).sum()/w.size))
def prune_mass(w,p):
    order=np.argsort(-np.abs(w),axis=0,kind='stable')
    square=np.take_along_axis(w*w,order,axis=0)
    cum=np.cumsum(square,axis=0)/square.sum(0)
    count=np.minimum((cum<p).sum(0)+1,w.shape[0]);mask=np.arange(w.shape[0])[:,None]<count
    out=np.zeros_like(w);np.put_along_axis(out,order,np.where(mask,np.take_along_axis(w,order,axis=0),0),axis=0)
    assert np.all((out*out).sum(0)/(w*w).sum(0)>=p-1e-12)
    return out
def corr_scores(a,b):
    a=a-a.mean(0);b=b-b.mean(0)
    return (a*b).sum(0)/np.sqrt((a*a).sum(0)*(b*b).sum(0))
def project(data,w):
    return {s:np.asarray(data['raw'][s]@(w[s]/data['scales'][s][:,None]))-data['means'][s]@(w[s]/data['scales'][s][:,None]) for s in ['image','text']}
def main(condition):
    out=BASE/condition;out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    cfg=old.old.load_config(ROOT/old.CONFIG[condition]);source=Path(next(m['path'] for m in cfg['models'] if m['key']=='cca_256'))
    save(out/'protocol.json',dict(condition=condition,config=cfg,source_model=str(source),source_model_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        pruning='Absolute coefficients, per column, both modalities. Preserve remaining coefficients and signs. No refit, no rescaling, no test-selected cutoff.',
        output_coordinates=256,k_values=KS,squared_coefficient_mass=MASSES,
        evaluation='COCO val2017 retrieval and independent own-caption/image representative matching with 42 eligible categories.',
        pearson='Centered separately on the population being measured; COCO image vectors repeated according to their caption parent IDs.',
        ridge=.01,important='Squared coefficient mass is NOT explained variance because the inputs are correlated.'))
    with np.load(source) as z:full={s:z[s].copy() for s in ['image','text']}
    save(out/'concentration.json',{s:concentration(w) for s,w in full.items()})
    with np.load(Path(cfg['parent_run'])/'moments.npz') as z:
        ni=len(z['image_ids']);fit=Moments(int(z['fit_n']),z['fit_mean'],z['fit_second'])
    cov=fit.standardized(fit);xx,xy,yy=cov[:ni,:ni],cov[:ni,ni:],cov[ni:,ni:]
    cf={'image':xx@full['image'],'text':yy@full['text']}
    vf={s:(full[s]*cf[s]).sum(0) for s in full}
    for s,c in [('image',xx),('text',yy)]:
        w=full[s];assert np.allclose(w.T@c@w+.01*w.T@w,np.eye(w.shape[1]),atol=2e-7)
    data,parents=old.eval_population(cfg,'test',out)
    save(out/'evaluation.json',dict(metadata=data['metadata'],concepts=data['concepts'],eligible_ids=[c['id'] for c,e in zip(data['concepts'],data['eligible']) if e],fit_pairs=fit.n))
    fullscores=project(data,full)
    jobs=[('cca_full',full)]+[(f'keep_{k}',{s:prune_coefficients(w,k,rule='column') for s,w in full.items()}) for k in KS]
    jobs += [(f'mass_{int(p*100)}',{s:prune_mass(w,p) for s,w in full.items()}) for p in MASSES]
    for name,weights in jobs:
        path=out/(name+'.npz');np.savez_compressed(path,**weights,centered=True)
        a,b=weights['image'],weights['text'];cx=xx@a;cy=yy@b
        va=(a*cx).sum(0);vb=(b*cy).sum(0);ab=(a*(xy@b)).sum(0)
        fitrho=ab/np.sqrt(va*vb);reg=ab/np.sqrt((va+.01*(a*a).sum(0))*(vb+.01*(b*b).sum(0)))
        assert np.max(np.abs(fitrho))<=1+1e-8 and np.max(np.abs(reg))<=1+1e-8
        scores=fullscores if name=='cca_full' else project(data,weights)
        testrho=corr_scores(scores['image'][parents],scores['text'])
        diag=dict(method=name,condition=condition,nonzero_counts={s:stats((w!=0).sum(0)) for s,w in weights.items()},
             retained_squared_coefficient_mass={s:stats((weights[s]**2).sum(0)/(full[s]**2).sum(0)) for s in weights},
             fit_pearson=stats(fitrho),fit_regularized_pair_correlation=stats(reg),test_pearson=stats(testrho),
             fit_pearson_sum=float(fitrho.sum()),fit_regularized_sum=float(reg.sum()),test_pearson_sum=float(testrho.sum()),
             fit_above_09=int((fitrho>=.9).sum()),test_above_09=int((testrho>=.9).sum()),
             fit_above_095=int((fitrho>=.95).sum()),test_above_095=int((testrho>=.95).sum()),
             fit_above_099=int((fitrho>=.99).sum()),test_above_099=int((testrho>=.99).sum()),
             original_vs_pruned_fit_correlation={s:stats((weights[s]*cf[s]).sum(0)/np.sqrt(v*vf[s])) for s,v in [('image',va),('text',vb)]},
             original_vs_pruned_test_correlation={s:stats(corr_scores(fullscores[s],scores[s])) for s in scores})
        if name=='cca_full':
            diag['fit_objective_trace']=float(ab.sum());diag['fit_objective_maximum_bound']=256
            assert np.allclose(ab,reg,atol=1e-8)
        np.savez_compressed(out/(name+'-correlations.npz'),fit_pearson=fitrho,fit_regularized=reg,test_pearson=testrho,
            **{s+'_counts':(w!=0).sum(0) for s,w in weights.items()})
        save(out/(name+'-diagnostics.json'),diag)
        old.evaluate(cfg,data,parents,name,path,out,'test')
        print(json.dumps(dict(condition=condition,method=name,fit_pearson=diag['fit_pearson'],test_pearson=diag['test_pearson'],seconds=time.monotonic()-start)),flush=True)
    sparsepath=Path(next(m['path'] for m in cfg['models'] if m['key']=='sparse_cca_16'))
    old.evaluate(cfg,data,parents,'sparse_cca_16',sparsepath,out,'test')
    save(out/'complete.json',dict(completed=True,seconds=time.monotonic()-start))
if __name__=='__main__':
    BASE.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    try:
        for condition in old.CONFIG:main(condition)
        save(BASE/'complete.json',dict(completed=True,seconds=time.monotonic()-start))
    except Exception as e:save(BASE/'error.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
