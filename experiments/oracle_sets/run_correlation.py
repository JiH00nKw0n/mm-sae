"""Exploratory Ridge concept probes plus paired correlation, fit moments only."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from mm_sae.analysis.data import CachedSplit, save_json
from mm_sae.analysis.evaluation import auroc
from mm_sae.analysis.mapping_evaluation import paired_retrieval
from mm_sae.metrics.regression import Moments


def objective(a, b, xx, yy, xy, qx, qy, variance, alpha, ridge, local=False):
    if local:
        ax = torch.einsum('cij,cj->ci', xx, a)
        by = torch.einsum('cij,cj->ci', yy, b)
        vx, vy = (a * ax).sum(1), (b * by).sum(1)
        cov = torch.einsum('ci,cij,cj->c', a, xy, b)
        errors = variance - 2*(a*qx).sum(1)+vx + variance-2*(b*qy).sum(1)+vy
        reg = (a*a).sum(1)+(b*b).sum(1)
    else:
        vx, vy = (a*(xx@a)).sum(0), (b*(yy@b)).sum(0)
        cov = (a*(xy@b)).sum(0)
        errors = 2*variance-2*(a*qx).sum(0)+vx-2*(b*qy).sum(0)+vy
        reg = (a*a).sum(0)+(b*b).sum(0)
    corr = cov / torch.sqrt(vx.clamp_min(1e-12)*vy.clamp_min(1e-12))
    return (0.5*errors+0.5*ridge*reg-alpha*corr).mean()


def fit(xx, yy, xy, qx, qy, variance, a, b, alpha, cfg, progress):
    initial_a, initial_b = a.clone(), b.clone()
    history=[]
    k=cfg['budget']; stop='iteration_limit'
    def full(a,b): return objective(a,b,xx,yy,xy,qx,qy,variance,alpha,cfg['ridge'])
    initial=float(full(a,b)); value=initial
    for it in range(cfg['max_outer']):
        aa=a.detach().requires_grad_();bb=b.detach().requires_grad_()
        ga,gb=torch.autograd.grad(full(aa,bb),(aa,bb))
        # Rescale the gradient of the concept mean to a per-concept step.
        ga=ga*a.shape[1];gb=gb*b.shape[1]
        accepted=False
        for attempt in range(12):
            eta=0.1/(2**attempt)
            ia=(a-eta*ga).abs().topk(k,dim=0).indices.T
            ib=(b-eta*gb).abs().topk(k,dim=0).indices.T
            cc=torch.arange(a.shape[1],device=a.device)[:,None]
            la=a.T.gather(1,ia).clone().requires_grad_();lb=b.T.gather(1,ib).clone().requires_grad_()
            args=(xx[ia[:,:,None],ia[:,None,:]],yy[ib[:,:,None],ib[:,None,:]],xy[ia[:,:,None],ib[:,None,:]],qx[ia,cc],qy[ib,cc],variance,alpha,cfg['ridge'])
            opt=torch.optim.LBFGS([la,lb],lr=1,max_iter=cfg['inner_iter'],line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-12)
            def closure():
                opt.zero_grad(); loss=objective(la,lb,*args,local=True);loss.backward();return loss
            opt.step(closure)
            candidate=float(objective(la,lb,*args,local=True).detach())
            if np.isfinite(candidate) and candidate <= value+1e-10:
                na=torch.zeros_like(a).scatter(0,ia.T,la.detach().T)
                nb=torch.zeros_like(b).scatter(0,ib.T,lb.detach().T)
                accepted=True;break
        if not accepted:stop='line_search_failed';break
        changed=int(((na!=0)!=(a!=0)).sum()+((nb!=0)!=(b!=0)).sum())
        gain=value-candidate;a,b=na,nb;value=candidate
        record=dict(iteration=it+1,objective=value,gain=gain,support_changes=changed)
        history.append(record);progress(record)
        if changed==0 and gain<1e-8:stop='objective_tolerance_and_stable_support';break
    assert value<=initial+1e-9
    return a,b,dict(initial_objective=initial,objective=value,stop_reason=stop,history=history,support_changes_from_ridge=int(((a!=0)!=(initial_a!=0)).sum()+((b!=0)!=(initial_b!=0)).sum()),global_optimum_claim=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args();cfg=yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    root=Path(cfg['root']);out=root/cfg['output'];out.mkdir(parents=True,exist_ok=True)
    save_json(out/'protocol.json',dict(config=cfg,method='ridge_plus_per_concept_Pearson_correlation_not_full_CCA',loss='mean over concepts of half image MSE plus half text MSE plus L2 minus alpha times Pearson correlation',concept_decorrelation=False,fit_only=True,intercept='fixed fit label prevalence',retrieval='fit unit variance linear scores without intercept',zero_variance_floor=1e-12))
    for condition in cfg['conditions']:
        c=condition['name'];dest=out/c;dest.mkdir(exist_ok=True)
        def status(state,**kw):
            j=dict(condition=c,state=state,time=time.time(),**kw);save_json(out/'progress.json',j);print(json.dumps(j),flush=True)
        status('loading_moments')
        parent=root/condition['parent'];old=root/'runs/oracle-losses-2026-10-06'/c
        pop=json.loads((parent/'population.json').read_text())
        sets=[set(pop[k+'_image_ids']) for k in ['fit','tune','test']];assert all(not sets[i]&sets[j] for i,j in [(0,1),(0,2),(1,2)])
        with np.load(parent/'moments.npz') as z:
            ids={s:z[s+'_ids'] for s in ['image','text']};mom=Moments(int(z['fit_n']),z['fit_mean'],z['fit_second'])
        cov=mom.standardized(mom);ni=len(ids['image']);xx=cov[:ni,:ni];yy=cov[ni:,ni:];xy=cov[:ni,ni:]
        qs=[];starts=[]
        for side,block in [('image',slice(0,ni)),('text',slice(ni,None))]:
            with np.load(old/f'{side}_moments.npz') as z:
                np.testing.assert_allclose(z['mean'],mom.mean[block]);np.testing.assert_allclose(z['scale'],mom.scale[block]);np.testing.assert_allclose(z['cov'],cov[block,block],atol=1e-8)
                qs.append(z['cross']);prevalence=z['label_mean'];variance=z['label_variance']
            with np.load(old/f'ridge_0.01_{side}.npz') as z:starts.append(z['raw'])
        cv=torch.as_tensor(cov,device='cuda',dtype=torch.float64)
        tensors=[cv[:ni,:ni],cv[ni:,ni:],cv[:ni,ni:],*[torch.as_tensor(v,device='cuda',dtype=torch.float64) for v in qs],torch.as_tensor(variance,device='cuda',dtype=torch.float64)]
        test=CachedSplit(root/condition['source'],'val2017');np.testing.assert_array_equal(test.image_ids,pop['test_image_ids'])
        for alpha in cfg['alphas']:
            file=dest/f'alpha_{alpha:g}.json'
            if file.exists():continue
            status('fitting',alpha=alpha)
            a,b=[torch.as_tensor(v,device='cuda',dtype=torch.float64).clone() for v in starts]
            begin=time.monotonic()
            if alpha: a,b,audit=fit(*tensors,a,b,alpha,cfg,lambda j:status('fitting',alpha=alpha,**j))
            else:audit=dict(stop_reason='existing_ridge_reference',support_changes_from_ridge=0)
            ws=[a.cpu().numpy(),b.cpu().numpy()];scores={};semantic={};train_errors={}
            for side,w,cc,q,block in [('image',ws[0],xx,qs[0],slice(0,ni)),('text',ws[1],yy,qs[1],slice(ni,None))]:
                sd=np.sqrt(np.maximum(np.einsum('ic,ic->c',w,cc@w),1e-12));coef=w/sd
                raw=np.asarray(test.activations[side][:,ids[side]]@(w/mom.scale[block,None]))-mom.mean[block]@(w/mom.scale[block,None])
                scores[side]=raw/sd
                labels=test.presence if side=='image' else test.presence[test.parents]
                aucs=[auroc(np.asarray(labels[:,j]).ravel(),raw[:,j]) for j in range(w.shape[1])]
                semantic[side]=dict(mean_auroc=float(np.nanmean(aucs)),per_concept_auroc=aucs,test_mse=float(np.mean((raw+prevalence-labels)**2)))
                train_errors[side]=float(np.mean(variance-2*(w*q).sum(0)+(w*(cc@w)).sum(0)))
            status('evaluating',alpha=alpha)
            metrics=paired_retrieval(scores['image'],scores['text'],test.parents,device='cuda',chunk_size=128)
            save_json(file,dict(alpha=alpha,audit=audit,train_mse=train_errors,semantic=semantic,retrieval=metrics,seconds=time.monotonic()-begin))
            np.savez_compressed(dest/f'alpha_{alpha:g}.npz',image=ws[0],text=ws[1],image_ids=ids['image'],text_ids=ids['text'])
        del cv,tensors,test,a,b;torch.cuda.empty_cache()
        status('condition_completed')
    status('completed')
if __name__=='__main__':main()
