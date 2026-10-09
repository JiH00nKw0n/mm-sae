"""Known-factor, annotation-free alignment under controlled co-occurrence.

The training functions receive observations only. Generator factors/loadings
are used after fitting to measure counterfactual selectivity, never fit models.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from mm_sae.analysis.sparse_cca import fit_sparse_cca
from mm_sae.analysis.mapping_ablation import fit_common_projection
from mm_sae.analysis.mapping_evaluation import paired_retrieval


def roc_auc_score(labels,values):
    labels=np.asarray(labels,dtype=bool)
    n1=int(labels.sum());n0=len(labels)-n1
    return (rankdata(values)[labels].sum()-n1*(n1+1)/2)/(n1*n0)


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False,
        default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()))
    tmp.replace(path)


def loadings(seed,d=128,r=8):
    """Eight private anchor coordinates plus eight overlapping coordinates per factor."""
    rng=np.random.default_rng(seed)
    D=np.zeros((r,d))
    anchors=[]
    for j in range(r):
        cols=np.arange(8*j,8*(j+1));anchors.append(cols)
        D[j,cols]=rng.uniform(.75,1.25,8)
    col=64
    for cycle in range(4):
        order=rng.permutation(r)
        for j in range(r):
            D[[order[j],order[(j+1)%r]],col]=rng.uniform(.75,1.25,2)
            col+=1
    D/=np.linalg.norm(D,axis=1,keepdims=True)
    perm=rng.permutation(d)
    inverse=np.argsort(perm)
    D=D[:,perm]
    # Feasible sparse reference. Eight anchor inputs exactly isolate a factor.
    oracle=np.zeros((d,r))
    for j,cols in enumerate(anchors):
        cc=inverse[cols];oracle[cc,j]=1/(8*D[j,cc])
    np.testing.assert_allclose(D@oracle,np.eye(r),atol=1e-12)
    assert (np.count_nonzero(D,axis=1)==16).all()
    return D.astype('float32'),oracle.astype('float32')


def observations(seed,n,rho,di,dt):
    rng=np.random.default_rng(seed)
    c1=rng.random(n)<.3
    independent=rng.random(n)<.3
    copy=rng.random(n)<rho
    c2=np.where(copy,c1,independent)
    other=(rng.random((n,6))<.35)*rng.gamma(4,.25,(n,6))
    z=np.column_stack([c1,c2,other]).astype('float32')
    def obs(D):
        noise=(rng.random((n,D.shape[1]))<.05)*rng.gamma(2,.025,(n,D.shape[1]))
        return (z@D+noise).astype('float32')
    return obs(di),obs(dt),z


@torch.no_grad()
def project(w,k=16,positive=False):
    if positive:w.clamp_(min=0)
    idx=w.abs().topk(k,dim=0).indices
    mask=torch.zeros_like(w).scatter_(0,idx,1)
    w.mul_(mask).div_(w.norm(dim=0,keepdim=True).clamp_min(1e-10))


def nce(x,y,a,b):
    score=F.normalize(x@a,dim=1)@F.normalize(y@b,dim=1).T/.07
    ix=torch.arange(len(x),device=x.device)
    return (F.cross_entropy(score,ix)+F.cross_entropy(score.T,ix))/2


def train(x,y,tx,ty,initial,positive,seed,epochs,out,name):
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    tensors=[torch.as_tensor(v,device='cuda') for v in (x,y,tx,ty)]
    x,y,tx,ty=tensors
    a=torch.nn.Parameter(torch.tensor(initial[0],device='cuda'))
    b=torch.nn.Parameter(torch.tensor(initial[1],device='cuda'))
    if positive:
        with torch.no_grad():a.abs_();b.abs_()
    for w in (a,b):project(w,positive=positive)
    opt=torch.optim.Adam([a,b],lr=.003)
    history=[];best=float('inf');best_epoch=0;stale=0
    began=time.monotonic()
    for epoch in range(epochs+1):
        losses=[]
        if epoch:
            perm=rng.permutation(len(x))
            for start in range(0,len(x),512):
                ix=torch.as_tensor(perm[start:start+512],device='cuda')
                opt.zero_grad(set_to_none=True)
                v=nce(x[ix],y[ix],a,b)
                if not torch.isfinite(v):raise ValueError('Nonfinite loss')
                v.backward();torch.nn.utils.clip_grad_norm_([a,b],5.)
                opt.step()
                for w in (a,b):project(w,positive=positive)
                losses.append(v.item())
        with torch.no_grad():
            tune=float(np.mean([nce(tx[j:j+512],ty[j:j+512],a,b).item() for j in range(0,len(tx),512)]))
        if tune<best-1e-5:
            best,best_epoch,stale=tune,epoch,0
            weights=[v.detach().cpu().numpy().copy() for v in (a,b)]
        else:stale+=1
        history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)) if losses else None,tune_loss=tune))
        if epoch%10==0:
            print(json.dumps(dict(condition=out.name,method=name,epoch=epoch,tune_loss=tune,seconds=time.monotonic()-began)),flush=True)
        if epoch>=20 and stale>=10:break
    for w in weights:
        assert np.count_nonzero(w,axis=0).max()<=16
        if positive:assert w.min()>=0
    save(out/f'{name}-fit.json',dict(history=history,best_epoch=best_epoch,selected_tune_loss=best,
         seconds=time.monotonic()-began,optimizer='projected Adam, unit norm columns',seed=seed))
    return weights


def selectivity(di,dt,a,b,si,st):
    mi=(di/si)@a;mt=(dt/st)@b
    qi=mi**2/np.maximum((mi**2).sum(0),1e-15)
    qt=mt**2/np.maximum((mt**2).sum(0),1e-15)
    # Require corresponding coordinates and the same polarity in both modalities.
    score=np.minimum(qi,qt)*(mi*mt>0)
    targets,cols=linear_sum_assignment(-score[:2])
    rows,allcols=linear_sum_assignment(-score)
    return dict(target_mean=float(score[targets,cols].mean()),target_scores=score[targets,cols].tolist(),
        target_coordinates=cols.tolist(),target_image_signs=np.sign(mi[targets,cols]).tolist(),
        target_text_signs=np.sign(mt[targets,cols]).tolist(),all_factor_mean=float(score[rows,allcols].mean()),
        response_image=mi.tolist(),response_text=mt.tolist(),joint_selectivity=score.tolist())


def evaluate(name,a,b,center,means,scales,di,dt,tests,out):
    sel=selectivity(di,dt,a,b,*scales)
    result=dict(selectivity=sel,retrieval={},exclusive_category_auc={})
    for mode,(x,y,z) in tests.items():
        u=((x-means[0] if center else x)/scales[0])@a
        v=((y-means[1] if center else y)/scales[1])@b
        rr=paired_retrieval(u,v,np.arange(len(x)),device='cuda',chunk_size=256)
        result['retrieval'][mode]=rr
        exclusive=z[:,0]!=z[:,1]
        if exclusive.sum():
            aucs=[]
            for values,signs in [(u,sel['target_image_signs']),(v,sel['target_text_signs'])]:
                for target,col in enumerate(sel['target_coordinates']):
                    aucs.append(float(roc_auc_score(z[exclusive,target],values[exclusive,col]*signs[target])))
            result['exclusive_category_auc'][mode]=float(np.mean(aucs))
        else:result['exclusive_category_auc'][mode]=None
    np.savez_compressed(out/f'{name}.npz',image=a,text=b,centered=center)
    save(out/f'{name}-evaluation.json',result)
    print(json.dumps(dict(condition=out.name,method=name,target_selectivity=sel['target_mean'],
        iid_R5=result['retrieval']['iid']['image_to_text']['recall'][5])),flush=True)


def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    protocol=dict(train_n=8192,tune_n=2048,test_n=2048,input_dimensions=128,output_dimensions=8,max_support=16,
        rhos=[0,.5,.9,1],seeds=list(range(args.seeds)),max_epochs=args.epochs,
        target_generation='C1 Bernoulli(.3); C2 copies C1 with probability rho, otherwise independent Bernoulli(.3). Both target amplitudes are binary, so rho=1 offers no independent target changes in training.',
        other_factors='Six independent Bernoulli(.35) times Gamma(shape=4,scale=.25) variables, shared across views.',
        views='Different nonnegative loading matrices. Each factor touches 16 inputs: 8 exclusive anchors plus 8 overlapping inputs. Independent sparse nonnegative noise per view.',
        pairing='P=I throughout. No partial correspondence learned.',
        training='Observations and pair identities only. No latent labels or generator loadings enter fitting or checkpoint selection.',
        test='IID test plus separate independent-target test at rho=0. Known loadings measure exact one-factor intervention responses.',
        selectivity='For each output and modality, squared target response divided by sum of squared responses to all 8 unit interventions. Take minimum across modalities with matching sign. Assign two distinct output coordinates to two target factors to maximize the total score, evaluation only.',
        reference='Known-generator anchor reference uses the true loading matrix and is not an annotation-free learning method.',
        limitations='Fixed small additive generator, same Sparse CCA warm start per condition, no universal concept-identifiability claim.')
    save(out/'protocol.json',protocol)
    total=args.seeds*4;completed=0;started=time.monotonic()
    for seed in range(args.seeds):
        di,oi=loadings(1000+seed);dt,ot=loadings(2000+seed)
        for rho in [0,.5,.9,1]:
            dest=out/f'seed{seed}-rho{rho:.1f}';dest.mkdir(exist_ok=True)
            if (dest/'complete.json').exists():completed+=1;continue
            save(out/'progress.json',dict(state='running',seed=seed,rho=rho,completed=completed,total=total,seconds=time.monotonic()-started))
            x,y,z=observations(3000+seed,8192,rho,di,dt)
            tx,ty,tz=observations(4000+seed,2048,rho,di,dt)
            tests=dict(iid=observations(5000+seed,2048,rho,di,dt),
                       recombined=observations(6000+seed,2048,0,di,dt))
            means=[x.mean(0),y.mean(0)];scales=[x.std(0).clip(1e-6),y.std(0).clip(1e-6)]
            xn=(x-means[0])/scales[0];yn=(y-means[1])/scales[1]
            cov=np.cov(np.column_stack([xn,yn]).astype(float),rowvar=False,bias=True)
            base=fit_common_projection(cov[:128,:128],cov[:128,128:],cov[128:,128:],8,whiten=True,ridge=.01)
            sparse_model=fit_sparse_cca(cov[:128,:128],cov[:128,128:],cov[128:,128:],8,k=16,ridge=.01,
                initial_image=base.image,initial_text=base.text,max_iter=100,swap_steps=2,candidate_pool=8)
            initial=[sparse_model.image.astype('float32'),sparse_model.text.astype('float32')]
            np.savez_compressed(dest/'generator.npz',image=di,text=dt,mean_image=means[0],mean_text=means[1],
                                scale_image=scales[0],scale_text=scales[1])
            save(dest/'data.json',dict(target_prevalence=z[:,:2].mean(0),empirical_gate_correlation=float(np.corrcoef(z[:,:2].T)[0,1]),
                 n_exclusive_train=int((z[:,0]!=z[:,1]).sum()),n_joint_train=int(((z[:,0]==1)&(z[:,1]==1)).sum())))
            evaluate('sparse_cca',*initial,True,means,scales,di,dt,tests,dest)
            # Transform the known anchor reference into the standardized input parameterization.
            evaluate('known_anchor_reference',oi*scales[0][:,None],ot*scales[1][:,None],False,means,scales,di,dt,tests,dest)
            for name,center,positive in [('nce_centered',True,False),('nce_uncentered',False,False),('nce_nonnegative',False,True)]:
                prepared=[(v-means[i%2] if center else v)/scales[i%2] for i,v in enumerate([x,y,tx,ty])]
                weights=train(*prepared,initial,positive,seed,args.epochs,dest,name)
                evaluate(name,*weights,center,means,scales,di,dt,tests,dest)
            completed+=1
            save(dest/'complete.json',dict(completed=True))
    save(out/'progress.json',dict(state='completed',completed=completed,total=total,seconds=time.monotonic()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=60);p.add_argument('--seeds',type=int,default=3)
    a=p.parse_args()
    try:main(a)
    except Exception as e:
        save(Path(a.output)/'progress.json',dict(state='failed',error=repr(e)))
        raise
