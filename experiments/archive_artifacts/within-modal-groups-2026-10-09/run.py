"""Fixed-budget within-modality grouping. No category annotations in model fitting.

Compare |Pearson| with |conditional binary interaction| using the SAME overlapping
sparse SymNMF grouping and fixed-support nonnegative contrastive readouts.
The conditional model is a ridge-regularized pairwise 0/1 Ising pseudolikelihood.
This is an exploratory adaptation, not a reproduction of the referenced paper.
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import experiments.apbt_search_20261009 as prior

ROOT = Path('/mnt/working/mm-sae')
BASE = ROOT / 'runs/within-modal-groups-2026-10-09'
prior.BASE = BASE
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
save = prior.save
K, R = 16, 256


def progress(**kw):
    save(BASE / 'progress.json', dict(updated_unix=time.time(), **kw))
    print(json.dumps(kw), flush=True)


def ising_nll(z, w, h):
    """Mean per-node conditional binary NLL; diagonal has no self predictor."""
    j = (w + w.T) / 2
    j = j - torch.diag_embed(j.diagonal())
    return F.binary_cross_entropy_with_logits(z @ j + h, z), j


def self_test():
    # Conditional probability from the explicit joint energy must equal logits.
    from itertools import product
    z = torch.tensor(list(product([0., 1.], repeat=3)), dtype=torch.float64)
    w = torch.tensor([[0., .4, -.2], [.4, 0., .7], [-.2, .7, 0.]], dtype=torch.float64, requires_grad=True)
    h = torch.tensor([-.6, .2, -.1], dtype=torch.float64, requires_grad=True)
    for row in z:
        for i in range(3):
            v0, v1 = row.clone(), row.clone()
            v0[i], v1[i] = 0., 1.
            energy = lambda v: .5 * v @ w @ v + v @ h
            actual = torch.sigmoid(energy(v1) - energy(v0))
            expected = torch.sigmoid(row @ w[:, i] + h[i])
            torch.testing.assert_close(actual, expected)
    assert torch.autograd.gradcheck(lambda w, h: ising_nll(z, w, h)[0], (w, h))
    # The efficient off-diagonal SymNMF objective must equal direct reconstruction.
    g = torch.rand(7, 7, dtype=torch.float64); g = (g + g.T) / 2; g.fill_diagonal_(0)
    v = torch.rand(7, 3, dtype=torch.float64)
    direct = (g - v @ v.T).square(); direct.fill_diagonal_(0)
    efficient = g.square().sum() - 2 * (v * (g @ v)).sum() + (v.T @ v).square().sum() - v.square().sum(1).square().sum()
    torch.testing.assert_close(direct.sum(), efficient)
    print('Self-tests passed: Ising joint/conditional identity, gradients, SymNMF loss.', flush=True)


def conditional_graph(x, xt, out, condition, side, max_steps):
    path = out / f'{side}-conditional.npz'
    if path.exists():
        with np.load(path) as z:
            return torch.tensor(z['interaction'], device='cuda')
    binary = (x > 0).float(); bt = (xt > 0).float(); d = x.shape[1]
    rate = binary.mean(0).clamp(1e-5, 1-1e-5)
    w = torch.nn.Parameter(torch.zeros(d, d, device='cuda'))
    h = torch.nn.Parameter(torch.logit(rate))
    opt = torch.optim.LBFGS([w, h], lr=1., max_iter=1, history_size=8,
                           line_search_fn='strong_wolfe', tolerance_grad=1e-7,
                           tolerance_change=1e-10)
    lam = .01
    history = []; best = float('inf'); stale = 0; start = time.monotonic()

    def closure():
        opt.zero_grad(set_to_none=True)
        total = torch.zeros((), device='cuda')
        for first in range(0, len(binary), 2048):
            batch = binary[first:first+2048]
            loss, _ = ising_nll(batch, w, h)
            part = loss * (len(batch) / len(binary))
            part.backward(); total += part.detach()
        j = (w + w.T) / 2; j = j - torch.diag_embed(j.diagonal())
        penalty = (lam / (2*d)) * j.square().sum()
        penalty.backward()
        return total + penalty.detach()

    for step in range(max_steps+1):
        t0 = time.monotonic()
        train = float(opt.step(closure)) if step else None
        with torch.no_grad():
            tune, j = ising_nll(bt, w, h)
            value = float(tune)
            if value < best - 1e-7:
                best = value; beststep = step; stale = 0
                bestj = j.detach().cpu().numpy().copy(); besth = h.detach().cpu().numpy().copy()
            else:
                stale += 1
        row = dict(step=step, train_penalized_before_step=train, tune_binary_nll=value,
                   seconds=time.monotonic()-start, step_seconds=time.monotonic()-t0)
        history.append(row); save(out / f'{side}-conditional-history.json', history)
        progress(state='conditional_graph', condition=condition, side=side, **row)
        if step >= 15 and stale >= 8:
            break
    np.savez_compressed(path, interaction=bestj, intercept=besth,
                        prevalence=rate.cpu().numpy(), ridge=lam, best_step=beststep)
    save(out / f'{side}-conditional-fit.json', dict(best_step=beststep, completed_step=step,
         best_tune_nll=best, zero_interaction_tune_nll=history[0]['tune_binary_nll'],
         stopped_by='patience' if stale>=8 else 'iteration_budget', seconds=time.monotonic()-start,
         meaning='Pairwise binary conditional associations, not causal connections or guaranteed concepts.'))
    del binary, bt, w, h, opt
    torch.cuda.empty_cache()
    return torch.tensor(bestj, device='cuda')


def group_graph(relation, eligible, path, condition, side, method, steps=400):
    if path.exists():
        with np.load(path) as z:
            return z['weights'].copy()
    t0 = time.monotonic()
    # Both inputs use absolute edge strength; sign retention is diagnostic only.
    g = relation.abs().clone(); g.fill_diagonal_(0)
    g[~eligible, :] = 0; g[:, ~eligible] = 0
    degree = g.sum(1).clamp_min(1e-12)
    g /= torch.sqrt(degree[:, None] * degree[None, :])
    g /= g.square().mean().sqrt().clamp_min(1e-9)
    d = len(g)
    assert int(eligible.sum()) >= R
    # Deterministic farthest-first seeds in the graph's normalized row profiles.
    profiles = F.normalize(g, dim=1)
    novelty = torch.zeros(d, device='cuda'); available = eligible.clone()
    seeds = []; current = int(torch.where(available, degree, -torch.inf).argmax())
    for _ in range(R):
        seeds.append(current); available[current] = False
        novelty = torch.maximum(novelty, profiles @ profiles[current])
        current = int(torch.where(available, novelty, torch.inf).argmin())
    h0 = torch.zeros(d, R, device='cuda')
    for col, seed in enumerate(seeds):
        scores = g[:, seed].clone(); scores[~eligible] = -torch.inf
        scores[seed] = torch.inf
        ix = scores.topk(K).indices
        vals = g[ix, seed].clamp_min(1e-3).sqrt(); vals[ix == seed] = vals.max()
        h0[ix, col] = vals
    h = torch.nn.Parameter(h0.clone()); opt = torch.optim.Adam([h], lr=.03)
    norm = g.square().sum(); history = []; best = float('inf')
    for step in range(steps+1):
        opt.zero_grad(set_to_none=True)
        # Diagonal omitted so trivial self-correlation cannot drive grouping.
        loss = (norm - 2*(h*(g@h)).sum() + (h.T@h).square().sum()
                - h.square().sum(1).square().sum()) / norm
        val = float(loss.detach())
        if val < best:
            best = val; besth = h.detach().cpu().numpy().copy(); beststep = step
        if step % 20 == 0 or step == steps:
            history.append(dict(step=step, relative_offdiagonal_error=val,
                                elapsed=time.monotonic()-t0))
        if step == steps:
            break
        loss.backward(); opt.step()
        with torch.no_grad():
            h.clamp_(min=0); h[~eligible] = 0
            ix = h.topk(K, dim=0).indices
            h *= torch.zeros_like(h).scatter_(0, ix, 1)
            dead = h.norm(dim=0)<1e-8
            h[:, dead] = h0[:, dead]
    norms = np.linalg.norm(besth, axis=0)
    assert np.all(norms > 0)
    weights = besth / norms
    supports = weights != 0
    pairs = []
    rel = relation.detach().cpu().numpy()
    for k in range(R):
        inds = np.flatnonzero(supports[:, k]); v = rel[np.ix_(inds, inds)]
        pairs.extend(v[np.triu_indices(len(inds), 1)].tolist())
    np.savez_compressed(path, weights=weights, seeds=seeds, eligible=eligible.cpu().numpy())
    save(path.with_suffix('.json'), dict(method=method, side=side, groups=R, max_group_size=K,
         distinct_supports=len({tuple(np.flatnonzero(supports[:, c])) for c in range(R)}),
         coordinates_covered=int(supports.any(1).sum()), eligible_coordinates=int(eligible.sum()),
         mean_within_group_signed_relation=float(np.mean(pairs)),
         fraction_negative_within_group=float(np.mean(np.array(pairs)<0)),
         best_step=beststep, relative_error=best, seconds=time.monotonic()-t0, history=history))
    progress(state='groups_ready', condition=condition, method=method, side=side,
             seconds=time.monotonic()-t0, relative_error=best)
    return weights


def fit_readout(arr, init, path, condition, method, seed=0, epochs=60):
    if path.exists():
        return
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    a,b=[torch.nn.Parameter(torch.tensor(w, device='cuda')) for w in init]
    masks=[w.detach().ne(0) for w in [a,b]]
    with torch.no_grad():
        cross=prior.corr(arr['fit','image'][:4096]@a, arr['fit','text'][:4096]@b)
        _,pp=linear_sum_assignment(-cross.cpu().numpy())
    p=torch.tensor(pp, device='cuda')
    opt=torch.optim.Adam([a,b],lr=.003)
    def loss(x,y):
        u=F.normalize(x@a,dim=1);v=F.normalize((y@b)[:,p],dim=1)
        logits=u@v.T/.07; labels=torch.arange(len(x), device='cuda')
        return (F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))/2
    def validation():
        with torch.no_grad():
            return float(torch.stack([loss(arr['tune','image'][i:i+512],arr['tune','text'][i:i+512])
                                      for i in range(0,len(arr['tune','image']),512)]).mean())
    start=time.monotonic();best=float('inf');stale=0;hist=[]
    for epoch in range(epochs+1):
        tl=[]
        if epoch:
            seq=rng.permutation(len(arr['fit','image']))
            for first in range(0,len(seq),512):
                ix=torch.tensor(seq[first:first+512],device='cuda')
                value=loss(arr['fit','image'][ix],arr['fit','text'][ix])
                assert torch.isfinite(value)
                opt.zero_grad(set_to_none=True); value.backward()
                torch.nn.utils.clip_grad_norm_([a,b],5);opt.step()
                with torch.no_grad():
                    for w,m in zip([a,b],masks):
                        w.clamp_(min=0);w.mul_(m);w.div_(w.norm(dim=0,keepdim=True).clamp_min(1e-9))
                tl.append(float(value.detach()))
        vl=validation()
        if vl<best-1e-5:
            best=vl;bestepoch=epoch;stale=0;ww=[w.detach().cpu().numpy().copy() for w in [a,b]]
        else:stale+=1
        row=dict(epoch=epoch,tune_loss=vl,train_loss=float(np.mean(tl)) if tl else None,
                 elapsed=time.monotonic()-start)
        hist.append(row);save(path.with_name(path.stem+'-history.json'),hist)
        if epoch%5==0:progress(state='readout_training',condition=condition,method=method,**row)
        if epoch and stale>=5:break
    np.savez_compressed(path,image=ww[0],text=ww[1],permutation=pp,centered=False,
                        image_support=masks[0].cpu().numpy(),text_support=masks[1].cpu().numpy())
    save(path.with_name(path.stem+'-fit.json'),dict(best_epoch=bestepoch,completed_epoch=epoch,
         best_tune_loss=best,seconds=time.monotonic()-start,permutation_changed=int(np.sum(pp!=np.arange(R))),
         fixed_support=True,annotation_free_fitting=True))


def run(args):
    self_test()
    out=BASE/args.condition;out.mkdir(parents=True,exist_ok=True)
    save(out/'protocol.json',dict(condition=args.condition,seed=0,n_fit=32768,n_tune=2048,
         groups=R,activation_limit=K,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         graph='Absolute Pearson correlation versus absolute symmetric Ising interaction. Same degree normalization, sparse overlapping SymNMF, diagonal excluded.',
         conditional='0/1 activation indicators. Symmetric interactions, free per-node intercept, pseudo-likelihood plus .01*||J||_F^2/(2*d). L-BFGS. Choose checkpoint by held-out conditional NLL.',
         group='400 projected Adam steps, learning rate .03, <=16 nonnegative memberships per column; shared clustering algorithm.',
         representation='Raw nonnegative SAE activations divided by train standard deviation; mean is not subtracted for readout. Correlation graph itself is centered.',
         readout='Fixed supports, nonnegative unit-L2 columns, symmetric cosine InfoNCE temperature .07, Adam .003, batch512, max60 epochs, patience5.',
         matching='Full 256-group permutation maximizing correlation on first 4096 FIT pairs at initialization, then fixed. Partial/unmatched matching NOT tested.',
         annotation='No category annotations in grouping, fitting, pairing or checkpoint choice. Existing evaluation data has been examined previously.',
         evaluation='COCO val2017 retrieval; image presence and OWN caption dictionary mentions select representatives independently, existing 42 eligible categories.',
         control='Existing Sparse CCA supports, absolute normalized weights, same initialization permutation rule and same fixed-support readout.',
         assumptions='Pairwise binary model is approximate; TopK competition and correlated distinct concepts can affect groups. Not a semantic identification theorem.'))
    progress(state='loading',condition=args.condition)
    cfg,arr,initial=prior.prepare(args.condition,out,32768,0)
    with np.load(Path(cfg['parent_run'])/'moments.npz') as z:
        np.savez_compressed(out/'feature-ids.npz',image=z['image_ids'],text=z['text_ids'])
    grouped={key:[] for key in ['correlation_groups','conditional_groups']}
    for side in ['image','text']:
        x,xt=arr['fit',side],arr['tune',side]
        counts=(x>0).sum(0)
        eligible=(counts>=32)&(counts<=len(x)-32)&(x.std(0)>1e-7)
        xc=F.normalize(x-x.mean(0),dim=0)
        corr=xc.T@xc;del xc
        np.savez_compressed(out/f'{side}-correlation.npz',correlation=corr.cpu().numpy(),
                            eligible=eligible.cpu().numpy(),counts=counts.cpu().numpy())
        grouped['correlation_groups'].append(group_graph(corr,eligible,out/f'{side}-correlation-groups.npz',
                                              args.condition,side,'correlation'))
        del corr
        interaction=conditional_graph(x,xt,out,args.condition,side,args.ising_steps)
        grouped['conditional_groups'].append(group_graph(interaction,eligible,out/f'{side}-conditional-groups.npz',
                                              args.condition,side,'conditional'))
        del interaction
    control=[np.abs(w)/np.maximum(np.linalg.norm(w,axis=0,keepdims=True),1e-9) for w in initial]
    for name,weights in [('cca_support_control',control),*grouped.items()]:
        fit_readout(arr,weights,out/(name+'.npz'),args.condition,name)
    del arr;torch.cuda.empty_cache()
    data,parents=prior.eval_population(cfg,'test',out)
    prior.save(out/'evaluation-population.json',dict(metadata=data['metadata'],
         eligible_ids=[c['id'] for c,e in zip(data['concepts'],data['eligible']) if e],
         text_annotation='Own caption mentions.npy replaces inherited image labels.',
         counts={k:data['labels'][k.split('_')[0]][m].sum(0) for k,m in data['masks'].items()}))
    jobs=[(key,Path(next(m['path'] for m in cfg['models'] if m['key']==key)))
          for key in ['sparse_cca_16','cca_256']]
    jobs += [(name,out/(name+'.npz')) for name in ['cca_support_control',*grouped]]
    for name,path in jobs:
        prior.evaluate(cfg,data,parents,name,path,out,'test')
    progress(state='condition_completed',condition=args.condition)
    save(out/'complete.json',dict(completed=True,condition=args.condition,time_unix=time.time()))


def binary_control(args):
    """Isolate binarization from conditional modeling, with every other step fixed."""
    self_test()
    out=BASE/args.condition
    assert (out/'complete.json').exists(), 'Main comparison must complete first.'
    save(out/'binary-control-protocol.json',dict(
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rationale='Conditional model uses binary activations. This control uses Pearson correlation of the same indicators, isolating the input binarization from conditional associations.',
        all_other_settings='Exactly the same fit/tune rows, count>=32 eligibility, graph normalization, grouping, matching and readout as main comparison. No evaluation-dependent tuning.'))
    cfg,arr,_=prior.prepare(args.condition,out,32768,0)
    init=[]
    for side in ['image','text']:
        z=(arr['fit',side]>0).float();counts=z.sum(0)
        eligible=(counts>=32)&(counts<=len(z)-32)&(arr['fit',side].std(0)>1e-7)
        z=F.normalize(z-z.mean(0),dim=0); corr=z.T@z;del z
        np.savez_compressed(out/f'{side}-binary-correlation.npz',correlation=corr.cpu().numpy())
        init.append(group_graph(corr,eligible,out/f'{side}-binary-correlation-groups.npz',
                                 args.condition,side,'binary_correlation'))
        del corr
    name='binary_correlation_groups'
    fit_readout(arr,init,out/(name+'.npz'),args.condition,name)
    del arr;torch.cuda.empty_cache()
    data,parents=prior.eval_population(cfg,'test',out)
    prior.evaluate(cfg,data,parents,name,out/(name+'.npz'),out,'test')
    save(out/'binary-control-complete.json',dict(completed=True,time_unix=time.time()))
    progress(state='binary_control_completed',condition=args.condition)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',choices=list(prior.CONFIG),default='coco-coco')
    p.add_argument('--ising-steps',type=int,default=80);p.add_argument('--self-test',action='store_true')
    p.add_argument('--binary-only',action='store_true')
    args=p.parse_args()
    try:
        if args.self_test:self_test()
        elif args.binary_only:binary_control(args)
        else:run(args)
    except Exception as error:
        progress(state='failed',condition=args.condition,error=repr(error),traceback=traceback.format_exc())
        raise
