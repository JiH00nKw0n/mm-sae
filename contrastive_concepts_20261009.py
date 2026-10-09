"""Exploratory sparse contrastive alignment. No category labels enter training.

Frozen CC3M SAE, original COCO fit/tune split, identity concept pairing.
Top-k projected Adam is a heuristic, not an exact cardinality optimizer.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse
import json
import time
from pathlib import Path
from collections import Counter
import numpy as np
from scipy import sparse
import torch
import torch.nn.functional as F
from experiments.representative_agreement.run import load_config, load_population, evaluation
from mm_sae.analysis.semantic_evaluation import auc_matrix
from mm_sae.analysis.representative_agreement import variable_coordinates
from mm_sae.analysis.mapping_evaluation import paired_retrieval


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False,
        default=lambda v: v.tolist() if isinstance(v,np.ndarray) else v.item()))
    tmp.replace(path)


@torch.no_grad()
def project(w, k, positive):
    if positive:
        w.clamp_(min=0)
    idx = w.abs().topk(k, dim=0).indices
    mask = torch.zeros_like(w).scatter_(0, idx, 1)
    w.mul_(mask)
    # Fix arbitrary column scale; all trained arms use this same constraint.
    w.div_(w.norm(dim=0, keepdim=True).clamp_min(1e-8))


def loss(x, y, a, b):
    scores = F.normalize(x @ a, dim=1) @ F.normalize(y @ b, dim=1).T / .07
    labels = torch.arange(len(x), device=x.device)
    return (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)) / 2


def train(cfg, out, args):
    parent, source = Path(cfg['parent_run']), Path(cfg['source_run'])
    pop = json.loads((parent / 'population.json').read_text())
    with np.load(parent / 'moments.npz') as z:
        ids = {s: z[s+'_ids'].copy() for s in ('image', 'text')}
        mean = z['fit_mean'].astype('float32')
        scale = np.sqrt(np.maximum(z['fit_second'].diagonal()-mean.astype('float64')**2, 1e-12)).astype('float32')
    # Use the library's exact scale convention instead of recomputing it.
    from mm_sae.metrics.regression import Moments
    with np.load(parent / 'moments.npz') as z:
        mom = Moments(int(z['fit_n']), z['fit_mean'], z['fit_second'])
        mean, scale = mom.mean.astype('float32'), mom.scale.astype('float32')
    ni = len(ids['image'])
    means = dict(image=mean[:ni], text=mean[ni:])
    scales = dict(image=scale[:ni], text=scale[ni:])
    initial_path = next(m['path'] for m in cfg['models'] if m['key']=='sparse_cca_16')
    with np.load(initial_path) as z:
        initial = {s:z[s].astype('float32') for s in ids}
    assert initial['image'].shape[1] == 256
    records = json.loads((source/'index/train2017/images.json').read_text())
    image_ids = np.array([r['image_id'] for r in records])
    fit_rows = np.flatnonzero(np.isin(image_ids, pop['fit_image_ids']))
    tune_rows = np.flatnonzero(np.isin(image_ids, pop['tune_image_ids']))
    assert not np.intersect1d(fit_rows, tune_rows).size
    parents = np.load(source/'index/train2017/parents.npy')
    order = np.argsort(parents, kind='stable')
    counts = np.bincount(parents, minlength=len(image_ids))
    starts = np.r_[0, counts.cumsum()[:-1]]
    assert (counts[fit_rows] > 0).all()
    raw = {s:sparse.load_npz(source/'activations/train2017'/f'{s}.npz').tocsr()[:,ids[s]] for s in ids}
    # Fixed tuning images and captions, disjoint from all gradient updates.
    rng = np.random.default_rng(738)
    tune_rows = rng.choice(tune_rows, min(2048, len(tune_rows)), replace=False)
    tune_text = order[starts[tune_rows]]
    tune_base = {s:torch.as_tensor(raw[s][rows].toarray()/scales[s], device='cuda')
                 for s,rows in [('image',tune_rows),('text',tune_text)]}
    protocol = dict(config=vars(args), source=cfg, fit_images=len(fit_rows),
        fit_available_caption_pairs=int(counts[fit_rows].sum()), tune_images=len(tune_rows),
        dimensions=256, support=16, pairing='P=I; no partial matching learned',
        sampler='Each fit image appears once per epoch with one uniformly sampled associated caption; no duplicate image within a batch.',
        objective='Symmetric image-caption InfoNCE, cosine, temperature 0.07',
        optimizer='Adam, projected to 16 entries and unit L2 norm per column after every update',
        checkpoint='Lowest fixed tune InfoNCE, never selected by val2017 or category labels',
        scope='One seed exploratory warm-start pilot; not a convergence or semantic-identifiability guarantee',
        labels='No category annotations are loaded before training completes')
    save(out/'protocol.json', protocol)
    np.savez_compressed(out/'split.npz', fit_image_ids=image_ids[fit_rows], tune_image_ids=image_ids[tune_rows])
    variants = [('nce_centered', True, False), ('nce_uncentered', False, False), ('nce_nonnegative', False, True)]
    for name, center, positive in variants:
        if (out/f'{name}.npz').exists():
            continue
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        a = torch.nn.Parameter(torch.tensor(initial['image'], device='cuda'))
        b = torch.nn.Parameter(torch.tensor(initial['text'], device='cuda'))
        if positive:
            with torch.no_grad():
                a.abs_(); b.abs_()
        for w in (a,b): project(w, 16, positive)
        support0 = [w.detach().ne(0).clone() for w in (a,b)]
        offsets = {s:torch.as_tensor(means[s]/scales[s] if center else np.zeros_like(means[s]), device='cuda') for s in ids}
        tune = {s:tune_base[s]-offsets[s] for s in ids}
        opt = torch.optim.Adam([a,b], lr=args.lr)
        history, best, best_epoch, stale = [], float('inf'), 0, 0
        started = time.monotonic()
        for epoch in range(args.epochs+1):
            epoch_start = time.monotonic()
            losses = []
            if epoch:
                sequence = rng.permutation(fit_rows)
                for start in range(0, len(sequence), args.batch):
                    rows = sequence[start:start+args.batch]
                    if len(rows)<2: continue
                    tr = order[starts[rows]+(rng.random(len(rows))*counts[rows]).astype(int)]
                    x = torch.as_tensor(raw['image'][rows].toarray()/scales['image'], device='cuda')-offsets['image']
                    y = torch.as_tensor(raw['text'][tr].toarray()/scales['text'], device='cuda')-offsets['text']
                    opt.zero_grad(set_to_none=True)
                    value = loss(x,y,a,b)
                    if not torch.isfinite(value): raise ValueError('Nonfinite training loss')
                    value.backward()
                    torch.nn.utils.clip_grad_norm_([a,b], 5.)
                    opt.step()
                    for w in (a,b): project(w,16,positive)
                    losses.append(float(value.detach()))
            with torch.no_grad():
                vl = np.mean([float(loss(tune['image'][i:i+args.batch], tune['text'][i:i+args.batch], a,b))
                              for i in range(0,len(tune_rows),args.batch)])
            row = dict(epoch=epoch, train_loss=float(np.mean(losses)) if losses else None,
                       tune_loss=float(vl), epoch_seconds=time.monotonic()-epoch_start,
                       elapsed_seconds=time.monotonic()-started,
                       changed_support_entries=[int((w.ne(0)!=s).sum()) for w,s in zip((a,b),support0)])
            history.append(row)
            if vl < best-1e-5:
                best, best_epoch, stale = vl, epoch, 0
                best_weights = [w.detach().cpu().numpy().copy() for w in (a,b)]
            else: stale += 1
            save(out/f'{name}-history.json', history)
            save(out/'progress.json', dict(state='training',method=name,**row, max_epochs=args.epochs))
            print(json.dumps(dict(method=name,**row)), flush=True)
            if epoch and stale >= 5: break
        assert all(np.max(np.count_nonzero(w,axis=0))<=16 for w in best_weights)
        if positive: assert all(np.min(w)>=0 for w in best_weights)
        np.savez_compressed(out/f'{name}.npz', image=best_weights[0],text=best_weights[1],
                            centered=center,image_ids=ids['image'],text_ids=ids['text'])
        save(out/f'{name}-fit.json',dict(best_epoch=best_epoch, best_tune_loss=best,
             completed_epoch=epoch, early_stopped=stale>=5, support_limit=16, nonnegative=positive))


def evaluate_all(cfg,out):
    data = load_population(cfg,'test')
    src = Path(cfg['source_run'])
    parents = np.load(src/'index/val2017/parents.npy')
    baseline = next(m['path'] for m in cfg['models'] if m['key']=='sparse_cca_16')
    with np.load(baseline) as z:
        normalized={s:z[s]/np.maximum(np.linalg.norm(z[s],axis=0,keepdims=True),1e-12) for s in ('image','text')}
    np.savez_compressed(out/'sparse_cca_unitnorm.npz',**normalized)
    jobs = [('sparse_cca_16', Path(baseline), True),
            ('sparse_cca_unitnorm',out/'sparse_cca_unitnorm.npz',True)] + [(n,out/f'{n}.npz',c) for n,c in
        [('nce_centered',True),('nce_uncentered',False),('nce_nonnegative',False)]]
    for name,path,center in jobs:
        if (out/f'{name}-evaluation.json').exists(): continue
        save(out/'progress.json',dict(state='evaluating',method=name))
        with np.load(path) as z: weights={s:z[s].copy() for s in ('image','text')}
        scores={s:np.asarray(data['raw'][s]@(weights[s]/data['scales'][s][:,None])) for s in weights}
        if center:
            for s in scores: scores[s]-=data['means'][s]@(weights[s]/data['scales'][s][:,None])
        result = dict(retrieval=paired_retrieval(scores['image'],scores['text'],parents,device='cuda',chunk_size=256))
        arrays=dict(eligible=data['eligible'],category_ids=np.array([c['id'] for c in data['concepts']]))
        for s in scores:
            for suffix in ('a','b'):
                key=s+'_'+suffix
                block=scores[s][data['masks'][key]]
                arrays[key]=auc_matrix(block,data['labels'][s][data['masks'][key]],chunk_size=32)
                arrays[key+'_variable']=variable_coordinates(block)
        np.savez_compressed(out/f'{name}-auc.npz',**arrays)
        result['signed_agreement']=evaluation(arrays,cfg,signed=True)
        result['positive_agreement']=evaluation(arrays,cfg,signed=False)
        # Collision diagnostic is descriptive, not a new semantic-accuracy metric.
        collisions={}
        for key in ('image_a','text_b'):
            auc=arrays[key].copy()
            auc[~arrays[key+'_variable']]=-np.inf
            representatives=np.argmax(auc[:,data['eligible']],axis=0)
            counts=Counter(map(int,representatives))
            collisions[key]=dict(distinct_representatives=len(counts),eligible_categories=int(data['eligible'].sum()),
                categories_in_shared_representatives=sum(v for v in counts.values() if v>1),
                largest_multiplicity=max(counts.values()),coordinate_counts=dict(counts))
        result['positive_representative_reuse']=collisions
        result['protocol']='COCO val2017 retrieval; 80 object labels used only for independent-half representative evaluation. Captions inherit parent-image labels. Reuse is not proof of polysemanticity.'
        save(out/f'{name}-evaluation.json', result)
        print(json.dumps(dict(evaluated=name,recall={k:v['recall'] for k,v in result['retrieval'].items()},agreement=result['positive_agreement']['summary']['agree_at1_count'])),flush=True)
    save(out/'progress.json',dict(state='completed'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=str,required=True)
    p.add_argument('--output',type=str,required=True)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch',type=int,default=512)
    p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--seed',type=int,default=0)
    args=p.parse_args()
    cfg=load_config(Path(args.config))
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    try:
        train(cfg,out,args)
        evaluate_all(cfg,out)
    except Exception as e:
        save(out/'progress.json',dict(state='failed',error=repr(e)))
        raise
