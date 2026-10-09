"""Ridge penalty sweep and fixed-support CE probes, using only fit annotations."""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from experiments.oracle_sets.run import load_fit, load_test, test_semantics
from experiments.mapping_semantics.run_support_sweep import compact_retrieval
from mm_sae.analysis.concept_sets import fit_concept_sets
from mm_sae.analysis.concept_losses import fit_fixed_support
from mm_sae.analysis.concept_supervision import standardize_concept_weights, project_concepts
from mm_sae.analysis.data import CachedSplit
from mm_sae.analysis.mapping_evaluation import paired_retrieval
from mm_sae.io import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    root = (args.config.parent / config['output']).resolve()
    torch.manual_seed(0)
    for condition in config['conditions']:
        cfg = {**config, **condition, 'label_targets': ['propagated_presence']}
        for key in ('source_run', 'parent_run', 'reference_run'):
            cfg[key] = str((args.config.parent / cfg[key]).resolve())
        out = root / condition['name']
        out.mkdir(parents=True, exist_ok=True)
        def status(state, **extra):
            record = dict(condition=condition['name'], state=state, updated_at=time.time(), **extra)
            atomic_json(out/'progress.json', record)
            print(json.dumps(record), flush=True)
        status('loading_fit_annotations')
        data = load_fit(cfg)
        # Persist exact fit statistics so later diagnostics need no training scan.
        for side in ('image', 'text'):
            target = 'image' if side == 'image' else 'propagated_presence'
            np.savez_compressed(out/f'{side}_moments.npz', **data['stats'][side],
                                label_mean=data['moments'][target]['label_mean'],
                                label_variance=data['moments'][target]['label_variance'],
                                cross=data['moments'][target]['cross'])
        atomic_json(out/'protocol.json', dict(config=cfg, fit_images=data['fit_image_count'],
                    fit_captions=data['fit_caption_count'], concepts=data['concepts'],
                    population=data['population'], supports=16, outputs=171,
                    test_labels_for_fitting=False, output_scaling='fit_unit_variance_without_intercept',
                    code_sha256=sha256(Path(__file__)),
                    loss_code_sha256=sha256(Path('src/mm_sae/analysis/concept_losses.py')),
                    inputs={str(Path(cfg['parent_run'])/'moments.npz'):
                            sha256(Path(cfg['parent_run'])/'moments.npz')}))
        for penalty in cfg['ridge_penalties']:
            key = f'ridge_{penalty:g}'
            for side in ('image', 'text'):
                dest = out/f'{key}_{side}.npz'
                if dest.exists():
                    continue
                target = 'image' if side == 'image' else 'propagated_presence'
                m, stats = data['moments'][target], data['stats'][side]
                status('fitting_ridge', method=key, side=side)
                fit = fit_concept_sets(stats['cov'], m['cross'], m['label_mean'], m['label_variance'],
                                      budgets=[16], penalty=penalty)
                raw = fit.coefficients[16]
                if penalty == .01:
                    ref = 'image' if side == 'image' else 'propagated_presence'
                    with np.load(Path(cfg['reference_run'])/'fits'/f'{ref}_16.npz') as z:
                        np.testing.assert_allclose(raw, z['raw_coefficients'], rtol=1e-6, atol=1e-8)
                coef, scale, active = standardize_concept_weights(raw, stats['cov'])
                np.savez_compressed(dest, raw=raw, coefficients=coef, scale=scale, active=active)
                atomic_json(out/f'{key}_{side}.json', dict(penalty=penalty,
                            support_selection='forward greedy with refitting at this penalty',
                            max_inputs=int(np.count_nonzero(raw, axis=0).max())))
                del fit
        train = CachedSplit(Path(cfg['source_run']), 'train2017')
        keep = np.isin(train.image_ids, data['population']['fit_image_ids'])
        caption_keep = keep[train.parents]
        for side in ('image', 'text'):
            rows = keep if side == 'image' else caption_keep
            raw = train.activations[side][rows][:, data['ids'][side]]
            labels = np.asarray(train.presence[keep] if side == 'image'
                                else train.presence[train.parents[caption_keep]], dtype=np.float32)
            weights = (np.bincount(train.parents[caption_keep], minlength=len(keep))[keep]
                       if side == 'image' else np.ones(int(caption_keep.sum())))
            with np.load(out/f'ridge_0.01_{side}.npz') as z:
                support = z['raw']
            stats = data['stats'][side]
            for kind in cfg['ce_losses']:
                dest = out/f'{kind}_{side}.npz'
                if dest.exists():
                    continue
                started = time.monotonic()
                status('fitting_ce', method=kind, side=side)
                def progress(record):
                    status('fitting_ce', method=kind, side=side,
                           elapsed_seconds=time.monotonic()-started, **record)
                fitted, audit = fit_fixed_support(raw, labels, weights, stats['mean'], stats['scale'],
                    support, kind=kind, penalty=cfg['ce_penalty'], max_iter=cfg['max_iter'],
                    device=cfg['device'], progress=progress)
                coef, scale, active = standardize_concept_weights(fitted, stats['cov'])
                assert (np.count_nonzero(fitted, axis=0) <= 16).all()
                np.savez_compressed(dest, raw=fitted, coefficients=coef, scale=scale, active=active)
                audit['seconds'] = time.monotonic()-started
                atomic_json(out/f'{kind}_{side}.json', audit)
                gc.collect()
                torch.cuda.empty_cache()
        del train, raw, labels, weights
        gc.collect()
        status('loading_evaluation')
        test = load_test(cfg, data)
        for key in [f'ridge_{p:g}' for p in cfg['ridge_penalties']] + cfg['ce_losses']:
            dest = out/f'{key}_result.json'
            if dest.exists():
                continue
            status('evaluating', method=key)
            scores = {}
            for side in ('image', 'text'):
                with np.load(out/f'{key}_{side}.npz') as z:
                    coef = z['coefficients']
                stats = data['stats'][side]
                scores[side] = project_concepts(test['raw'][side], stats['mean'], stats['scale'], coef)
            metrics = paired_retrieval(scores['image'], scores['text'], test['parents'],
                                       device=cfg['device'], chunk_size=cfg['retrieval_chunk'])
            semantic = test_semantics(data, test, scores['image'], scores['text'], 'propagated_presence')
            # A class with no positive or no negative examples has undefined AUROC.
            for record in semantic:
                for field, value in record.items():
                    if field.endswith('_auc') and value is not None and not np.isfinite(value):
                        record[field] = None
            atomic_json(dest, dict(key=key, retrieval=compact_retrieval(metrics), semantic=semantic))
        status('completed')
        del data, test, scores
        gc.collect()


if __name__ == '__main__':
    main()
