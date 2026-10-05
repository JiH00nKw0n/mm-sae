"""Fit annotation-guided semantic feature sets, then retrieve without test labels.

The main condition gives every training caption its parent image's COCO labels.
A separate condition uses literal dictionary mentions for caption supervision.
Both predict one shared coordinate per annotation class. This is a supervised
reference, not an oracle assignment of true semantic memberships to SAE units.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import yaml

from mm_sae.analysis.concept_sets import fit_concept_sets
from mm_sae.analysis.concept_supervision import (
    annotation_moments, project_concepts, standardize_concept_weights,
)
from mm_sae.analysis.data import CachedSplit, save_json
from mm_sae.analysis.evaluation import auroc
from mm_sae.analysis.mapping_evaluation import paired_retrieval
from mm_sae.io import sha256
from mm_sae.metrics.regression import Moments
from mm_sae.progress import ProgressReporter, iter_progress, stage_progress


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    for key in ('source_run', 'parent_run', 'semantics_run', 'pruning_run', 'output'):
        cfg[key] = str((path.parent / cfg[key]).resolve())
    if cfg['output_scaling'] != 'fit_unit_variance':
        raise ValueError('Only training-prediction unit variance is implemented')
    if set(cfg['label_targets']) - {'propagated_presence', 'caption_mentions'}:
        raise ValueError('Unknown caption supervision condition')
    return cfg


def freeze_inputs(cfg):
    source, parent, out = (Path(cfg[k]) for k in ('source_run', 'parent_run', 'output'))
    files = [parent/'population.json', parent/'moments.npz', source/'dataset.json']
    for split in ('train2017', 'val2017'):
        files += [source/'index'/split/n for n in
                  ('parents.npy', 'presence.npy', 'mentions.npy', 'concept_ids.json', 'images.json')]
        files += [source/'activations'/split/(s+'.npz') for s in ('image', 'text')]
    for control in cfg['controls']:
        if control == 'cca_171':
            files.append(Path(cfg['pruning_run'])/'transforms/cca_full.npz')
        else:
            k = int(control.split('_')[1].replace('cca', ''))
            files.append(Path(cfg['semantics_run'])/'sparse-fit'/f'sparse_cca_k{k}.npz')
    files += [Path(cfg['semantics_run'])/'results'/f'{key}.json' for key in cfg['references']]
    root = Path(__file__).resolve().parents[2]
    code = [Path(__file__), root/'src/mm_sae/analysis/concept_sets.py',
            root/'src/mm_sae/analysis/concept_supervision.py',
            root/'src/mm_sae/analysis/mapping_evaluation.py']
    manifest = dict(config=cfg, sources={str(p): sha256(p) for p in files},
                    code={str(p): sha256(p) for p in code})
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    dest = out/'manifest.json'
    if dest.exists() and json.loads(dest.read_text())['signature'] != signature:
        raise ValueError('Inputs or calculation code changed; use a fresh output folder')
    save_json(dest, dict(signature=signature, **manifest))


def load_fit(cfg):
    parent, source = Path(cfg['parent_run']), Path(cfg['source_run'])
    population = json.loads((parent/'population.json').read_text())
    partitions = [set(population[k+'_image_ids']) for k in ('fit', 'tune', 'test')]
    if any(partitions[a] & partitions[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('Image partitions overlap')
    train = CachedSplit(source, 'train2017')
    keep = np.isin(train.image_ids, population['fit_image_ids'])
    if set(train.image_ids[keep]) != partitions[0]:
        raise ValueError('Training activation cache lacks requested images')
    captions = keep[train.parents]
    image_weights = np.bincount(train.parents[captions], minlength=len(keep))[keep]
    with np.load(parent/'moments.npz') as saved:
        ids = {s: saved[s+'_ids'].copy() for s in ('image', 'text')}
        fit = Moments(int(saved['fit_n']), saved['fit_mean'], saved['fit_second'])
    if fit.n != int(captions.sum()) or fit.n != int(image_weights.sum()):
        raise ValueError('Feature normalization and supervision use different fit populations')
    ni = len(ids['image'])
    covariance = fit.standardized(fit)
    blocks = {'image': slice(0, ni), 'text': slice(ni, None)}
    stats, arrays, target = {}, {}, {}
    for side in ('image', 'text'):
        block = blocks[side]
        stats[side] = dict(mean=fit.mean[block], scale=fit.scale[block], cov=covariance[block, block])
        rows = keep if side == 'image' else captions
        arrays[side] = train.activations[side][rows][:, ids[side]].astype(np.float64)
    target['image'] = np.asarray(train.presence[keep], bool)
    target['propagated_presence'] = np.asarray(train.presence[train.parents[captions]], bool)
    target['caption_mentions'] = np.asarray(train.mentions[captions], bool)
    name_lookup = {r['id']: r['name'] for r in json.loads((source/'dataset.json').read_text())['concepts']}
    concepts = [dict(id=int(c), name=name_lookup[c], group='object' if c < 91 else 'background')
                for c in train.ids]
    moments = {}
    for key in iter_progress(['image', *cfg['label_targets']], 'Compute training annotation moments', unit='targets'):
        side = 'image' if key == 'image' else 'text'
        s = stats[side]
        moments[key] = annotation_moments(arrays[side], target[key], s['mean'], s['scale'],
                                          weights=image_weights if side == 'image' else None)
        np.testing.assert_allclose(moments[key]['observed_mean'], s['mean'], atol=2e-9, rtol=2e-7)
    # No evaluation split was loaded while producing training supervision.
    return dict(population=population, ids=ids, stats=stats, moments=moments, concepts=concepts,
                fit_image_count=int(keep.sum()), fit_caption_count=int(captions.sum()))


def fit_sets(cfg, data):
    out = Path(cfg['output'])
    budgets = [*cfg['budgets'], *(['full'] if cfg['include_dense'] else [])]
    for key in iter_progress(['image', *cfg['label_targets']], 'Fit concept feature sets', unit='targets'):
        paths = [out/'fits'/f'{key}_{k}.npz' for k in budgets]
        if all(p.exists() for p in paths):
            continue
        side = 'image' if key == 'image' else 'text'
        moments, stats = data['moments'][key], data['stats'][side]
        start = time.monotonic()
        fit = fit_concept_sets(stats['cov'], moments['cross'], moments['label_mean'],
                              moments['label_variance'], budgets=cfg['budgets'], penalty=cfg['ridge'],
                              min_gain=cfg['min_gain'])
        for k in budgets:
            coef = fit.dense_coefficients if k == 'full' else fit.coefficients[k]
            normalized, scale, active = standardize_concept_weights(coef, stats['cov'])
            np.savez_compressed(out/'fits'/f'{key}_{k}.npz', raw_coefficients=coef,
                                coefficients=normalized, prediction_scale=scale, active=active,
                                feature_ids=data['ids'][side], feature_mean=stats['mean'],
                                feature_scale=stats['scale'], label_mean=fit.label_mean,
                                label_variance=fit.label_variance)
        save_json(out/'fits'/f'{key}_audit.json', dict(
            selected_features=fit.selected_features, objective_history=fit.objective_history,
            gain_history=fit.gain_history, penalty=fit.penalty,
            observed_mean_difference=float(np.max(np.abs(moments['standardized_mean']))),
            seconds=time.monotonic()-start))
        print(json.dumps(dict(fitted=key, seconds=time.monotonic()-start)), flush=True)


def load_test(cfg, data):
    test = CachedSplit(Path(cfg['source_run']), 'val2017')
    np.testing.assert_array_equal(test.image_ids, data['population']['test_image_ids'])
    if test.ids != [r['id'] for r in data['concepts']]:
        raise ValueError('Concept order differs across training and evaluation')
    return dict(raw={s: test.activations[s][:, data['ids'][s]].toarray().astype(np.float64)
                     for s in ('image', 'text')}, parents=test.parents, image_ids=test.image_ids,
                image_labels=np.asarray(test.presence, bool),
                text_presence=np.asarray(test.presence[test.parents], bool),
                text_mentions=np.asarray(test.mentions, bool))


def test_semantics(data, test, image, text, target):
    chosen = test['text_presence' if target == 'propagated_presence' else 'text_mentions']
    records = []
    for c, concept in enumerate(data['concepts']):
        records.append(dict(**concept, image_auc=auroc(test['image_labels'][:, c], image[:, c]),
                            text_auc=auroc(chosen[:, c], text[:, c]),
                            text_presence_auc=auroc(test['text_presence'][:, c], text[:, c]),
                            text_mention_auc=auroc(test['text_mentions'][:, c], text[:, c]),
                            image_test_positive=int(test['image_labels'][:, c].sum()),
                            text_test_positive=int(chosen[:, c].sum())))
    return records


def conditional_semantics(cfg, data, test, image, text, target):
    names = {r['name']: c for c, r in enumerate(data['concepts'])}
    result = []
    for pair in cfg['conditional_pairs']:
        for name, other in (pair, pair[::-1]):
            c, d = names[name], names[other]
            for side, scores, labels in (
                ('image', image, test['image_labels']),
                ('text', text, test['text_presence' if target == 'propagated_presence' else 'text_mentions']),
            ):
                for present in (False, True):
                    rows = labels[:, d] == present
                    y = labels[rows, c]
                    result.append(dict(name=name, held_concept=other, held_present=present, side=side,
                                       positive_count=int(y.sum()), negative_count=int((~y).sum()),
                                       auroc=auroc(y, scores[rows, c]) if len(y) else None))
    return result


def case_weights(cfg, data, wi, wt):
    cases = []
    for c, concept in enumerate(data['concepts']):
        if concept['name'] not in cfg['case_concepts']:
            continue
        row = dict(concept)
        for side, coef in [('image', wi), ('text', wt)]:
            selected = np.flatnonzero(coef[:, c])
            selected = selected[np.argsort(-np.abs(coef[selected, c]), kind='stable')]
            row[side+'_nonzero'] = len(selected)
            row[side+'_weights'] = [dict(feature=int(data['ids'][side][j]), weight=float(coef[j, c]))
                                   for j in selected]
        cases.append(row)
    return cases


def evaluate_sets(cfg, data, test):
    out = Path(cfg['output'])
    budgets = [*cfg['budgets'], *(['full'] if cfg['include_dense'] else [])]
    for target, k in iter_progress([(t, k) for t in cfg['label_targets'] for k in budgets],
                                    'Evaluate annotation-guided retrieval', unit='conditions'):
        key = f'{target}_{k}'
        dest = out/'results'/f'{key}.json'
        if dest.exists():
            continue
        start = time.monotonic()
        scores, coefs, structures, paths = {}, {}, {}, []
        for side, fit_key in [('image', 'image'), ('text', target)]:
            path = out/'fits'/f'{fit_key}_{k}.npz'
            paths.append(path)
            with np.load(path) as saved:
                b = saved['coefficients']
                scores[side] = project_concepts(test['raw'][side], saved['feature_mean'],
                                                saved['feature_scale'], b)
                coefs[side] = b
                structures[side] = dict(nonzero=int(np.count_nonzero(b)),
                                        max_inputs=int(np.count_nonzero(b, axis=0).max()),
                                        inactive_concepts=int((~saved['active']).sum()))
        retrieval = paired_retrieval(scores['image'], scores['text'], test['parents'],
                                    chunk_size=cfg['retrieval_chunk'], device=cfg['device'])
        semantic = test_semantics(data, test, scores['image'], scores['text'], target)
        conditional = conditional_semantics(cfg, data, test, scores['image'], scores['text'], target)
        label = ('이미지 COCO 주석을 양쪽 학습에 사용' if target == 'propagated_presence'
                 else '이미지 COCO 주석과 텍스트 사전 언급으로 학습')
        label += f', 개념당 특징 {k}개' if k != 'full' else ', 전체 특징 사용'
        save_json(out/'cases'/f'{key}.json', case_weights(cfg, data, coefs['image'], coefs['text']))
        np.savez_compressed(out/'predictions'/f'{key}.npz', image=scores['image'], text=scores['text'])
        save_json(dest, dict(key=key, label=label, label_target=target, k=k, dimensions=len(data['concepts']),
                            retrieval=retrieval, semantic=semantic, conditional=conditional,
                            structure=structures, coefficient_files={str(p): sha256(p) for p in paths},
                            seconds=time.monotonic()-start))
        print(json.dumps(dict(completed=key, seconds=time.monotonic()-start,
                              recall={d: r['recall'] for d, r in retrieval.items()})), flush=True)


def evaluate_controls(cfg, data, test):
    out, sem = Path(cfg['output']), Path(cfg['semantics_run'])
    jobs = [(c, s) for c in cfg['controls'] for s in cfg['control_output_scaling']]
    for base_key, scaling in iter_progress(jobs, 'Evaluate equal-dimension controls', unit='conditions'):
        key = base_key if scaling == 'native' else base_key+'_unit_variance'
        dest = out/'results'/f'{key}.json'
        if dest.exists():
            continue
        start = time.monotonic()
        if base_key == 'cca_171':
            path, k = Path(cfg['pruning_run'])/'transforms/cca_full.npz', 'full'
        else:
            k = int(base_key.split('_')[1].replace('cca', ''))
            path = sem/'sparse-fit'/f'sparse_cca_k{k}.npz'
        scores, structures = {}, {}
        for side in ('image', 'text'):
            with np.load(path) as saved:
                coefficients = saved[side][:, :len(data['concepts'])]
            stats = data['stats'][side]
            if scaling == 'fit_unit_variance':
                coefficients, _, _ = standardize_concept_weights(coefficients, stats['cov'])
            scores[side] = project_concepts(test['raw'][side], stats['mean'], stats['scale'], coefficients)
            structures[side] = dict(nonzero=int(np.count_nonzero(coefficients)),
                                    max_inputs=int(np.count_nonzero(coefficients, axis=0).max()))
        retrieval = paired_retrieval(scores['image'], scores['text'], test['parents'],
                                    chunk_size=cfg['retrieval_chunk'], device=cfg['device'])
        label = 'CCA 171차원' if base_key == 'cca_171' else f'Sparse CCA {k}개, 171차원'
        label += ' · 학습 예측 분산 1로 보정' if scaling == 'fit_unit_variance' else ' · 기존 좌표 크기'
        save_json(dest, dict(key=key, label=label, output_scaling=scaling,
                            label_target='unsupervised', k=k, dimensions=len(data['concepts']),
                            retrieval=retrieval, semantic=[], structure=structures,
                            coefficient_files={str(path): sha256(path)}, seconds=time.monotonic()-start))
        print(json.dumps(dict(completed=key, seconds=time.monotonic()-start,
                              recall={d: r['recall'] for d, r in retrieval.items()})), flush=True)
    for key in cfg['references']:
        path = sem/'results'/f'{key}.json'
        record = json.loads(path.read_text())
        save_json(out/'results'/f'{key}_256.json', dict(
            key=key+'_256', label=record['label']+' · 256차원 참고값', label_target='unsupervised',
            k=record.get('options', {}).get('k') or 'full', dimensions=256,
            retrieval=record['retrieval'], semantic=[], reference={str(path): sha256(path)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg['output'])
    for folder in ('fits', 'results', 'cases', 'predictions'):
        (out/folder).mkdir(parents=True, exist_ok=True)
    stages = ['freeze_inputs', 'training_moments', 'fit_sets', 'evaluate_sets', 'controls']
    with ProgressReporter(out, stages, [], interval=cfg['progress_interval_seconds']):
        with stage_progress('freeze_inputs'):
            freeze_inputs(cfg)
        with stage_progress('training_moments'):
            data = load_fit(cfg)
            protocol = dict(
                expected_conditions=len(cfg['label_targets'])*(len(cfg['budgets'])+int(cfg['include_dense']))
                                    +len(cfg['controls'])*len(cfg['control_output_scaling'])+len(cfg['references']),
                fit_images=data['fit_image_count'], fit_captions=data['fit_caption_count'],
                tune_images=len(data['population']['tune_image_ids']),
                test_images=len(data['population']['test_image_ids']), concepts=data['concepts'],
                budgets=cfg['budgets'], ridge=cfg['ridge'], output_scaling=cfg['output_scaling'],
                fit_method='forward greedy minimization of ridge-penalized mean squared binary-label error',
                labels=dict(image='COCO-Stuff presence in model central crop',
                            propagated_presence='parent image COCO presence assigned to training caption',
                            caption_mentions='reviewed dictionary mentions in each training caption'),
                train_weighting='each caption is one pair; image examples weighted by their caption count',
                target_alignment='coordinate c predicts annotated category c in both modalities',
                selection='only fit partition selects support, weights and normalization; fixed ridge .01',
                tune_usage='reserved; no hyperparameters or budgets selected using tune or test scores',
                evaluation='COCO val2017 has previously been inspected; exploratory held-out validation',
                test_annotations_used_for_retrieval=False, sae_retrained=False,
                interpretation='supervised reference, not perfect latent semantic memberships or performance upper bound')
            save_json(out/'protocol.json', protocol)
            np.savez_compressed(out/'training_moments.npz', **{
                key+'_'+field: value for key, fields in data['moments'].items() for field, value in fields.items()})
        with stage_progress('fit_sets'):
            fit_sets(cfg, data)
        # Evaluation data, including its annotations, are opened only after all
        # supervised feature-set fitting has finished.
        test = load_test(cfg, data)
        protocol['test_captions'] = len(test['parents'])
        save_json(out/'protocol.json', protocol)
        with stage_progress('evaluate_sets'):
            evaluate_sets(cfg, data, test)
        with stage_progress('controls'):
            evaluate_controls(cfg, data, test)


if __name__ == '__main__':
    main()
