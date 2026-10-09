"""Evaluate frozen native correspondence coordinates using independently held labels.

Map fitting uses the existing 80% COCO train partition. Semantic coordinate and
polarity selection uses the disjoint 20% train partition, source modality only.
COCO val2017 is evaluated without any per-method category filtering. Its prior
use in exploratory analyses is explicitly retained in the protocol metadata.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy import sparse
import yaml

from experiments.mapping_ablation.run import product
from mm_sae.analysis.data import save_json
from mm_sae.analysis.annotation_population import load_population
from mm_sae.analysis.mapping_evaluation import paired_retrieval
from mm_sae.analysis.mapping_pruning import coefficient_summary
from mm_sae.analysis.semantic_evaluation import auc_matrix, evaluate_selected_coordinates, select_from_auc
from mm_sae.io import sha256
from mm_sae.metrics.regression import Moments
from mm_sae.progress import ProgressReporter, iter_progress, stage_progress


@dataclass
class Model:
    key: str
    section: str
    family: str
    label: str
    space: str
    image: np.ndarray
    text: np.ndarray
    paths: list[Path]
    reference: Path | None = None
    options: dict[str, Any] | None = None


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    for key in ('source_run', 'parent_run', 'projection_run', 'pruning_run', 'output'):
        cfg[key] = str((Path(path).parent / cfg[key]).resolve())
    if cfg.get('annotation_population'):
        cfg['annotation_population'] = str((Path(path).parent / cfg['annotation_population']).resolve())
    return cfg


def load_data(cfg):
    src, parent = Path(cfg['source_run']), Path(cfg['parent_run'])
    pop = load_population(parent, cfg.get('annotation_population'))
    partitions = [set(pop[k + '_image_ids']) for k in ('fit', 'tune', 'test')]
    if any(partitions[a] & partitions[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('Fit, coordinate calibration, and evaluation images must be disjoint')
    with np.load(parent / 'moments.npz') as saved:
        ids = {side: saved[side + '_ids'].copy() for side in ('image', 'text')}
        fit = Moments(int(saved['fit_n']), saved['fit_mean'], saved['fit_second'])
    means = {'image': fit.mean[:len(ids['image'])], 'text': fit.mean[len(ids['image']):]}
    scales = {'image': fit.scale[:len(ids['image'])], 'text': fit.scale[len(ids['image']):]}
    data: dict[str, Any] = {'ids': ids, 'population': pop, 'fit': fit}
    for kind, split in [('tune', 'train2017'), ('test', 'val2017')]:
        index = src / 'index' / split
        images = json.loads((index / 'images.json').read_text())
        image_ids = np.array([r['image_id'] for r in images])
        parents = np.load(index / 'parents.npy')
        keep = (np.isin(image_ids, pop['tune_image_ids']) if kind == 'tune'
                else np.ones(len(image_ids), bool))
        if kind == 'tune' and set(image_ids[keep]) != partitions[1]:
            raise ValueError('Calibration image IDs are missing from the activation cache')
        assert len(set(pop['fit_image_ids']) & set(image_ids[keep])) == 0
        if kind == 'test':
            np.testing.assert_array_equal(image_ids, pop['test_image_ids'])
        labels = {'image': np.load(index / 'presence.npy'), 'text': np.load(index / 'mentions.npy')}
        arrays, ys = {}, {}
        for side in ('image', 'text'):
            rows = keep if side == 'image' else keep[parents]
            z = sparse.load_npz(src / 'activations' / split / (side + '.npz')).tocsr()
            a = z[rows][:, ids[side]].toarray().astype(np.float64)
            arrays[side] = (a - means[side]) / scales[side]
            ys[side] = labels[side][rows].astype(bool)
        data[kind] = {'x': arrays, 'labels': ys}
        if kind == 'test':
            data['parents'] = parents
            data['images'] = images
            data['captions'] = json.loads((index / 'captions.json').read_text())
    cids = json.loads((src / 'index/val2017/concept_ids.json').read_text())
    assert cids == json.loads((src / 'index/train2017/concept_ids.json').read_text())
    names = {c['id']: c['name'] for c in json.loads((src / 'dataset.json').read_text())['concepts']}
    data['concepts'] = [{'id': int(c), 'name': names[c], 'group': 'object' if c < 91 else 'background'}
                        for c in cids]
    return data


def models(cfg, data, section):
    parent, pruning = Path(cfg['parent_run']), Path(cfg['pruning_run'])
    ni, nt = [len(data['ids'][s]) for s in ('image', 'text')]
    ii, it = np.eye(ni), np.eye(nt)
    result = []

    def direct(key, family, label, m, paths, normalized=False, reference_prefix=None, options=None):
        row, col = m.sum(1), m.sum(0)
        a = np.divide(m, row[:, None], out=np.zeros_like(m), where=row[:, None] > 0) if normalized else m
        b = np.divide(m, col[None, :], out=np.zeros_like(m), where=col[None, :] > 0) if normalized else m
        for space, wi, wt in [('image', ii, a.T), ('text', b, it)]:
            reference = None
            if reference_prefix:
                suffix = 'text_projected_to_image' if space == 'image' else 'image_projected_to_text'
                reference = Path(str(reference_prefix) + '__' + suffix + '.json')
            if family == 'procrustes' and space == ('image' if ni >= nt else 'text'):
                reference = pruning / 'results' / (key + '.json')
            result.append(Model(key + '__' + space, 'baseline', family, label, space, wi, wt, paths,
                                reference, options))

    if section in ('baseline', 'all'):
        selected = json.loads((parent / 'selected.json').read_text())
        chosen = next(r for r in selected if r['family'] == 'hungarian')
        path = parent / 'candidates' / (chosen['key'] + '.npz')
        with np.load(path) as saved:
            m = saved['mapping']
        direct('hungarian', 'hungarian', '헝가리안 일대일 대응', m, [path], True,
               Path(cfg['projection_run']) / 'results/hungarian__standardized')
        for k in [None, *cfg['baseline_k']]:
            tag = 'full' if k is None else str(k)
            for family in ('cca', 'procrustes'):
                path = pruning / 'transforms' / f'{family}_{tag}.npz'
                with np.load(path) as saved:
                    if family == 'cca':
                        wi, wt = saved['image'], saved['text']
                        label = 'CCA 전체 계수' if k is None else f'CCA 학습 후 한 좌표당 최대 {k}개 계수 유지'
                        result.append(Model('cca_' + tag, 'baseline', family, label, 'common', wi, wt,
                                            [path], pruning / 'results' / f'cca_{tag}.json', {'k': k}))
                    else:
                        label = 'Procrustes 전체 계수' if k is None else f'Procrustes 특징당 최대 {k}개 연결 유지'
                        direct('procrustes_' + tag, family, label, saved['image_by_text'], [path], options={'k': k})
            ep = cfg['sinkhorn_epsilon']
            key = f'sinkhorn_e{ep}_k{tag}'
            path = pruning / 'sinkhorn/transforms' / (key + '.npz')
            with np.load(path) as saved:
                m = saved['mapping']
            label = f'Sinkhorn ε={ep} 전체 연결' if k is None else f'Sinkhorn ε={ep}, 특징당 최대 {k}개 연결 유지'
            direct(key, 'sinkhorn', label, m, [path], True,
                   pruning / 'sinkhorn/results' / key, {'epsilon': ep, 'k': k})
    if section in ('sign', 'all'):
        for k in cfg['sign_k']:
            for constraint in ('signed', 'nonnegative'):
                key = f'procrustes_support_{k}_{constraint}'
                path = Path(cfg['output']) / 'sign-fit' / (key + '.npz')
                with np.load(path) as saved:
                    t2i, i2t = saved['text_to_image'], saved['image_to_text']
                label = f'연결 상한 {k}개를 고정한 회귀, ' + ('음수 허용' if constraint == 'signed' else '양수만 허용')
                for space, wi, wt in [('image', ii, t2i), ('text', i2t, it)]:
                    result.append(Model(key + '__' + space, 'sign', constraint, label, space, wi, wt,
                                        [path, path.with_suffix('.json')], options={'k': k, 'constraint': constraint}))
    if section in ('sparse', 'all'):
        for k in cfg['sparse_k']:
            control_path = pruning / 'transforms' / f'cca_{k}.npz'
            with np.load(control_path) as saved:
                ai, at = saved['image'], saved['text']
            cov = data['fit'].standardized(data['fit'])
            xx, yy = cov[:ni, :ni] + .01 * ii, cov[ni:, ni:] + .01 * it
            ai = ai / np.sqrt(np.sum(ai * product(xx, ai), axis=0))
            at = at / np.sqrt(np.sum(at * product(yy, at), axis=0))
            result.append(Model(f'cca_pruned_renormalized_{k}', 'sparse', 'cca_renormalized',
                                f'CCA 학습 후 최대 {k}개 유지, 좌표 크기 재정규화', 'common', ai, at,
                                [control_path], options={'k': k, 'ridge_variance_normalization': .01}))
            path = Path(cfg['output']) / 'sparse-fit' / f'sparse_cca_k{k}.npz'
            diagnostics = json.loads(path.with_suffix('.json').read_text())
            if diagnostics['converged_components'] != len(diagnostics['components']):
                raise ValueError(f'Sparse CCA did not converge in every component: {path}')
            with np.load(path) as saved:
                wi, wt = saved['image'], saved['text']
            result.append(Model(f'sparse_cca_{k}', 'sparse', 'sparse_cca',
                                f'Sparse CCA 학습부터 한 좌표당 최대 {k}개 특징 사용', 'common', wi, wt,
                                [path, path.with_suffix('.json')], options={'k': k}))
    return result


def project(values, coefficients):
    if coefficients.shape[0] == coefficients.shape[1] and np.array_equal(coefficients, np.eye(len(coefficients))):
        return values
    return product(values, coefficients)


def calibration(cfg, data, side, coefficients):
    key = hashlib.sha256(coefficients.tobytes()).hexdigest()
    dest = Path(cfg['output']) / 'calibration' / f'{side}_{key}.npz'
    if dest.exists():
        with np.load(dest) as saved:
            return saved['auc']
    print(json.dumps({'stage': 'calibrate', 'side': side, 'coordinates': coefficients.shape[1]}), flush=True)
    scores = project(data['tune']['x'][side], coefficients)
    auc = auc_matrix(scores, data['tune']['labels'][side], chunk_size=cfg['auc_chunk'])
    np.savez_compressed(dest, auc=auc)
    return auc


def examples(cfg, data, model, values, selections):
    result = []
    for source in ('image', 'text'):
        for c, concept in enumerate(data['concepts']):
            if concept['name'] not in cfg['case_concepts']:
                continue
            sel = selections[source][c]
            if sel['coordinate'] is None:
                continue
            j, sign = sel['coordinate'], sel['sign']
            record = {'source': source, **concept, 'coordinate': j, 'sign': sign}
            for side in ('image', 'text'):
                coefficient = getattr(model, side)[:, j] * sign
                order = np.argsort(-np.abs(coefficient), kind='stable')
                record[side + '_feature_count'] = int(np.count_nonzero(coefficient))
                record[side + '_weights'] = [{'feature': int(data['ids'][side][f]), 'weight': float(coefficient[f])}
                                            for f in order[:16] if coefficient[f] != 0]
                best = np.argsort(-sign * values[side][:, j], kind='stable')[:cfg['case_examples']]
                rows = []
                for row in best:
                    row = int(row)
                    entry = {'row': row, 'score': float(sign * values[side][row, j]),
                             'positive': bool(data['test']['labels'][side][row, c])}
                    if side == 'image':
                        entry['image_id'] = data['images'][row]['image_id']
                        entry['image_path'] = str(Path(__file__).resolve().parents[2] / 'data/coco/images/val2017'
                                                  / f"{entry['image_id']:012d}.jpg")
                    else:
                        entry['text'] = data['captions'][row]['text']
                        entry['image_id'] = data['images'][data['parents'][row]]['image_id']
                    rows.append(entry)
                record[side + '_examples'] = rows
            result.append(record)
    return result


def evaluate_model(cfg, data, model):
    out = Path(cfg['output'])
    result_path = out / 'results' / (model.key + '.json')
    files = {str(p): sha256(p) for p in model.paths}
    if model.reference is not None and model.reference.exists():
        files[str(model.reference)] = sha256(model.reference)
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous['sources'] != files:
            raise ValueError(f'Model changed: {model.key}')
        return
    start = time.monotonic()
    selections = {side: select_from_auc(calibration(cfg, data, side, getattr(model, side)))
                  for side in ('image', 'text')}
    values = {s: project(data['test']['x'][s], getattr(model, s)) for s in ('image', 'text')}
    semantic = []
    for source, target in [('image', 'text'), ('text', 'image')]:
        records = evaluate_selected_coordinates(selections[source], values[source], data['test']['labels'][source],
                                                values[target], data['test']['labels'][target])
        for concept, record in zip(data['concepts'], records, strict=True):
            semantic.append({**concept, 'direction': source + '_to_' + target, **record})
    if model.reference is not None and model.reference.exists():
        retrieval = json.loads(model.reference.read_text())['retrieval']
        retrieval_source = str(model.reference)
    else:
        print(json.dumps({'stage': 'retrieval', 'model': model.key}), flush=True)
        retrieval = paired_retrieval(values['image'], values['text'], data['parents'],
                                    chunk_size=cfg['retrieval_chunk'], device=cfg['device'])
        retrieval_source = 'computed from frozen projected val2017 scores'
    for metric in retrieval.values():
        for k, v in metric['recall'].items():
            np.testing.assert_allclose(v, np.mean(np.asarray(metric['ranks']) <= min(int(k), metric['candidate_count'])))
    cases = examples(cfg, data, model, values, selections)
    save_json(out / 'cases' / (model.key + '.json'), cases)
    structures = {s: coefficient_summary(getattr(model, s)) for s in ('image', 'text')}
    save_json(result_path, {'key': model.key, 'section': model.section, 'family': model.family,
                           'label': model.label, 'space': model.space, 'options': model.options,
                           'structure': structures, 'sources': files, 'semantic': semantic,
                           'retrieval': retrieval, 'retrieval_source': retrieval_source,
                           'seconds': time.monotonic() - start})
    print(json.dumps({'completed': model.key, 'seconds': time.monotonic() - start}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--section', choices=['baseline', 'sign', 'sparse', 'all'], default='all')
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg['output'])
    for folder in ['results', 'cases', 'calibration']:
        (out / folder).mkdir(parents=True, exist_ok=True)
    src = Path(cfg['source_run'])
    paths = [Path(cfg['parent_run']) / n for n in ['moments.npz', 'population.json']]
    if cfg.get('annotation_population'):
        paths.append(Path(cfg['annotation_population']))
    for split in ('train2017', 'val2017'):
        paths += [src / 'index' / split / n for n in ['presence.npy', 'mentions.npy', 'parents.npy', 'concept_ids.json', 'images.json', 'captions.json']]
        paths += [src / 'activations' / split / (s + '.npz') for s in ('image', 'text')]
    manifest = {'config': cfg, 'sources': {str(p): sha256(p) for p in paths}}
    destination = out / 'manifest.json'
    if destination.exists() and json.loads(destination.read_text()) != manifest:
        raise ValueError('Source data or configuration changed. Use another output directory.')
    save_json(destination, manifest)
    with ProgressReporter(out, ['load', args.section], [], cfg['progress_interval_seconds']):
        with stage_progress('load'):
            data = load_data(cfg)
            protocol = {'fit_images': data['population']['fit_images'], 'calibration_images': data['population']['tune_images'],
                        'evaluation_images': len(data['images']), 'evaluation_captions': len(data['captions']),
                        'source_only_coordinate_and_polarity_selection': True,
                        'target_polarity_refit': False, 'new_semantic_classifier': False,
                        'image_labels': 'COCO-Stuff cropped image presence',
                        'text_labels': 'caption mentions using existing reviewed dictionary, not manual caption labels',
                        'evaluation_note': 'Exploratory COCO val2017 already examined in earlier analyses, not untouched final test',
                        'concepts': data['concepts']}
            save_json(out / 'protocol.json', protocol)
        with stage_progress(args.section):
            jobs = models(cfg, data, args.section)
            for model in iter_progress(jobs, 'Joint semantic and retrieval evaluation', unit='conditions'):
                evaluate_model(cfg, data, model)


if __name__ == '__main__':
    main()
