"""Compare fixed native correspondence pairs using independent COCO annotations.

No encoder, SAE, projection, or classifier is fitted here. Annotation groups
are split by image identity, so all captions of an image stay in its group.
The test outcome is representative agreement, not source-to-target AUROC.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy import sparse
import yaml

from mm_sae.analysis.representative_agreement import (
    evaluate_representative_agreement,
    variable_coordinates,
)
from mm_sae.analysis.semantic_evaluation import auc_matrix
from mm_sae.io import sha256
from mm_sae.metrics.regression import Moments


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    for key in ('source_run', 'parent_run', 'output'):
        cfg[key] = str((path.parent / cfg[key]).resolve())
    for model in cfg['models']:
        model['path'] = str((path.parent / model['path']).resolve())
    return cfg


def load_population(cfg: dict[str, Any], kind: str) -> dict[str, Any]:
    src, parent = Path(cfg['source_run']), Path(cfg['parent_run'])
    population = json.loads((parent / 'population.json').read_text())
    split = 'train2017' if kind == 'tune' else 'val2017'
    index = src / 'index' / split
    records = json.loads((index / 'images.json').read_text())
    all_ids = np.array([r['image_id'] for r in records], dtype=np.int64)
    keep = np.isin(all_ids, population[kind + '_image_ids'])
    image_rows = np.flatnonzero(keep)
    if set(all_ids[keep]) != set(population[kind + '_image_ids']):
        raise ValueError('Requested population is not fully present in the cache')
    if set(all_ids[keep]) & set(population['fit_image_ids']):
        raise ValueError('Annotation evaluation overlaps correspondence fitting images')
    all_parents = np.load(index / 'parents.npy')
    text_rows = np.flatnonzero(keep[all_parents])
    remap = np.cumsum(keep) - 1
    parents = remap[all_parents[text_rows]]
    ids = json.loads((index / 'concept_ids.json').read_text())
    objects = np.array([i for i, cid in enumerate(ids) if cid < 91])
    if len(objects) != 80:
        raise ValueError('Expected exactly 80 COCO object categories')
    names = {c['id']: c['name'] for c in json.loads((src / 'dataset.json').read_text())['concepts']}
    concepts = [{'id': int(ids[i]), 'name': names[ids[i]]} for i in objects]
    labels = np.load(index / 'presence.npy')[image_rows][:, objects].astype(bool)
    half = np.zeros(len(image_rows), dtype=bool)
    half[np.random.default_rng(cfg['seed']).permutation(len(half))[:len(half) // 2]] = True
    masks = {'image_a': half, 'image_b': ~half, 'text_a': half[parents], 'text_b': ~half[parents]}
    ys = {'image': labels, 'text': labels[parents]}
    counts = {key: ys[key.split('_')[0]][mask].sum(0) for key, mask in masks.items()}
    # The main pairing is image A with text B, as in the original protocol.
    eligible = (counts['image_a'] >= cfg['min_positive']) & (counts['text_b'] >= cfg['min_positive'])
    for key in ('image_a', 'text_b'):
        eligible &= counts[key] < int(masks[key].sum())
    with np.load(parent / 'moments.npz') as z:
        feature_ids = {side: z[side + '_ids'].copy() for side in ('image', 'text')}
        fit = Moments(int(z['fit_n']), z['fit_mean'], z['fit_second'])
    ni = len(feature_ids['image'])
    means = {'image': fit.mean[:ni], 'text': fit.mean[ni:]}
    scales = {'image': fit.scale[:ni], 'text': fit.scale[ni:]}
    activation_files = {side: src / 'activations' / split / (side + '.npz') for side in ('image', 'text')}
    raw = {}
    for side, rows in (('image', image_rows), ('text', text_rows)):
        raw[side] = sparse.load_npz(activation_files[side]).tocsr()[rows][:, feature_ids[side]]
    provenance_files = [parent / 'moments.npz', parent / 'population.json', index / 'presence.npy',
                        index / 'parents.npy', index / 'images.json', index / 'concept_ids.json',
                        *activation_files.values()]
    metadata = {
        'population': kind, 'split': split, 'n_images': len(image_rows), 'n_captions': len(text_rows),
        'n_eligible_categories': int(eligible.sum()), 'seed': cfg['seed'],
        'groups': {key: int(mask.sum()) for key, mask in masks.items()},
        'half_a_image_ids': all_ids[image_rows[half]].tolist(),
        'half_b_image_ids': all_ids[image_rows[~half]].tolist(),
        'concepts': concepts, 'eligible_category_ids': [c['id'] for c, ok in zip(concepts, eligible) if ok],
        'positive_counts': {key: value.tolist() for key, value in counts.items()},
        'source_hashes': {str(path): sha256(path) for path in provenance_files},
    }
    return dict(raw=raw, labels=ys, masks=masks, means=means, scales=scales,
                concepts=concepts, counts=counts, eligible=eligible, metadata=metadata,
                feature_ids=feature_ids)


def load_model(spec: dict[str, Any], data: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[dict[str, int]]]:
    with np.load(spec['path']) as z:
        if spec['kind'] == 'permutation':
            m = z['mapping']
            ii, tt = np.nonzero(m)
            if len(set(ii)) != len(ii) or len(set(tt)) != len(tt):
                raise ValueError('Permutation baseline must have one-to-one nonzero edges')
            wi, wt = np.eye(m.shape[0])[:, ii], np.eye(m.shape[1])[:, tt]
            identities = [dict(coordinate=r, image_feature=int(data['feature_ids']['image'][i]),
                               text_feature=int(data['feature_ids']['text'][t]))
                          for r, (i, t) in enumerate(zip(ii, tt))]
        elif spec['kind'] == 'projection':
            wi, wt = z['image'].copy(), z['text'].copy()
            identities = [dict(coordinate=r) for r in range(wi.shape[1])]
        else:
            raise ValueError('Unknown native correspondence kind')
    for side, weights in (('image', wi), ('text', wt)):
        if weights.shape[0] != len(data['feature_ids'][side]) or not np.isfinite(weights).all():
            raise ValueError('Projection shape or values do not match the frozen activation space')
    if wi.shape[1] != wt.shape[1]:
        raise ValueError('Native correspondence coordinates must be paired')
    return {'image': wi, 'text': wt}, identities


def calculate_aucs(cfg: dict[str, Any], data: dict[str, Any], weights: dict[str, np.ndarray],
                   progress) -> dict[str, np.ndarray]:
    result = {}
    for side in ('image', 'text'):
        scaled = weights[side] / data['scales'][side][:, None]
        # Sparse multiplication avoids allocating a dense original activation table.
        scores = np.asarray(data['raw'][side] @ scaled) - data['means'][side] @ scaled
        for suffix in ('a', 'b'):
            key = side + '_' + suffix
            mask = data['masks'][key]
            progress(key)
            block = scores[mask]
            result[key] = auc_matrix(block, data['labels'][side][mask], chunk_size=cfg['auc_chunk'])
            result[key + '_variable'] = variable_coordinates(block)
        del scores
    result['eligible'] = data['eligible']
    result['category_ids'] = np.array([c['id'] for c in data['concepts']])
    return result


def evaluation(arrays: dict[str, np.ndarray], cfg: dict[str, Any], *, signed: bool) -> dict[str, Any]:
    return evaluate_representative_agreement(
        arrays['image_a'], arrays['text_b'], image_variable=arrays['image_a_variable'],
        text_variable=arrays['text_b_variable'], category_ids=arrays['category_ids'],
        eligible_categories=arrays['eligible'], signed=signed, image_repeat_auc=arrays['image_b'],
        text_repeat_auc=arrays['text_a'], image_repeat_variable=arrays['image_b_variable'],
        text_repeat_variable=arrays['text_a_variable'], n_null=cfg['n_null'], seed=cfg['seed'],
    )


def write_summary(output: Path) -> None:
    rows = []
    for path in sorted((output / 'results').glob('*.json')):
        value = json.loads(path.read_text())
        row = {k: value[k] for k in ('population', 'method', 'label', 'n_candidate_coordinates',
                                    'n_evaluated_categories')}
        row.update({k: value['summary'][k] for k in ('agree_at1', 'agree_at1_count')})
        for direction in ('image_to_text', 'text_to_image'):
            row.update({direction + '_' + k: v for k, v in value['summary'][direction].items()})
        rows.append(row)
    if rows:
        with (output / 'summary.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run(cfg: dict[str, Any]) -> None:
    output = Path(cfg['output'])
    for name in ('auc', 'results'):
        (output / name).mkdir(parents=True, exist_ok=True)
    protocol = {
        'config': cfg, 'concepts': [],
        'annotation': 'Image COCO-Stuff object presence in the encoder crop; captions inherit parent image labels.',
        'selection': 'AUROC for concept presence/absence, independent image A and caption B groups.',
        'candidate_rule': 'Fixed native pairs; nonconstant coordinates on both selection groups; no firing-support cutoff.',
        'sign_rule': 'All methods use max(AUROC,1-AUROC) and require matching independently chosen polarity.',
        'positive_only_sensitivity': 'Also report raw-AUROC selection for all methods without sign reversal.',
        'min_positive': cfg['min_positive'], 'seed': cfg['seed'],
        'eligibility': 'At least min_positive image-A positives and caption-B positives; both negatives present.',
        'scope': 'Representative agreement only; neither exclusive concept purity nor image-caption retrieval recall.',
        'training': 'Existing COCO-trained SAEs and fixed mapping weights; no new fitting.',
        'differences_from_rebuttal': [
            'Current COCO-trained models, not the original CC3M-trained checkpoint.',
            'Current cropped COCO-Stuff labels, not the original salience-filtered label cache.',
            'Nonconstant paired coordinates for every method; the original 5% firing-support filter is not used.',
            'Primary comparison permits signed selection for every method; positive-only sensitivity is also reported.',
            'Exact deterministic representative agreement resolves ties by coordinate index; tie-aware top1 is separate.',
        ],
        'population_notes': {
            'tune': '23,657 COCO train images outside correspondence fitting; SAEs saw train images; earlier mapping selection used this partition without concept annotations.',
            'test': '5,000 COCO val images outside SAE and correspondence fitting; previously used for exploratory analyses.',
        },
    }
    total = len(cfg['models']) * len(cfg['populations'])
    started, completed = time.monotonic(), 0
    population_metadata = {}
    for kind in cfg['populations']:
        print(json.dumps({'stage': 'load', 'population': kind}), flush=True)
        data = load_population(cfg, kind)
        population_metadata[kind] = data['metadata']
        save_json(output / 'population.json', population_metadata)
        protocol['concepts'] = data['concepts']
        save_json(output / 'protocol.json', protocol)
        for spec in cfg['models']:
            key = kind + '__' + spec['key']
            provenance = dict(data['metadata']['source_hashes'])
            provenance[spec['path']] = sha256(Path(spec['path']))
            provenance['configuration'] = json.dumps(cfg, sort_keys=True)
            result_path, cache_path = output / 'results' / (key + '.json'), output / 'auc' / (key + '.npz')

            def progress(substep):
                elapsed = time.monotonic() - started
                state = dict(stage='evaluate', population=kind, method=spec['key'], substep=substep,
                             completed=completed, total=total, elapsed_seconds=round(elapsed, 1),
                             eta_seconds=round(elapsed / completed * (total - completed), 1) if completed else None)
                save_json(output / 'status.json', state)
                print(json.dumps(state), flush=True)

            if result_path.exists():
                previous = json.loads(result_path.read_text())
                if previous['sources'] != provenance:
                    raise ValueError(f'Existing result provenance changed for {key}')
                completed += 1
                continue
            weights, identities = load_model(spec, data)
            arrays = calculate_aucs(cfg, data, weights, progress)
            np.savez_compressed(cache_path, **arrays)
            result = evaluation(arrays, cfg, signed=True)
            result.update(method=spec['key'], label=spec['label'], population=kind,
                          candidate_control=spec.get('candidate_control', False),
                          sources=provenance, coordinate_identities=identities,
                          population_metadata={k: v for k, v in data['metadata'].items()
                                               if k not in ('half_a_image_ids', 'half_b_image_ids', 'source_hashes')})
            for row, concept in zip(result['per_category'], data['concepts']):
                row['name'] = concept['name']
                row['n_positive_image_a'] = int(data['counts']['image_a'][row['category_index']])
                row['n_positive_text_b'] = int(data['counts']['text_b'][row['category_index']])
            result['positive_only_sensitivity'] = evaluation(arrays, cfg, signed=False)
            save_json(result_path, result)
            completed += 1
            progress('complete')
            write_summary(output)
        del data
    save_json(output / 'status.json', dict(stage='complete', completed=completed, total=total,
                                         elapsed_seconds=round(time.monotonic() - started, 1), eta_seconds=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    run(load_config(args.config))
