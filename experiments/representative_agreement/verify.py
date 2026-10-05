"""Independently audit representative choices and ranks from saved AUROC arrays.

This verifier does not import or call the experiment's agreement evaluator.
It recomputes the deterministic tie policy, signs, ranks, strict and tie-aware
agreement, and available within-modality repeat checks from the saved arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _equal(observed, expected, label: str) -> None:
    if expected is None:
        assert observed is None, (label, observed, expected)
    elif isinstance(expected, (float, np.floating)):
        assert observed is not None and np.isclose(observed, expected, rtol=0, atol=2e-14), (
            label, observed, expected)
    else:
        assert observed == expected, (label, observed, expected)


def _broadcast(mask, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask)
    assert mask.dtype == bool
    if mask.shape == (shape[0],):
        mask = np.repeat(mask[:, None], shape[1], axis=1)
    assert mask.shape == shape
    return mask


def _choose(raw, usable, signed: bool, tolerance: float) -> dict[str, Any]:
    ids = [int(x) for x in np.flatnonzero(usable & np.isfinite(raw))]
    score = {j: float(max(raw[j], 1 - raw[j]) if signed else raw[j]) for j in ids}
    polarity = {j: -1 if signed and raw[j] < .5 else 1 for j in ids}
    # Group near-equal scores at the descending group's largest value. Select
    # increasing native coordinate IDs inside each group, without rounding.
    remaining = sorted(ids, key=lambda j: (-score[j], j))
    ordered: list[int] = []
    first: list[int] = []
    while remaining:
        peak = score[remaining[0]]
        count = next((i for i, j in enumerate(remaining) if peak - score[j] > tolerance), len(remaining))
        group = sorted(remaining[:count])
        if not ordered:
            first = group
        ordered.extend(group)
        remaining = remaining[count:]
    chosen = ordered[0] if ordered else None
    return dict(coordinate=chosen, order=ordered, ties=first, score=score, polarity=polarity)


def _fraction(total: int, count: int) -> float | None:
    return total / count if count else None


def _verify_metric(result: dict[str, Any], arrays: dict[str, np.ndarray], label: str) -> dict[str, Any]:
    image, text = arrays['image_a'], arrays['text_b']
    assert image.ndim == 2 and image.shape == text.shape
    shape = image.shape
    iv = _broadcast(arrays['image_a_variable'], shape)
    tv = _broadcast(arrays['text_b_variable'], shape)
    joint = iv & tv & np.isfinite(image) & np.isfinite(text)
    eligible = arrays['eligible'].astype(bool)
    assert eligible.shape == (shape[1],)
    signed = result['protocol']['signed']
    tolerance = result['protocol']['tie_tolerance']
    assert isinstance(signed, bool) and tolerance >= 0
    rows = result['per_category']
    assert len(rows) == shape[1]
    pool_size = int(np.count_nonzero(np.any(iv & tv, axis=1)))
    _equal(result['n_coordinates'], shape[0], label + '.n_coordinates')
    _equal(result['n_categories'], shape[1], label + '.n_categories')
    _equal(result['n_candidate_coordinates'], pool_size, label + '.candidates')
    _equal(result['n_eligible_categories'], int(eligible.sum()), label + '.eligible')
    records = []
    for c, row in enumerate(rows):
        _equal(row['category_index'], c, label + '.category_index')
        _equal(row['n_candidate_coordinates'], int(joint[:, c].sum()), label + '.category_candidates')
        if not eligible[c] or not joint[:, c].any():
            expected = 'excluded_by_caller' if not eligible[c] else 'no_valid_paired_coordinates'
            _equal(row['status'], expected, label + '.status')
            assert row['image_coordinate'] is None and row['text_coordinate'] is None
            continue
        _equal(row['status'], 'ok', label + '.status')
        selections = [_choose(a[:, c], joint[:, c], signed, tolerance) for a in (image, text)]
        record: dict[str, Any] = dict(category=c)
        for side, a, selected in zip(('image', 'text'), (image, text), selections, strict=True):
            j = selected['coordinate']
            assert j is not None
            sign = selected['polarity'][j]
            _equal(row[side + '_coordinate'], j, label + '.' + side + '_coordinate')
            _equal(row[side + '_sign'], sign, label + '.' + side + '_sign')
            _equal(row[side + '_auc'], selected['score'][j], label + '.' + side + '_auc')
            _equal(row[side + '_raw_auc'], float(a[j, c]), label + '.' + side + '_raw_auc')
            _equal(row[side + '_tie_count'], len(selected['ties']), label + '.' + side + '_ties')
            record[side] = (j, sign)
        agree = record['image'] == record['text']
        _equal(row['agree_at1'], agree, label + '.agree_at1')
        record['agree'] = agree
        for source, target, selected_source, selected_target, target_auc in (
            ('image', 'text', selections[0], selections[1], text),
            ('text', 'image', selections[1], selections[0], image),
        ):
            j, sign = record[source]
            polarity = sign == selected_target['polarity'][j]
            rank = selected_target['order'].index(j) + 1 if polarity else None
            in_tie = polarity and j in selected_target['ties']
            _equal(row['rank_in_' + target], rank, label + '.rank_in_' + target)
            _equal(row['polarity_match_in_' + target], polarity, label + '.polarity_' + target)
            _equal(row[source + '_pick_in_' + target + '_top_tie'], in_tie, label + '.top_tie')
            oriented_auc = target_auc[j, c] if sign > 0 else 1 - target_auc[j, c]
            _equal(row[source + '_pick_in_' + target + '_auc_with_' + source + '_sign'],
                   float(oriented_auc), label + '.transferred_auc')
            record['rank_' + target] = rank
            record['tie_' + target] = in_tie
            record['polarity_' + target] = polarity
        records.append(record)
    n = len(records)
    _equal(result['n_evaluated_categories'], n, label + '.evaluated')
    hits = sum(row['agree'] for row in records)
    summary = result['summary']
    _equal(summary['agree_at1_count'], hits, label + '.agree_count')
    _equal(summary['agree_at1'], _fraction(hits, n), label + '.agreement')
    for direction, target in [('image_to_text', 'text'), ('text_to_image', 'image')]:
        observed = summary[direction]
        _equal(observed['n_categories'], n, label + '.direction_n')
        for k in (1, 5, 10):
            count = sum(row['rank_' + target] is not None and row['rank_' + target] <= k for row in records)
            _equal(observed[f'top{k}_count'], count, label + '.topk_count')
            _equal(observed[f'top{k}'], _fraction(count, n), label + '.topk')
        ties = sum(row['tie_' + target] for row in records)
        _equal(observed['top1_tie_aware_count'], ties, label + '.tie_count')
        _equal(observed['top1_tie_aware'], _fraction(ties, n), label + '.tie_fraction')
        _equal(observed['opposite_polarity_count'], sum(not row['polarity_' + target] for row in records),
               label + '.opposite_signs')
        _equal(observed['top1_count'], hits, label + '.top1_equals_exact_agreement')
    signs_agree = sum(row['image'][1] == row['text'][1] for row in records)
    expected_null = signs_agree / n / pool_size if n and pool_size else None
    _equal(result['controls']['random_pairing_expected_agreement'], expected_null, label + '.expected_null')
    for side, auc_key, variable_key in [('image', 'image_b', 'image_b_variable'),
                                       ('text', 'text_a', 'text_a_variable')]:
        stability = result['controls'][side + '_split_stability']
        if stability is None:
            continue
        assert auc_key in arrays and variable_key in arrays, (label, 'missing repeat arrays', auc_key, variable_key)
        repeat = arrays[auc_key]
        valid = _broadcast(arrays[variable_key], shape) & joint
        compared = []
        equal = []
        for row in records:
            c = row['category']
            selected = _choose(repeat[:, c], valid[:, c], signed, tolerance)
            j = selected['coordinate']
            if j is not None:
                compared.append(rows[c]['category_id'])
                equal.append((j, selected['polarity'][j]) == row[side])
        _equal(stability['n_categories'], len(compared), label + '.stability_n')
        _equal(stability['category_ids'], compared, label + '.stability_categories')
        _equal(stability['agree_at1_count'], sum(equal), label + '.stability_count')
        _equal(stability['agree_at1'], _fraction(sum(equal), len(equal)), label + '.stability')
        _equal(stability['n_missing_repeat_categories'], n - len(compared), label + '.missing_repeat')
    return dict(label=label, categories_checked=n, coordinates=shape[0], signed=signed, agreement_count=hits)


def _metrics(value, path: str = ''):
    if isinstance(value, dict):
        if value.get('metric') == 'independently_selected_representative_agreement':
            yield path, value
        for key, child in value.items():
            yield from _metrics(child, path + '.' + str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _metrics(child, path + f'[{index}]')


def _audit_populations(run: Path) -> dict[str, np.ndarray]:
    """Rebuild annotation counts and image-group independence from raw indexes."""
    protocol = json.loads((run / 'protocol.json').read_text())
    populations = json.loads((run / 'population.json').read_text())
    cfg = protocol['config']
    fitted = json.loads((Path(cfg['parent_run']) / 'population.json').read_text())
    expected = {}
    for name, population in populations.items():
        index = Path(cfg['source_run']) / 'index' / population['split']
        image_ids = np.array([r['image_id'] for r in json.loads((index / 'images.json').read_text())])
        parents = np.load(index / 'parents.npy')
        category_ids = np.array(json.loads((index / 'concept_ids.json').read_text()))
        objects = category_ids < 91
        assert objects.sum() == 80
        labels = np.load(index / 'presence.npy')[:, objects]
        a, b = (set(population[key]) for key in ('half_a_image_ids', 'half_b_image_ids'))
        assert not (a & b), (name, 'image groups overlap')
        assert a | b == set(fitted[name + '_image_ids']), (name, 'image groups omit or add images')
        assert not ((a | b) & set(fitted['fit_image_ids'])), (name, 'mapping training overlaps evaluation')
        image_masks = {'a': np.isin(image_ids, list(a)), 'b': np.isin(image_ids, list(b))}
        eligible = np.ones(80, dtype=bool)
        for suffix, mask in image_masks.items():
            _equal(int(mask.sum()), population['groups']['image_' + suffix], name + '.image_count')
            caption_mask = mask[parents]
            _equal(int(caption_mask.sum()), population['groups']['text_' + suffix], name + '.caption_count')
            for side, counts, total in [('image', labels[mask].sum(0), int(mask.sum())),
                                        ('text', labels[parents[caption_mask]].sum(0), int(caption_mask.sum()))]:
                key = side + '_' + suffix
                np.testing.assert_array_equal(counts, population['positive_counts'][key])
                if key in ('image_a', 'text_b'):
                    eligible &= (counts >= cfg['min_positive']) & (counts < total)
        _equal(int(eligible.sum()), population['n_eligible_categories'], name + '.eligible_count')
        np.testing.assert_array_equal(category_ids[objects][eligible], population['eligible_category_ids'])
        expected[name] = eligible
    return expected


def verify(run: Path) -> dict[str, Any]:
    checked = []
    hashes = {}
    population_eligible = _audit_populations(run)
    files = sorted((run / 'results').glob('*.json'))
    assert files, 'No result files found'
    for result_path in files:
        result = json.loads(result_path.read_text())
        metrics = list(_metrics(result))
        if not metrics:
            continue
        auc_path = run / 'auc' / (result_path.stem + '.npz')
        with np.load(auc_path) as saved:
            arrays = {key: saved[key] for key in saved.files}
        np.testing.assert_array_equal(arrays['eligible'], population_eligible[result['population']])
        np.testing.assert_array_equal(arrays['category_ids'], [r['category_id'] for r in result['per_category']])
        for where, metric in metrics:
            checked.append(_verify_metric(metric, arrays, result_path.stem + where))
        for path in (result_path, auc_path):
            hashes[str(path.relative_to(run))] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert checked, 'No representative agreement metrics found'
    return dict(status='passed', verification='Independent recomputation from stored AUROC arrays; no agreement evaluator imported.',
                metrics_checked=len(checked), category_records_checked=sum(r['categories_checked'] for r in checked),
                records=checked, input_sha256=hashes,
                verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run.resolve())
    destination = args.run / 'verification.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'metrics_checked', 'category_records_checked')}))


if __name__ == '__main__':
    main()
