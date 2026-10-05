"""Check output provenance, feature budgets and independent scalar predictions."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from experiments.oracle_sets.run import load_config
from mm_sae.analysis.data import CachedSplit, save_json
from mm_sae.io import sha256


def independent_auc(labels, scores):
    labels = np.asarray(labels, bool)
    positive, negative = int(labels.sum()), int((~labels).sum())
    if not positive or not negative:
        return None
    ranks = rankdata(scores, method='average')
    return float((ranks[labels].sum()-positive*(positive+1)/2)/(positive*negative))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg['output'])
    manifest = json.loads((out/'manifest.json').read_text())
    for path, digest in {**manifest['sources'], **manifest['code']}.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f'Source changed after execution: {path}')
    protocol = json.loads((out/'protocol.json').read_text())
    files = sorted((out/'results').glob('*.json'))
    if len(files) != protocol['expected_conditions']:
        raise ValueError('Some evaluation conditions are missing')
    test = CachedSplit(Path(cfg['source_run']), 'val2017')
    class_indices = [j for j, c in enumerate(protocol['concepts'])
                     if c['name'] in ('person', 'dog', 'skis', 'snow', 'grass', 'road', 'sand')]
    checks: dict[str, Any] = dict(result_conditions=0, recall_values=0, scalar_predictions=0, scalar_auc=0,
                  coefficient_budgets=0, inputs_unchanged=True)
    for path in files:
        record = json.loads(path.read_text())
        checks['result_conditions'] += 1
        for metrics in record['retrieval'].values():
            ranks = np.asarray(metrics['ranks'])
            for k, value in metrics['recall'].items():
                np.testing.assert_allclose(value, np.mean(ranks <= min(int(k), metrics['candidate_count'])))
                checks['recall_values'] += 1
        if record['label_target'] == 'unsupervised':
            continue
        target, k = record['label_target'], record['k']
        with np.load(out/'predictions'/f'{record["key"]}.npz') as saved:
            scores = {s: saved[s] for s in ('image', 'text')}
        for side, fit_key in (('image', 'image'), ('text', target)):
            with np.load(out/'fits'/f'{fit_key}_{k}.npz') as saved:
                b, ids = saved['coefficients'], saved['feature_ids']
                mean, scale = saved['feature_mean'], saved['feature_scale']
            if k != 'full':
                assert np.all(np.count_nonzero(b, axis=0) <= k)
            checks['coefficient_budgets'] += 1
            assert np.isfinite(scores[side]).all()
            rows = np.array([0, 1, 20, 100, len(scores[side])-1])
            values = test.activations[side][rows][:, ids].toarray().astype(np.float64)
            for c in class_indices:
                manual = np.sum(((values-mean)/scale)*b[:, c], axis=1)
                np.testing.assert_allclose(manual, scores[side][rows, c], rtol=1e-11, atol=1e-11)
                checks['scalar_predictions'] += len(rows)
                labels = (test.presence if side == 'image' else test.presence[test.parents]
                          if target == 'propagated_presence' else test.mentions)
                value = independent_auc(labels[:, c], scores[side][:, c])
                expected = record['semantic'][c][side+'_auc']
                if value is None:
                    assert expected is None
                else:
                    np.testing.assert_allclose(value, expected, atol=1e-12)
                checks['scalar_auc'] += 1
    checks['verification_code_sha256'] = sha256(Path(__file__))
    save_json(out/'verification.json', checks)
    print(json.dumps(checks))


if __name__ == '__main__':
    main()
