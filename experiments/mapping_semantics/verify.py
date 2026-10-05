"""Recompute retrieval and independently audit fixed-coordinate AUROCs."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.mapping_semantics.run import load_config, load_data, models, project
from mm_sae.analysis.data import save_json
from mm_sae.analysis.evaluation import auroc
from mm_sae.analysis.mapping_evaluation import paired_retrieval
from mm_sae.io import sha256
from mm_sae.progress import ProgressReporter, iter_progress, stage_progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg['output'])
    progress = out / 'verification-progress'
    progress.mkdir(exist_ok=True)
    checked, differences = 0, []
    with ProgressReporter(progress, ['load', 'verify'], [], 10):
        with stage_progress('load'):
            data = load_data(cfg)
            jobs = models(cfg, data, 'all')
        with stage_progress('verify'):
            for model in iter_progress(jobs, 'Recompute all retrieval results and scalar AUROC checks', unit='models'):
                result = json.loads((out / 'results' / (model.key + '.json')).read_text())
                values = {s: project(data['test']['x'][s], getattr(model, s)) for s in ('image', 'text')}
                scores = paired_retrieval(values['image'], values['text'], data['parents'],
                                          chunk_size=cfg['retrieval_chunk'], device='cpu')
                max_difference = 0.
                rank_differences = {}
                for direction in scores:
                    rank_delta = np.asarray(scores[direction]['ranks']) - result['retrieval'][direction]['ranks']
                    changed = np.flatnonzero(rank_delta)
                    rank_differences[direction] = {
                        'changed_queries': int(len(changed)), 'max_absolute_rank_change': int(np.max(np.abs(rank_delta))),
                        'first_changed_rows': changed[:10].tolist(),
                    }
                    for k, v in scores[direction]['recall'].items():
                        difference = abs(v - result['retrieval'][direction]['recall'][str(k)])
                        assert difference < 1e-14
                        max_difference = max(max_difference, difference)
                        checked += 1
                semantic_checks = 0
                for row in result['semantic']:
                    if row['name'] not in ['person', 'dog', 'skis', 'snow', 'grass', 'road', 'sand']:
                        continue
                    if row['coordinate'] is None:
                        continue
                    source, target = row['direction'].split('_to_')
                    for side, field in [(source, 'source_auc'), (target, 'target_transfer_auc')]:
                        value = auroc(data['test']['labels'][side][:, row['concept_index']],
                                      row['sign'] * values[side][:, row['coordinate']])
                        if row[field] is None:
                            assert np.isnan(value)
                        else:
                            np.testing.assert_allclose(value, row[field], atol=1e-12, rtol=0)
                        semantic_checks += 1
                differences.append({'key': model.key, 'max_recall_difference': max_difference,
                                    'rank_differences': rank_differences,
                                    'independent_scalar_auc_checks': semantic_checks})
                print(json.dumps(differences[-1]), flush=True)
    repo = Path(__file__).resolve().parents[2]
    code_paths = list((repo / 'experiments/mapping_semantics').glob('*.py'))
    code_paths += [repo / 'src/mm_sae/analysis' / n for n in
                   ['semantic_evaluation.py', 'sparse_cca.py', 'signed_mapping.py', 'mapping_evaluation.py',
                    'mapping_ablation.py', 'evaluation.py']]
    save_json(out / 'verification.json', {
        'all_models_checked': len(jobs), 'recall_values_checked': checked,
        'all_query_ranks_identical': all(not d['changed_queries'] for m in differences for d in m['rank_differences'].values()),
        'semantic_reference': 'independent scalar score-group AUROC with half tie credit',
        'models': differences,
        'code_hashes_at_verification': {str(p): sha256(p) for p in code_paths},
        'dataset_description_sha256': sha256(Path(cfg['source_run']) / 'dataset.json'),
        'same_fixed_fit_tune_test_partitions': True,
    })


if __name__ == '__main__':
    main()
