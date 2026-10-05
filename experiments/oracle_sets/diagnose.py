"""Exploratory frozen-prediction diagnostics without selecting new feature sets.

Output-scale alternatives use training statistics only. Test-label substitutions
are explicitly privileged diagnostics and cannot be compared as deployed models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.oracle_sets.run import load_config
from mm_sae.analysis.data import save_json
from mm_sae.analysis.mapping_evaluation import paired_retrieval
from mm_sae.io import sha256
from mm_sae.progress import ProgressReporter, iter_progress, stage_progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg['output'])
    out = root/'diagnostics'
    out.mkdir(exist_ok=True)
    protocol = json.loads((root/'protocol.json').read_text())
    objects = np.array([r['group'] == 'object' for r in protocol['concepts']])
    index = Path(cfg['source_run'])/'index/val2017'
    parents = np.load(index/'parents.npy')
    labels = np.load(index/'presence.npy')
    manifest = dict(exploratory=True, selection_or_refitting=False,
                    diagnostic_budgets=[16, 'full'], source_run=str(root),
                    note='No best condition is selected or used to replace the original preregistered result',
                    code_sha256=sha256(Path(__file__)))
    results = []
    with ProgressReporter(out, ['diagnostics'], [], interval=10):
        with stage_progress('diagnostics'):
            for k in iter_progress([16, 'full'], 'Diagnose fixed concept predictions', unit='budgets'):
                prediction = root/'predictions'/f'propagated_presence_{k}.npz'
                manifest[str(prediction)] = sha256(prediction)
                with np.load(prediction) as saved:
                    values = {s: saved[s] for s in ('image', 'text')}
                centered, label_scaled, truth = {}, {}, {}
                for side, fit_key in [('image', 'image'), ('text', 'propagated_presence')]:
                    fit_path = root/'fits'/f'{fit_key}_{k}.npz'
                    manifest[str(fit_path)] = sha256(fit_path)
                    with np.load(fit_path) as saved:
                        prediction_scale = saved['prediction_scale']
                        label_scale = np.sqrt(saved['label_variance'])
                        mean = saved['label_mean']
                    centered[side] = values[side] * prediction_scale[None, :]
                    label_scaled[side] = np.divide(centered[side], label_scale[None, :],
                                                  out=np.zeros_like(centered[side]),
                                                  where=label_scale[None, :] > 0)
                    y = labels if side == 'image' else labels[parents]
                    truth[side] = np.divide(y-mean, label_scale[None, :], out=np.zeros_like(y, dtype=float),
                                           where=label_scale[None, :] > 0)
                conditions = [
                    ('prediction_unit_variance', values['image'], values['text'], False),
                    ('centered_prediction', centered['image'], centered['text'], False),
                    ('label_unit_variance', label_scaled['image'], label_scaled['text'], False),
                    ('objects_prediction_unit_variance', values['image'][:, objects], values['text'][:, objects], False),
                    ('background_prediction_unit_variance', values['image'][:, ~objects], values['text'][:, ~objects], False),
                    ('true_image_labels_predicted_text', truth['image'], label_scaled['text'], True),
                    ('predicted_image_true_text_labels', label_scaled['image'], truth['text'], True),
                ]
                for name, image, text, privileged in conditions:
                    retrieval = paired_retrieval(image, text, parents, chunk_size=cfg['retrieval_chunk'])
                    result = dict(k=k, condition=name, uses_test_annotations=privileged,
                                  dimensions=image.shape[1], retrieval=retrieval)
                    results.append(result)
                    print(json.dumps(dict(k=k, condition=name,
                                          recall={d: m['recall'] for d, m in retrieval.items()})), flush=True)
                    save_json(out/'results.json', results)
    manifest[str(index/'presence.npy')] = sha256(index/'presence.npy')
    manifest[str(index/'parents.npy')] = sha256(index/'parents.npy')
    save_json(out/'manifest.json', manifest)


if __name__ == '__main__':
    main()
