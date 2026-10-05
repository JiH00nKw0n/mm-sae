"""Extend fixed-representation probes to configurable nested feature subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import yaml

from experiments.representation_diagnosis.run import Diagnosis, load_config
from mm_sae.analysis.data import save_json
from mm_sae.io import sha256
from mm_sae.progress import ProgressReporter, progress_task, stage_progress


def load_options(path):
    path = Path(path).resolve()
    extension = yaml.safe_load(path.read_text())
    base = path.parent / extension['diagnosis_config']
    options = load_config(base)
    for key in ['output', 'baseline_run']:
        options[key] = str((path.parent / extension[key]).resolve())
    for key in ['feature_counts', 'objectives', 'device', 'categories', 'report_exclude_categories']:
        if key in extension:
            options[key] = extension[key]
    if 'feature_run' in extension:
        options['feature_run'] = str((path.parent / extension['feature_run']).resolve())
    counts = options['feature_counts']
    if not counts or any(type(n) is not int or n < 1 or n == 5 for n in counts):
        raise ValueError('New subset sizes must be positive integers other than the existing five')
    if len(set(counts)) != len(counts):
        raise ValueError('Duplicate feature counts')
    if not options['objectives'] or not set(options['objectives']) <= {'presence', 'removal'}:
        raise ValueError('Unknown probe objective')
    if options['output'] in [options[k] for k in ['source_run', 'feature_run', 'baseline_run']]:
        raise ValueError('Output must be separate from every input run')
    return options


def verify_baselines(ctx):
    """Require identical inputs and statistical settings before reusing fitted probes."""
    root = Path(ctx.o['baseline_run'])
    baseline = json.loads((root / 'config.json').read_text())
    for key in ['source_run', 'feature_run', 'splits', 'penalties', 'maxiter',
                'bootstrap_resamples', 'bootstrap_seed', 'feature_count']:
        if baseline[key] != ctx.o[key]:
            raise ValueError(f'Baseline configuration differs: {key}')
    old_hashes = json.loads((root / 'input_hashes.json').read_text())
    new_hashes = json.loads((ctx.out / 'input_hashes.json').read_text())
    if any(old_hashes.get(p) != value for p, value in new_hashes.items()):
        raise ValueError('Baseline input files differ from the current frozen inputs')
    reuse = {}
    for cid in ctx.ids:
        for rep in ['embedding', 'all_sae', 'selected_five']:
            for task in ctx.o['objectives']:
                key = f'{cid}_{rep}_{task}'
                path = root / 'jobs' / f'{key}.json'
                record = json.loads(path.read_text())
                if record['category_id'] != cid or record['representation'] != rep or record['objective'] != task:
                    raise ValueError(f'Baseline identity differs: {path}')
                if rep == 'selected_five':
                    np.testing.assert_array_equal(record['model']['features'], ctx.selected(cid, 5))
                shutil.copyfile(path, ctx.out / 'jobs' / path.name)
                reuse[str(path)] = sha256(path)
    save_json(ctx.out / 'reused_baselines.json', reuse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    options = load_options(args.config)
    out = Path(options['output'])
    for name in ['jobs', 'scores', 'report']:
        (out / name).mkdir(parents=True, exist_ok=True)
    code_paths = [Path(__file__), Path(__file__).with_name('run.py'),
                  Path(__file__).parents[2] / 'src/mm_sae/analysis/probes.py',
                  Path(__file__).parents[2] / 'src/mm_sae/analysis/linear.py',
                  Path(__file__).parents[2] / 'src/mm_sae/analysis/evaluation.py',
                  Path(__file__).parents[2] / 'src/mm_sae/analysis/data.py']
    signature = dict(options=options, code={p.name: sha256(p) for p in code_paths})
    signature['fingerprint'] = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    lock = out / 'signature.json'
    if lock.exists() and json.loads(lock.read_text()) != signature:
        raise ValueError('Configuration or computational code changed; choose a new output directory')
    save_json(lock, signature)
    save_json(out / 'config.json', options)
    from .feature_count_report import make_report
    if args.report_only:
        make_report(out)
        return
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[key] = '1'
    stages = ['verify_inputs', 'reuse_baselines', 'linear_probes', 'report']
    with ProgressReporter(out, stages, [], interval=10):
        with stage_progress('verify_inputs'):
            ctx = Diagnosis(options)
            ctx.verify()
            for cid in ctx.ids:
                ctx.selected(cid, max(options['feature_counts']))
        with stage_progress('reuse_baselines'):
            verify_baselines(ctx)
        jobs = [(cid, f'selected_{n}', task) for cid in ctx.ids
                for n in sorted(options['feature_counts']) for task in options['objectives']]
        finished = sum((out / 'jobs' / f'{c}_{r}_{t}.json').exists() for c, r, t in jobs)
        with stage_progress('linear_probes'), progress_task(
            'Fit selected-feature probes', len(jobs), 'jobs', initial=finished
        ) as progress:
            for cid, rep, task in jobs:
                if (out / 'jobs' / f'{cid}_{rep}_{task}.json').exists():
                    continue
                progress.update(progress.completed, category_id=cid, name=ctx.data.names[cid],
                                representation=rep, objective=task, penalty=None)
                ctx.job(cid, rep, task, progress)
                progress.update(progress.completed + 1)
        with stage_progress('report'):
            make_report(out)


if __name__ == '__main__':
    main()
