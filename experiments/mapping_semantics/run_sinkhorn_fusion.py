"""Compare fixed Sinkhorn spaces with their equal-weight cosine-score mean."""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import yaml

from experiments.mapping_ablation.run import load_inputs, product
from experiments.mapping_semantics.run_support_sweep import compact_retrieval
from mm_sae.analysis.mapping_ablation import preprocess
from mm_sae.analysis.mapping_evaluation import paired_retrieval
from mm_sae.analysis.mapping_pruning import prune_coefficients
from mm_sae.io import atomic_json, sha256


def unit_rows(values):
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 0):
        raise ValueError("Score-mean concatenation requires finite, nonzero vectors in both spaces")
    return values / norms


def mean_cosine_vectors(image_in_text, text, image, text_in_image):
    """Cosine of these concatenations equals the mean of the two input cosines.

    Both blocks have unit norm, making every concatenation's norm sqrt(2).
    No parameters are fitted and no retrieval metrics are averaged.
    """
    return (np.concatenate((unit_rows(image_in_text), unit_rows(image)), axis=1),
            np.concatenate((unit_rows(text), unit_rows(text_in_image)), axis=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    root = (args.config.parent / config['output']).resolve()
    for condition in config['conditions']:
        cfg = {**config, **condition}
        for key in ('source_run', 'parent_run', 'projection_run', 'sinkhorn_transform'):
            cfg[key] = str((args.config.parent / cfg[key]).resolve())
        out = root / condition['name']
        out.mkdir(parents=True, exist_ok=True)
        def status(state, **extra):
            record = dict(condition=condition['name'], state=state, updated_at=time.time(), **extra)
            atomic_json(out / 'progress.json', record)
            print(record, flush=True)
        status('loading')
        _, ids, fit, values, parents = load_inputs(cfg, out)
        ni = len(ids['image'])
        x = preprocess(values['image'], fit.mean[:ni], fit.scale[:ni], 'standardized')
        y = preprocess(values['text'], fit.mean[ni:], fit.scale[ni:], 'standardized')
        del values
        with np.load(cfg['sinkhorn_transform']) as saved:
            full = saved['mapping']
        for support in config['supports']:
            started = time.monotonic()
            key = 'full' if support is None else str(support)
            dest = out / f'{key}.json'
            if dest.exists():
                continue
            status('projecting', support=support)
            mapping = full if support is None else prune_coefficients(full, support, rule='mutual')
            row_sum, col_sum = mapping.sum(1), mapping.sum(0)
            forward = np.divide(mapping, row_sum[:, None], out=np.zeros_like(mapping),
                                where=row_sum[:, None] > 0)
            backward = np.divide(mapping, col_sum[None, :], out=np.zeros_like(mapping),
                                 where=col_sum[None, :] > 0)
            xp, yp = product(x, backward), product(y, forward.T)
            retrieval = {}
            for name, left, right in [('image_transform', xp, y), ('text_transform', x, yp)]:
                status('evaluating', support=support, method=name)
                retrieval[name] = compact_retrieval(paired_retrieval(
                    left, right, parents, device=cfg['device'], chunk_size=cfg['retrieval_chunk']))
            fx, fy = mean_cosine_vectors(xp, y, x, yp)
            # Check the score identity on actual evaluation samples before retrieval.
            expected = .5 * (unit_rows(xp[:16]) @ unit_rows(y[:23]).T
                              + unit_rows(x[:16]) @ unit_rows(yp[:23]).T)
            actual = unit_rows(fx[:16]) @ unit_rows(fy[:23]).T
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
            status('evaluating', support=support, method='mean_cosine')
            retrieval['mean_cosine'] = compact_retrieval(paired_retrieval(
                fx, fy, parents, device=cfg['device'], chunk_size=cfg['retrieval_chunk']))
            atomic_json(dest, dict(support=support, epsilon=.05, score_weights=[.5, .5],
                                   mapping_sha256=sha256(Path(cfg['sinkhorn_transform'])),
                                   code_sha256=sha256(Path(__file__)), retrieval=retrieval,
                                   seconds=time.monotonic()-started))
            status('job_completed', support=support, elapsed_seconds=time.monotonic()-started)
            del xp, yp, fx, fy, forward, backward
        status('completed')
        del x, y, full, fit
        gc.collect()


if __name__ == '__main__':
    main()
