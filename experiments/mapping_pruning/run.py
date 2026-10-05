"""Compare fixed pruning budgets without selecting a budget on test recall."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from experiments.mapping_ablation.run import evaluate, load_inputs, product, verify_recall
from mm_sae.analysis.data import save_json
from mm_sae.analysis.mapping_ablation import preprocess
from mm_sae.analysis.mapping_pruning import (
    coefficient_summary,
    procrustes_matrix,
    prune_coefficients,
)
from mm_sae.io import sha256
from mm_sae.progress import ProgressReporter, iter_progress, stage_progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    for key in ("source_run", "parent_run", "projection_run", "output"):
        cfg[key] = str((args.config.parent / cfg[key]).resolve())
    out = Path(cfg["output"])
    for path in (out, out / "results", out / "transforms"):
        path.mkdir(parents=True, exist_ok=True)
    source = Path(cfg["projection_run"])
    cca_path = source / "transforms" / f'cca_{cfg["dimensions"]}.npz'
    fingerprint_paths = [Path(__file__), cca_path,
                         Path(__file__).parents[2] / "src/mm_sae/analysis/mapping_pruning.py"]
    fingerprint = {str(path): sha256(path) for path in fingerprint_paths}
    destination = out / "pruning_sources.json"
    if destination.exists() and json.loads(destination.read_text()) != fingerprint:
        raise ValueError("Pruning code or transforms changed. Use a new output directory.")
    save_json(destination, fingerprint)
    stages = ["load", "prune_and_evaluate", "verify"]
    with ProgressReporter(out, stages, [], 10):
        with stage_progress("load"):
            _, ids, fit, values, parents = load_inputs(cfg, out)
            ni = len(ids["image"])
            x = preprocess(values["image"], fit.mean[:ni], fit.scale[:ni], "standardized")
            y = preprocess(values["text"], fit.mean[ni:], fit.scale[ni:], "standardized")
            covariance = fit.standardized(fit)
            direct = procrustes_matrix(covariance[:ni, ni:])
            with np.load(cca_path) as saved:
                wi, wt = saved["image"], saved["text"]
            # Project the smaller modality into the larger coordinate space.
            # At full density this preserves both norms, exactly reproducing
            # the previous zero-padded square Procrustes cosine calculation.
            if ni >= y.shape[1]:
                np.testing.assert_allclose(product(direct.T, direct), np.eye(y.shape[1]), atol=1e-10)
            else:
                np.testing.assert_allclose(product(direct, direct.T), np.eye(ni), atol=1e-10)
        with stage_progress("prune_and_evaluate"):
            jobs = [(family, k) for family in ("cca", "procrustes") for k in [None, *cfg["k_grid"]]]
            for family, k in iter_progress(jobs, "Delete frozen coefficients", unit="conditions"):
                key = f'{family}_{"full" if k is None else k}'
                if (out / "results" / (key + ".json")).exists():
                    continue
                if family == "cca":
                    a = prune_coefficients(wi, k, rule="column")
                    b = prune_coefficients(wt, k, rule="column")
                    xi, yt = product(x, a), product(y, b)
                    coefficients = {"image": a, "text": b}
                    rule = "At most k SAE inputs per modality per each fixed common component"
                    space = "common"
                else:
                    a = prune_coefficients(direct, k, rule="mutual")
                    xi, yt = (x, product(y, a.T)) if ni >= y.shape[1] else (product(x, a), y)
                    coefficients = {"image_by_text": a}
                    rule = "At most k incident cross-modal coefficients at each SAE feature"
                    space = "image" if ni >= y.shape[1] else "text"
                np.savez_compressed(out / "transforms" / (key + ".npz"), **coefficients)
                evaluate(out, cfg, parents, key=key, family=family, experiment="frozen_pruning",
                         condition="full" if k is None else str(k), space=space, image=xi, text=yt,
                         metadata={"k": k, "support_rule": rule, "preserve_sign_and_weights": True,
                                   "preprocessing": "fixed fit mean and standard deviation",
                                   "dimensions": int(wi.shape[1]) if family == "cca" else int(max(direct.shape)),
                                   "structure": {name: coefficient_summary(m) for name, m in coefficients.items()}})
        with stage_progress("verify"):
            verify_recall(out)
            differences = {}
            for family in ("cca", "procrustes"):
                original = f'cca_{cfg["dimensions"]}' if family == "cca" else family
                reference = json.loads((source / "results" / (original + ".json")).read_text())
                current = json.loads((out / "results" / (family + "_full.json")).read_text())
                differences[family] = {}
                for direction, metric in current["retrieval"].items():
                    changes = {k: v - reference["retrieval"][direction]["recall"][k]
                               for k, v in metric["recall"].items()}
                    differences[family][direction] = changes
                    if any(abs(change) > 1e-12 for change in changes.values()):
                        raise ValueError(f"Full-density reference recall differs: {family}/{direction}")
            save_json(out / "reference_verification.json", differences)
    print(f"Completed {out}", flush=True)


if __name__ == "__main__":
    main()
