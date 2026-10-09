"""Describe train/test activation shifts and control Hungarian feature coverage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from experiments.mapping_ablation.run import evaluate, load_inputs, verify_recall
from mm_sae.analysis.data import save_json
from mm_sae.analysis.mapping_ablation import preprocess
from mm_sae.analysis.mapping_evaluation import mapping_retrieval
from mm_sae.io import sha256, write_csv
from mm_sae.metrics.regression import Moments
from mm_sae.progress import ProgressReporter, stage_progress


def projection_diagnostics(moments: Moments, reference: Moments, wi, wt):
    """Evaluation moments only describe frozen outputs; they never alter weights."""
    ni = len(wi)
    second = moments.standardized(reference)
    mean = (moments.mean - reference.mean) / reference.scale
    covariance = second - np.outer(mean, mean)
    vi = np.sum(wi * (covariance[:ni, :ni] @ wi), axis=0)
    vt = np.sum(wt * (covariance[ni:, ni:] @ wt), axis=0)
    cross = np.sum(wi * (covariance[:ni, ni:] @ wt), axis=0)
    denom = np.sqrt(np.maximum(vi * vt, 0))
    corr = np.divide(cross, denom, out=np.full_like(cross, np.nan), where=denom > 1e-12)
    mi, mt = mean[:ni] @ wi, mean[ni:] @ wt
    return [dict(coordinate=int(i), correlation=float(corr[i]), image_variance=float(vi[i]),
                 text_variance=float(vt[i]), image_mean=float(mi[i]), text_mean=float(mt[i]))
            for i in range(wi.shape[1])]


def run(cfg):
    out = Path(cfg["output"])
    (out / "results").mkdir(parents=True, exist_ok=True)
    with ProgressReporter(out, ["load", "distribution", "matched_features"], [], 10):
        with stage_progress("load"):
            selected, ids, fit, values, parents = load_inputs(cfg, out)
            with np.load(Path(cfg["parent_run"]) / "moments.npz") as z:
                test = Moments(int(z["test_n"]), z["test_mean"], z["test_second"])
            ni = len(ids["image"])
            source_paths = [Path(__file__), *[
                Path(cfg["projection_run"]) / "transforms" / f"{key}.npz"
                for key in ("cca_256", "cross_svd_256")]]
            for family in ("hungarian", "sinkhorn"):
                chosen = next(row for row in selected if row["family"] == family)
                source_paths.append(Path(cfg["parent_run"]) / "candidates" / f'{chosen["key"]}.npz')
            sources = {str(p): sha256(p) for p in source_paths}
            receipt = out / "diagnostic-sources.json"
            if receipt.exists() and json.loads(receipt.read_text()) != sources:
                raise ValueError("Diagnostic code or projection weights changed; use a new output directory")
            save_json(receipt, sources)
            rows = []
        with stage_progress("distribution"):
            for key in ("cca_256", "cross_svd_256"):
                path = Path(cfg["projection_run"]) / "transforms" / f"{key}.npz"
                with np.load(path) as z:
                    for name, moments in (("fit", fit), ("test", test)):
                        rows.extend(dict(method=key, population=name, **r) for r in
                                    projection_diagnostics(moments, fit, z["image"], z["text"]))
            write_csv(out / "component-distribution.csv", rows)
            summaries = {}
            for side, block in (("image", slice(None, ni)), ("text", slice(ni, None))):
                mean = fit.mean[block]
                variance = fit.variance[block]
                test_values = values[side].astype(np.float64)
                summaries[side] = dict(
                    fit_mean_energy_fraction=float(np.sum(mean ** 2) / np.sum(variance + mean ** 2)),
                    test_mean_energy_fraction=float(np.sum(test_values.mean(0) ** 2) /
                                                    np.mean(np.sum(test_values ** 2, axis=1))),
                    fit_feature_count=len(mean),
                    features_constant_in_test=int(np.count_nonzero(test.variance[block] <= 1e-12)),
                    fit_variance_quantiles=np.quantile(variance, [0, .25, .5, .75, 1]),
                    test_to_fit_variance_ratio_quantiles=np.quantile(test.variance[block] / variance,
                                                                    [0, .25, .5, .75, 1]),
                    fit_mean_to_test_mean_standardized_distance=float(np.linalg.norm(
                        (test.mean[block] - mean) / fit.scale[block])),
                )
            save_json(out / "activation-distribution.json", summaries)
        with stage_progress("matched_features"):
            maps = {}
            for family in ("hungarian", "sinkhorn"):
                chosen = next(row for row in selected if row["family"] == family)
                path = Path(cfg["parent_run"]) / "candidates" / f'{chosen["key"]}.npz'
                with np.load(path) as z:
                    maps[family] = z["mapping"]
            keep_i, keep_t = maps["hungarian"].sum(1) > 0, maps["hungarian"].sum(0) > 0
            for mode in ("raw", "centered", "standardized"):
                x = preprocess(values["image"], fit.mean[:ni], fit.scale[:ni], mode)[:, keep_i]
                y = preprocess(values["text"], fit.mean[ni:], fit.scale[ni:], mode)[:, keep_t]
                for family, mapping in maps.items():
                    key = f"{family}_{mode}"
                    scores = mapping_retrieval(x, y, mapping[np.ix_(keep_i, keep_t)], parents,
                                               device=cfg["device"], chunk_size=cfg["retrieval_chunk"])
                    for space, retrieval in scores.items():
                        evaluate(out, cfg, parents, key=f"{key}__{space}", family=family,
                                 experiment="matched_features", condition=mode, space=space,
                                 metadata=dict(image_features=int(keep_i.sum()), text_features=int(keep_t.sum()),
                                               fixed_hungarian_coordinates=True, refit=False), retrieval=retrieval)
        verify_recall(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    for key in ("source_run", "parent_run", "projection_run", "output"):
        cfg[key] = str((args.config.parent / cfg[key]).resolve())
    run(cfg)
    print(json.dumps(dict(completed=cfg["output"])))


if __name__ == "__main__":
    main()
