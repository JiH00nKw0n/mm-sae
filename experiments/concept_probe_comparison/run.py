"""Compare support refinement and fixed-support logistic readouts independently.

Both comparisons retain the previous frozen SAE, fit split, inherited image
labels, 171 named outputs, and fit-centered/unit-variance linear retrieval
scores. Logistic sigmoid probabilities never replace the linear search score.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import yaml

from experiments.oracle_sets.run import load_fit, load_test, test_semantics
from mm_sae.analysis.concept_sets import fit_concept_sets
from mm_sae.analysis.concept_refinement import refine_concept_supports
from mm_sae.analysis.concept_logistic import fit_fixed_support_logistic
from mm_sae.analysis.concept_supervision import project_concepts, standardize_concept_weights
from mm_sae.analysis.data import CachedSplit, save_json
from mm_sae.analysis.evaluation import distribution
from mm_sae.analysis.mapping_evaluation import paired_retrieval
from mm_sae.analysis.sparse_cca import fit_sparse_cca
from mm_sae.io import sha256
from mm_sae.metrics.regression import Moments
from mm_sae.progress import ProgressReporter, stage_progress, progress_task, iter_progress


METHODS = ("forward_ridge", "swap_ridge", "fixed_support_logistic")
LABELS = {
    "forward_ridge": "특징을 추가하며 선택한 제곱오차 회귀",
    "swap_ridge": "선택한 특징의 교체도 허용한 제곱오차 회귀",
    "fixed_support_logistic": "기존 특징 번호를 고정한 로지스틱 분류기",
    "sparse_cca": "Sparse CCA",
}
SIDES = ("image", "text")


def config_from(path, *, device=None, smoke_concepts=None):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    for key in ("source_run", "parent_run", "reference_run", "semantics_run", "output"):
        cfg[key] = str((path.parent / cfg[key]).resolve())
    if device is not None:
        cfg["device"] = device
    if smoke_concepts is not None:
        if smoke_concepts < 1:
            raise ValueError("smoke_concepts must be positive")
        cfg["concept_limit"] = smoke_concepts
        cfg["output"] += "-smoke"
    if cfg["label_targets"] != ["propagated_presence"]:
        raise ValueError("This comparison must retain inherited image presence labels")
    if not cfg["budgets"] or any(type(k) is not int or k < 1 for k in cfg["budgets"]):
        raise ValueError("Feature budgets must be positive integers")
    return cfg


def freeze_inputs(cfg):
    root = Path(__file__).resolve().parents[2]
    source, parent = Path(cfg["source_run"]), Path(cfg["parent_run"])
    files = [parent / "population.json", parent / "moments.npz", source / "dataset.json"]
    for split in ("train2017", "val2017"):
        files.extend(source / "index" / split / n for n in
                     ("images.json", "parents.npy", "presence.npy", "concept_ids.json", "mentions.npy"))
        files.extend(source / "activations" / split / (s + ".npz") for s in SIDES)
    for k in cfg["budgets"]:
        files.extend(p for p in (
            Path(cfg["reference_run"]) / "fits" / f"image_{k}.npz",
            Path(cfg["reference_run"]) / "fits" / f"propagated_presence_{k}.npz",
            Path(cfg["semantics_run"]) / "sparse-fit" / f"sparse_cca_k{k}.npz",
        ) if p.exists())
    code = [Path(__file__), root / "experiments/oracle_sets/run.py"]
    code.extend(root / "src/mm_sae/analysis" / n for n in (
        "concept_sets.py", "concept_refinement.py", "concept_logistic.py", "concept_supervision.py",
        "mapping_evaluation.py", "sparse_cca.py", "linear.py",
    ))
    payload = {"config": cfg, "sources": {str(p): sha256(p) for p in files},
               "code": {str(p): sha256(p) for p in code}}
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    dest = Path(cfg["output"]) / "manifest.json"
    if dest.exists() and json.loads(dest.read_text())["signature"] != signature:
        raise ValueError("Sources, configuration or calculation code changed; use a fresh output folder")
    save_json(dest, dict(signature=signature, **payload))


def apply_concept_limit(data, cfg):
    count = min(cfg.get("concept_limit", len(data["concepts"])), len(data["concepts"]))
    data["concepts"] = data["concepts"][:count]
    for moments in data["moments"].values():
        moments["cross"] = moments["cross"][:, :count]
        for key in ("label_mean", "label_variance"):
            moments[key] = moments[key][:count]
    return data


def save_fit(path, coefficients, stats, ids, bias, *, fit_prediction_mean=None):
    normalized, prediction_scale, active = standardize_concept_weights(coefficients, stats["cov"])
    bias = np.asarray(bias, float)
    center = bias if fit_prediction_mean is None else np.asarray(fit_prediction_mean, float)
    offset = np.divide(center - bias, prediction_scale, out=np.zeros_like(center), where=active)
    np.savez_compressed(path, raw_coefficients=coefficients, coefficients=normalized,
                        raw_bias=bias, fit_prediction_mean=center, projection_offset=offset,
                        prediction_scale=prediction_scale, active=active, feature_ids=ids,
                        feature_mean=stats["mean"], feature_scale=stats["scale"])


def fit_forward(cfg, data):
    out = Path(cfg["output"])
    for side in SIDES:
        if all((out / "fits" / f"forward_ridge_{side}_{k}.npz").exists() for k in cfg["budgets"]):
            continue
        key = "image" if side == "image" else "propagated_presence"
        moments, stats = data["moments"][key], data["stats"][side]
        with progress_task(f"Select forward ridge features: {side}", unit="concepts"):
            fitted = fit_concept_sets(stats["cov"], moments["cross"], moments["label_mean"],
                                     moments["label_variance"], budgets=cfg["budgets"],
                                     penalty=cfg["ridge"], min_gain=cfg["min_gain"])
        previous_checks = []
        for k, coef in fitted.coefficients.items():
            previous = Path(cfg["reference_run"]) / "fits" / f"{key}_{k}.npz"
            if previous.exists():
                with np.load(previous) as saved:
                    old = saved["raw_coefficients"][:, :len(data["concepts"])]
                    np.testing.assert_allclose(coef, old, rtol=1e-7, atol=1e-10)
                    np.testing.assert_array_equal(saved["feature_ids"], data["ids"][side])
                    previous_checks.append(dict(k=k, maximum_absolute_difference=float(np.max(np.abs(coef-old)))))
            save_fit(out / "fits" / f"forward_ridge_{side}_{k}.npz", coef, stats,
                     data["ids"][side], fitted.label_mean)
        save_json(out / "diagnostics" / f"forward_{side}.json", dict(
            objective_history=fitted.objective_history, selected_features=fitted.selected_features,
            previous_fit_checks=previous_checks, penalty=fitted.penalty))


def fit_refined(cfg, data):
    out = Path(cfg["output"])
    for k, side in iter_progress([(k, s) for k in cfg["budgets"] for s in SIDES],
                                "Refine annotated feature sets", unit="fits"):
        dest = out / "fits" / f"swap_ridge_{side}_{k}.npz"
        if dest.exists():
            continue
        with np.load(out / "fits" / f"forward_ridge_{side}_{k}.npz") as saved:
            initial = saved["raw_coefficients"]
        key = "image" if side == "image" else "propagated_presence"
        moments, stats = data["moments"][key], data["stats"][side]
        with progress_task(f"Feature swaps: {side}, k={k}", len(data["concepts"]), "concepts") as task:
            def callback(completed, total, trace):
                task.update(completed, concept=trace.concept, accepted_swaps=len(trace.swaps),
                            converged=trace.converged)
            fit = refine_concept_supports(
                stats["cov"], moments["cross"], moments["label_mean"], moments["label_variance"], initial,
                penalty=cfg["ridge"], max_passes=cfg["swap_max_passes"],
                tolerance=cfg["swap_tolerance"], callback=callback,
            )
        save_json(out / "diagnostics" / f"swap_{side}_{k}.json", dict(
            traces=[asdict(t) for t in fit.traces], penalty=fit.penalty,
            converged=sum(t.converged for t in fit.traces),
            hit_pass_limit=sum(t.hit_pass_limit for t in fit.traces)))
        if any(t.hit_pass_limit for t in fit.traces):
            raise RuntimeError("Feature swaps reached the configured limit before one-swap convergence")
        save_fit(dest, fit.coefficients, stats, data["ids"][side], fit.label_mean)


def logistic_training_arrays(cfg, data):
    train = CachedSplit(Path(cfg["source_run"]), "train2017")
    keep = np.isin(train.image_ids, data["population"]["fit_image_ids"])
    rows = keep[train.parents]
    count = len(data["concepts"])
    weights = np.bincount(train.parents[rows], minlength=len(keep))[keep].astype(float)
    if int(rows.sum()) != data["fit_caption_count"]:
        raise ValueError("Logistic fitting population changed")
    return {
        "image": (train.activations["image"][keep][:, data["ids"]["image"]],
                  np.asarray(train.presence[keep, :count], bool), weights),
        "text": (train.activations["text"][rows][:, data["ids"]["text"]],
                 np.asarray(train.presence[train.parents[rows], :count], bool), None),
    }


def fit_logistic(cfg, data):
    out = Path(cfg["output"])
    if all((out / "fits" / f"fixed_support_logistic_{s}_{k}.npz").exists()
           for k in cfg["budgets"] for s in SIDES):
        return
    arrays = logistic_training_arrays(cfg, data)
    for k, side in iter_progress([(k, s) for k in cfg["budgets"] for s in SIDES],
                                "Fit fixed-support logistic concepts", unit="fits"):
        dest = out / "fits" / f"fixed_support_logistic_{side}_{k}.npz"
        if dest.exists():
            continue
        with np.load(out / "fits" / f"forward_ridge_{side}_{k}.npz") as saved:
            initial = saved["raw_coefficients"]
        supports = [np.flatnonzero(initial[:, c]) for c in range(initial.shape[1])]
        x, y, weights = arrays[side]
        stats = data["stats"][side]
        traces = []
        with progress_task(f"Logistic concepts: {side}, k={k}", len(data["concepts"]), "concepts") as task:
            def callback(concept, diagnostic):
                traces.append(diagnostic)
                task.update(concept + 1, concept=concept)
                save_json(out / "diagnostics" / f"logistic_partial_{side}_{k}.json", traces)
            fitted = fit_fixed_support_logistic(
                x, y, stats["mean"], stats["scale"], supports, sample_weight=weights,
                penalty=cfg["logistic_penalty"], device=cfg["logistic_device"],
                maxiter=cfg["logistic_maxiter"], callback=callback,
            )
        if np.any((fitted.coefficients != 0) & (initial == 0)):
            raise ArithmeticError("Logistic fit changed frozen feature support")
        save_fit(dest, fitted.coefficients, stats, data["ids"][side], fitted.bias,
                 fit_prediction_mean=fitted.fit_prediction_mean)
        save_json(out / "diagnostics" / f"logistic_{side}_{k}.json", fitted.diagnostics)


def fit_controls(cfg, data):
    out = Path(cfg["output"])
    with np.load(Path(cfg["parent_run"]) / "moments.npz") as saved:
        moments = Moments(int(saved["fit_n"]), saved["fit_mean"], saved["fit_second"])
    cov = moments.standardized(moments)
    ni, count = len(data["ids"]["image"]), len(data["concepts"])
    for k in cfg["budgets"]:
        if all((out / "fits" / f"sparse_cca_{s}_{k}.npz").exists() for s in SIDES):
            continue
        path = Path(cfg["semantics_run"]) / "sparse-fit" / f"sparse_cca_k{k}.npz"
        if path.exists():
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved["image_ids"], data["ids"]["image"])
                np.testing.assert_array_equal(saved["text_ids"], data["ids"]["text"])
                coef = {s: saved[s][:, :count].copy() for s in SIDES}
            save_json(out / "diagnostics" / f"sparse_cca_{k}.json", dict(reused=str(path), sha256=sha256(path)))
        else:
            with progress_task(f"Fit Sparse CCA k={k}", count, "coordinates") as task:
                model = fit_sparse_cca(
                    cov[:ni, :ni], cov[:ni, ni:], cov[ni:, ni:], dimensions=count, k=k,
                    ridge=cfg["ridge"], max_iter=cfg["sparse_cca_maxiter"],
                    tolerance=cfg["sparse_cca_tolerance"],
                    progress=lambda record: task.update(record["component"] + 1,
                                                       iterations=record["iterations"]),
                )
            coef = {"image": model.image, "text": model.text}
            save_json(out / "diagnostics" / f"sparse_cca_{k}.json", model.metadata)
            if model.metadata["converged_components"] != count:
                raise RuntimeError("Sparse CCA did not converge for every output coordinate")
        for side in SIDES:
            save_fit(out / "fits" / f"sparse_cca_{side}_{k}.npz", coef[side],
                     data["stats"][side], data["ids"][side], np.zeros(count))


def evaluate(cfg, data):
    out = Path(cfg["output"])
    # Evaluation labels are first opened only after all fit stages are finished.
    test: dict[str, Any] = load_test(cfg, {**data, "concepts": _all_concepts(cfg)})
    count = len(data["concepts"])
    for key in ("image_labels", "text_presence", "text_mentions"):
        test[key] = test[key][:, :count]
    for method, k in iter_progress([(m, k) for m in (*METHODS, "sparse_cca") for k in cfg["budgets"]],
                                   "Evaluate both retrieval directions", unit="conditions"):
        key = f"{method}_{k}"
        dest = out / "results" / f"{key}.json"
        if dest.exists():
            continue
        start = time.monotonic()
        scores, structure = {}, {}
        for side in SIDES:
            with np.load(out / "fits" / f"{method}_{side}_{k}.npz") as saved:
                scores[side] = project_concepts(test["raw"][side], saved["feature_mean"],
                                                saved["feature_scale"], saved["coefficients"])
                scores[side] -= saved["projection_offset"][None, :]
                b = saved["raw_coefficients"]
                structure[side] = dict(nonzero=int(np.count_nonzero(b)),
                                       max_inputs=int(np.count_nonzero(b, axis=0).max()))
        retrieval = paired_retrieval(scores["image"], scores["text"], test["parents"],
                                    chunk_size=cfg["retrieval_chunk"], device=cfg["device"])
        semantic = [] if method == "sparse_cca" else test_semantics(
            data, test, scores["image"], scores["text"], "propagated_presence")
        summary = {s: distribution([v[s + "_auc"] for v in semantic]) for s in SIDES}
        np.savez_compressed(out / "predictions" / f"{key}.npz", **scores)
        save_json(dest, dict(key=key, method=method, label=LABELS[method], k=k, dimensions=count,
                            retrieval=retrieval, semantic=semantic, semantic_summary=summary,
                            structure=structure, seconds=time.monotonic()-start))
        print(json.dumps({"completed": key, "recall": {d: r["recall"] for d, r in retrieval.items()},
                          "seconds": time.monotonic()-start}), flush=True)


def _all_concepts(cfg):
    source = Path(cfg["source_run"])
    lookup = {int(r["id"]): r["name"] for r in json.loads((source / "dataset.json").read_text())["concepts"]}
    ids = json.loads((source / "index/train2017/concept_ids.json").read_text())
    return [dict(id=int(c), name=lookup[int(c)], group="object" if int(c) < 91 else "background") for c in ids]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device")
    parser.add_argument("--smoke-concepts", type=int)
    args = parser.parse_args()
    cfg = config_from(args.config, device=args.device, smoke_concepts=args.smoke_concepts)
    out = Path(cfg["output"])
    for folder in ("fits", "diagnostics", "results", "predictions"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    stages = ["freeze_inputs", "load_training", "forward_selection", "support_swaps",
              "fixed_support_logistic", "sparse_cca_controls", "evaluate", "report"]
    with ProgressReporter(out, stages, [], interval=cfg["progress_interval_seconds"]):
        with stage_progress("freeze_inputs"):
            freeze_inputs(cfg)
        with stage_progress("load_training"):
            data = apply_concept_limit(load_fit(cfg), cfg)
            save_json(out / "protocol.json", dict(
                fit_images=data["fit_image_count"], fit_caption_pairs=data["fit_caption_count"],
                test_images=len(data["population"]["test_image_ids"]), concepts=data["concepts"],
                budgets=cfg["budgets"], ridge=cfg["ridge"], logistic_penalty=cfg["logistic_penalty"],
                coefficient_tuning="none; both penalties fixed in advance",
                search_representation="linear score, centered and scaled using fit predictions; no sigmoid",
                label_target="image presence copied to paired captions, including unmentioned categories",
                comparison_1="same ridge objective; forward-only supports versus exact one-feature swaps",
                comparison_2="same forward supports; squared loss versus logistic loss",
                limitations="fixed penalties need not be optimal for either loss; repeated exploratory test split",
                smoke=bool(cfg.get("concept_limit")),
            ))
        with stage_progress("forward_selection"):
            fit_forward(cfg, data)
        with stage_progress("support_swaps"):
            fit_refined(cfg, data)
        with stage_progress("fixed_support_logistic"):
            fit_logistic(cfg, data)
        with stage_progress("sparse_cca_controls"):
            fit_controls(cfg, data)
        with stage_progress("evaluate"):
            evaluate(cfg, data)
        with stage_progress("report"):
            from .report import build_report
            build_report(out)


if __name__ == "__main__":
    main()
