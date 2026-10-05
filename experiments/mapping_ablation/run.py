"""Isolate input scaling, covariance correction, and direct feature-group scoring.

All transform statistics come from the original fit partition. Existing SAE
weights, correspondence configurations, feature IDs, and test queries stay fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Literal, cast

import numpy as np
from scipy import linalg, sparse
import yaml

from mm_sae.analysis.data import save_json
from mm_sae.analysis.mapping_ablation import fit_common_projection, group_projection, preprocess
from mm_sae.analysis.mapping_evaluation import mapping_retrieval, paired_retrieval
from mm_sae.io import sha256
from mm_sae.metrics.regression import Moments
from mm_sae.progress import ProgressReporter, iter_progress, progress_task, stage_progress

_gemm = linalg.get_blas_funcs(("gemm",), dtype=np.float64)[0]


def product(a, b):
    return _gemm(alpha=1, a=np.asarray(a, dtype=np.float64), b=np.asarray(b, dtype=np.float64))


def load_inputs(cfg, out):
    parent, source = Path(cfg["parent_run"]), Path(cfg["source_run"])
    selected = json.loads((parent / "selected.json").read_text())
    if cfg.get("families"):
        selected = [r for r in selected if r["family"] in cfg["families"]]
    with np.load(parent / "moments.npz") as saved:
        ids = {s: saved[s + "_ids"] for s in ("image", "text")}
        fit = Moments(int(saved["fit_n"]), saved["fit_mean"], saved["fit_second"])
    files = [parent / name for name in ("moments.npz", "selected.json", "population.json", "manifest.json")]
    files += [parent / "candidates" / (r["key"] + ".npz") for r in selected]
    files += [source / "activations/val2017" / (s + ".npz") for s in ("image", "text")]
    files += [source / "index/val2017" / name for name in ("parents.npy", "images.json")]
    source_hashes = {str(p): sha256(p) for p in files}
    # Assert that the test cache still matches the one used in the original run.
    old_hashes = json.loads((parent / "manifest.json").read_text())["source_hashes"]
    for path in files:
        if not path.is_relative_to(source):
            continue
        suffix = "/" + str(path.relative_to(source))
        matches = [digest for name, digest in old_hashes.items() if name.endswith(suffix)]
        if matches and (len(set(matches)) != 1 or matches[0] != source_hashes[str(path)]):
            raise ValueError(f"Original experiment source changed: {path}")
    root = Path(__file__).resolve().parents[2]
    code_paths = [Path(__file__), root / "src/mm_sae/analysis/mapping_ablation.py",
                  root / "src/mm_sae/analysis/mapping_evaluation.py"]
    manifest = {"config": cfg, "source_hashes": source_hashes,
                "code_hashes": {str(p.relative_to(root)): sha256(p) for p in code_paths}}
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    dest = out / "manifest.json"
    if dest.exists() and json.loads(dest.read_text())["signature"] != signature:
        raise ValueError("Inputs or calculation code changed. Use a new output directory.")
    save_json(dest, {**manifest, "signature": signature})
    values = {s: sparse.load_npz(source / "activations/val2017" / (s + ".npz"))[:, ids[s]].toarray()
              for s in ("image", "text")}
    parents = np.load(source / "index/val2017/parents.npy")
    population = json.loads((parent / "population.json").read_text())
    image_records = json.loads((source / "index/val2017/images.json").read_text())
    np.testing.assert_array_equal(population["test_image_ids"], [r["image_id"] for r in image_records])
    if cfg.get("smoke_images"):
        size = min(cfg["smoke_images"], len(values["image"]))
        keep = parents < size
        values["image"] = values["image"][:size]
        values["text"] = values["text"][keep]
        parents = parents[keep]
    population.update(test_images=len(values["image"]), test_captions=len(values["text"]),
                      test_image_ids=population["test_image_ids"][:len(values["image"])])
    save_json(out / "population.json", population)
    np.save(out / "parents.npy", parents)
    return selected, ids, fit, values, parents


def evaluate(out, cfg, parents, *, key, family, experiment, condition, space, metadata,
             image=None, text=None, retrieval=None):
    dest = out / "results" / (key + ".json")
    if dest.exists():
        return
    start = time.monotonic()
    if retrieval is None:
        with progress_task(key, unit="queries"):
            retrieval = paired_retrieval(image, text, parents, device=cfg["device"],
                                         chunk_size=cfg["retrieval_chunk"])
    save_json(dest, {"key": key, "family": family, "experiment": experiment,
                    "condition": condition, "space": space, "metadata": metadata,
                    "seconds": time.monotonic() - start, "retrieval": retrieval})
    print(json.dumps({"completed": key, "seconds": time.monotonic() - start,
                      "recall": {d: m["recall"] for d, m in retrieval.items()}}), flush=True)


def input_comparison(cfg, out, selected, prepared, parents):
    parent = Path(cfg["parent_run"])
    jobs = [(r, mode) for r in selected for mode in cfg["preprocessing"]]
    for chosen, mode in iter_progress(jobs, "Compare frozen maps across input scaling", unit="conditions"):
        keys = {s: f'{chosen["family"]}__{mode}__{s}' for s in
                ("text_projected_to_image", "image_projected_to_text")}
        if all((out / "results" / (key + ".json")).exists() for key in keys.values()):
            continue
        with np.load(parent / "candidates" / (chosen["key"] + ".npz")) as saved:
            mapping = saved["mapping"]
        x, y = prepared[mode]
        with progress_task(chosen["family"] + " " + mode, unit="queries"):
            retrieval = mapping_retrieval(x, y, mapping, parents, device=cfg["device"],
                                          chunk_size=cfg["retrieval_chunk"])
        for space, score in retrieval.items():
            evaluate(out, cfg, parents, key=keys[space], family=chosen["family"],
                     experiment="preprocessing", condition=mode, space=space,
                     metadata={"source_candidate": chosen["key"], "options": chosen["options"],
                               "mapping_fixed": True}, retrieval=score)


def common_comparison(cfg, out, x, y, xx, cross, yy, parents):
    dimensions = [min(cross.shape) if d == "full" else int(d) for d in cfg["dimensions"]]
    for whiten in iter_progress([False, True], "Compare within-modality covariance correction", unit="families"):
        family = "cca" if whiten else "cross_svd"
        projection = fit_common_projection(xx, cross, yy, max(dimensions), whiten=whiten, ridge=cfg["ridge"])
        for dim in dimensions:
            key = f"{family}_{dim}"
            if (out / "results" / (key + ".json")).exists():
                continue
            wi, wt = projection.image[:, :dim], projection.text[:, :dim]
            np.savez_compressed(out / "transforms" / (key + ".npz"), image=wi, text=wt)
            evaluate(out, cfg, parents, key=key, family=family, experiment="cca_components", condition=key,
                     space="common", metadata={"dimensions": dim, "whitening": whiten, "ridge": cfg["ridge"],
                                               "preprocessing": "standardized"},
                     image=product(x, wi), text=product(y, wt))
    key = "procrustes"
    if not (out / "results" / (key + ".json")).exists():
        u, _, vt = linalg.svd(cross, full_matrices=True)
        width = max(cross.shape)
        xi = np.pad(product(x, u), ((0, 0), (0, width - x.shape[1])))
        yt = np.pad(product(y, vt.T), ((0, 0), (0, width - y.shape[1])))
        evaluate(out, cfg, parents, key=key, family=key, experiment="cca_components", condition=key,
                 space="common", metadata={"dimensions": width, "whitening": False,
                                           "unmatched_image_complement_retained": True}, image=xi, text=yt)


def group_comparison(cfg, out, selected, x, y, xx, yy, parents):
    chosen = next(r for r in selected if r["family"] == "sparse_factorization")
    with np.load(Path(cfg["parent_run"]) / "candidates" / (chosen["key"] + ".npz")) as saved:
        u, v, p = saved["u"], saved["v"], saved["mapping"]
    np.testing.assert_allclose(product(u, v.T), p, atol=1e-10, rtol=1e-10)
    for condition in iter_progress(cfg["group_conditions"], "Compare direct feature-group coordinates", unit="conditions"):
        if condition == "unnormalized_mapping":
            pairs = {"text_projected_to_image": (x, product(y, p.T)),
                     "image_projected_to_text": (product(x, p), y)}
            for space, (xi, yt) in pairs.items():
                evaluate(out, cfg, parents, key=f"groups__{condition}__{space}", family=chosen["family"],
                         experiment="feature_groups", condition=condition, space=space,
                         metadata={"dimensions": u.shape[1], "source_candidate": chosen["key"]}, image=xi, text=yt)
        else:
            if condition == "raw_factors":
                wi, wt = u, v
            elif condition in ("weighted_average", "unit_variance"):
                projection = group_projection(u, v, xx, yy, normalize_variance=condition == "unit_variance")
                wi, wt = projection.image, projection.text
            else:
                raise ValueError(f"Unknown group condition: {condition}")
            key = "groups__" + condition
            np.savez_compressed(out / "transforms" / (key + ".npz"), image=wi, text=wt)
            evaluate(out, cfg, parents, key=key, family=chosen["family"], experiment="feature_groups",
                     condition=condition, space="common", metadata={"dimensions": wi.shape[1],
                     "source_candidate": chosen["key"], "preprocessing": "standardized",
                     "group_memberships_fixed": True}, image=product(x, wi), text=product(y, wt))


def verify_recall(out):
    count = 0
    for path in sorted((out / "results").glob("*.json")):
        r = json.loads(path.read_text())
        for metric in r["retrieval"].values():
            ranks = np.asarray(metric["ranks"])
            assert len(ranks) == metric["query_count"]
            for k, score in metric["recall"].items():
                np.testing.assert_allclose(score, np.mean(ranks <= min(int(k), metric["candidate_count"])), atol=1e-14)
                count += 1
    save_json(out / "verification.json", {"verified_recalls": count, "source": "saved per-query ranks",
                                          "test_queries_not_filtered": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    for name in ("source_run", "parent_run", "output"):
        cfg[name] = str((args.config.parent / cfg[name]).resolve())
    out = Path(cfg["output"])
    for folder in (out, out / "results", out / "transforms"):
        folder.mkdir(exist_ok=True, parents=True)
    stages = ["load_frozen_inputs", "preprocessing", "cca_components", "feature_groups", "verify_and_report"]
    with ProgressReporter(out, stages, [], cfg.get("progress_interval_seconds", 10)):
        with stage_progress("load_frozen_inputs"):
            selected, ids, fit, values, parents = load_inputs(cfg, out)
            ni = len(ids["image"])
            covariance = fit.standardized(fit)
            xx, cross, yy = covariance[:ni, :ni], covariance[:ni, ni:], covariance[ni:, ni:]
            modes = set(cfg["preprocessing"]) | {"standardized"}
            if not modes <= {"raw", "centered", "standardized"}:
                raise ValueError("Unknown preprocessing mode in configuration")
            prepared = {}
            for name in modes:
                mode = cast(Literal["raw", "centered", "standardized"], name)
                prepared[mode] = (preprocess(values["image"], fit.mean[:ni], fit.scale[:ni], mode=mode),
                                  preprocess(values["text"], fit.mean[ni:], fit.scale[ni:], mode=mode))
        with stage_progress("preprocessing"):
            input_comparison(cfg, out, selected, prepared, parents)
        x, y = prepared["standardized"]
        with stage_progress("cca_components"):
            common_comparison(cfg, out, x, y, xx, cross, yy, parents)
        with stage_progress("feature_groups"):
            group_comparison(cfg, out, selected, x, y, xx, yy, parents)
        with stage_progress("verify_and_report"):
            verify_recall(out)
            from .report import report
            report(out)


if __name__ == "__main__":
    main()
