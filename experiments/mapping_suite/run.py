"""Compare correspondence constraints without retraining the encoder or the SAEs.

Run with python -m experiments.mapping_suite.run --config configs/mapping-server.yaml.
Each image and all its captions stay in the same fit/tune partition. Test data never
select a method setting. Original signed Pearson scores are shared across methods.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import linalg
import yaml

from mm_sae.analysis.data import StudyData, save_json
from mm_sae.analysis.mapping import fit_mapping
from mm_sae.analysis.mapping_evaluation import (
    fit_nonnegative_ridge, mapping_retrieval, mapping_structure, paired_retrieval,
)
from mm_sae.io import sha256, write_csv
from mm_sae.metrics.regression import Moments, prediction_mse, sparse_moments, varying_columns
from mm_sae.progress import ProgressReporter, iter_progress, progress_task, stage_progress

from .paired_data import PairedStudyData


def skips_removal(config: dict[str, Any]) -> bool:
    mode = config.get("data_mode", "annotated")
    if mode not in {"annotated", "paired"}:
        raise ValueError(f"Unknown mapping data_mode: {mode}")
    skip = config.get("skip_removal", mode == "paired")
    if not isinstance(skip, bool):
        raise ValueError("skip_removal must be a boolean")
    if mode == "paired" and not skip:
        raise ValueError("Paired caches have no removal annotations; set skip_removal: true")
    return skip


def split_blocks(matrix: np.ndarray, ni: int):
    return matrix[:ni, :ni], matrix[:ni, ni:], matrix[ni:, ni:]


def calibrated_coefficients(p: np.ndarray, xx: np.ndarray, xy: np.ndarray,
                            penalty: float = 1e-6) -> np.ndarray:
    """Preserve mapping weight ratios; fit only one nonnegative scale per target."""
    total = np.sum(np.abs(p), axis=0)
    base = np.divide(p, total, out=np.zeros_like(p), where=total > 0)
    energy = np.sum(base * (xx @ base), axis=0)
    gain = np.divide(np.maximum(np.sum(base * xy, axis=0), 0), energy + penalty,
                     out=np.zeros_like(energy), where=energy + penalty > 0)
    return base * gain


def r2_and_error(coef, moments: Moments, reference: Moments, ni: int, reverse=False):
    xx, xy, yy = split_blocks(moments.standardized(reference), ni)
    mean = (moments.mean - reference.mean) / reference.scale
    if reverse:
        xx, xy, yy, mean_y = yy, xy.T, xx, mean[:ni]
    else:
        mean_y = mean[ni:]
    mse = prediction_mse(coef, xx, xy, yy)
    variance = np.diag(yy) - mean_y**2
    r2 = np.full_like(mse, np.nan)
    np.divide(mse, variance, out=r2, where=variance > 1e-12)
    return 1 - r2, mse


def candidate_settings(families: list[dict[str, Any]]):
    result = []
    for family in families:
        grid = family.get("grid", {})
        keys = sorted(grid)
        for number, values in enumerate(itertools.product(*(grid[k] for k in keys))):
            options = dict(family.get("options", {}))
            options.update(zip(keys, values, strict=True))
            result.append({"family": family["name"], "method": family["method"],
                           "key": f'{family["name"]}_{number:03d}', "options": options})
    return result


def check_manifest(config: dict[str, Any], out: Path, source: Path):
    skip_removal = skips_removal(config)
    paired = config.get("data_mode") == "paired"
    files = [source / "models/frozen.json"]
    if not paired or (source / "dataset.json").exists():
        files.append(source / "dataset.json")
    for name in ["train2017", "val2017"]:
        files.extend(source / "activations" / name / (side + ".npz") for side in ["image", "text"])
        index_files = ["images.json", "parents.npy"] if paired else [
            "images.json", "concept_ids.json", "parents.npy", "presence.npy", "mentions.npy"]
        files.extend(source / "index" / name / v for v in index_files)
    if not skip_removal:
        files += sorted((source / "counterfactual/val2017").glob("*/*_rows.npy"))
        files += sorted((source / "counterfactual/val2017").glob("*/*_activations.npz"))
    root = Path(__file__).resolve().parents[2]
    code = list((root / "experiments/mapping_suite").glob("*.py"))
    code += [root / "src/mm_sae/analysis" / f for f in
             ["mapping.py", "mapping_evaluation.py", "data.py"]]
    record = {"config": config, "source_hashes": {str(p): sha256(p) for p in files},
              "code_hashes": {str(p.relative_to(root)): sha256(p) for p in code}}
    signature = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    path = out / "manifest.json"
    if path.exists() and json.loads(path.read_text())["signature"] != signature:
        raise ValueError("Configuration, source data or code changed; choose a new output directory")
    save_json(path, {**record, "signature": signature})


def prepare(data: StudyData | PairedStudyData, cfg: dict[str, Any], out: Path):
    path = out / "moments.npz"
    if path.exists():
        with np.load(path) as a:
            ids = {s: a[s + "_ids"] for s in ["image", "text"]}
            moments = {p: Moments(int(a[p + "_n"]), a[p + "_mean"], a[p + "_second"])
                       for p in ["fit", "tune", "test"]}
        return ids, moments
    ids = {s: varying_columns(data.train.activations[s][data.fit[s]]) for s in ["image", "text"]}
    limit = cfg.get("smoke_features")
    if limit:
        # Choose frequent fit features, never test-derived features.
        for side in ids:
            count = np.asarray(data.train.activations[side][data.fit[side]].getnnz(0)).ravel()
            ids[side] = np.sort(ids[side][np.argsort(-count[ids[side]], kind="stable")[:limit]])
    moments = {}
    for partition in ["fit", "tune", "test"]:
        with progress_task(f"Compute {partition} covariance", unit="matrix"):
            split = data.test if partition == "test" else data.train
            mask = np.ones(len(split.parents), bool) if partition == "test" else getattr(data, partition)["text"]
            rows = np.flatnonzero(mask)
            limit_rows = cfg.get("smoke_pairs")
            if limit_rows:
                rows = rows[:limit_rows]
            x = split.activations["image"][split.parents[rows]][:, ids["image"]]
            y = split.activations["text"][rows][:, ids["text"]]
            moments[partition] = sparse_moments(x, y)
    np.savez_compressed(path, **{s + "_ids": a for s, a in ids.items()},
                        **{p + "_" + k: getattr(m, k) for p, m in moments.items()
                           for k in ["n", "mean", "second"]})
    save_json(out / "population.json", {
        "fit_images": int(data.fit["image"].sum()), "tune_images": int(data.tune["image"].sum()),
        "test_images": len(data.test.image_ids), "test_captions": len(data.test.parents),
        "fit_caption_pairs": moments["fit"].n, "tune_caption_pairs": moments["tune"].n,
        "image_features": len(ids["image"]), "text_features": len(ids["text"]),
        "test_features_constant_in_fit": "excluded using fit variance only",
        "fit_image_ids": data.train.image_ids[data.fit["image"]],
        "tune_image_ids": data.train.image_ids[data.tune["image"]],
        "test_image_ids": data.test.image_ids,
    })
    return ids, moments


def fit_candidates(cfg, out, moments, ni):
    fit, tune = moments["fit"], moments["tune"]
    xx, c, yy = split_blocks(fit.standardized(fit), ni)
    choices = candidate_settings(cfg["families"])
    records = []
    root = out / "candidates"
    root.mkdir(exist_ok=True)
    for candidate in iter_progress(choices, "Fit and tune correspondence methods", unit="settings"):
        if candidate["options"].get("edge_budget") == "one_to_one":
            candidate["options"]["edge_budget"] = min(c.shape)
        if candidate["options"].get("k_image") == "all":
            candidate["options"]["k_image"] = c.shape[1]
        if candidate["options"].get("k_text") == "all":
            candidate["options"]["k_text"] = c.shape[0]
        path = root / (candidate["key"] + ".json")
        if path.exists():
            records.append(json.loads(path.read_text()))
            continue
        started = time.monotonic()
        with progress_task(candidate["key"], unit="iterations") as task:
            def callback(info):
                task.update(int(info.get("iterations", info.get("iteration", 0))), **info)
            result = fit_mapping(c, candidate["method"], candidate["options"], progress=callback)
        p = result.weights
        forward = calibrated_coefficients(p, xx, c)
        reverse = calibrated_coefficients(p.T, yy, c.T)
        _, ferror = r2_and_error(forward, tune, fit, ni)
        _, rerror = r2_and_error(reverse, tune, fit, ni, True)
        record = {**candidate, "seconds": time.monotonic() - started,
                  "tune_image_to_text_mse": float(ferror.mean()),
                  "tune_text_to_image_mse": float(rerror.mean()),
                  "tune_objective": float((ferror.mean() + rerror.mean()) / 2),
                  "solver": result.metadata, "structure": mapping_structure(p)}
        arrays = {"mapping": p, "forward": forward, "reverse": reverse}
        if result.u is not None and result.v is not None:
            arrays.update(u=result.u, v=result.v)
        np.savez_compressed(root / (candidate["key"] + ".npz"), **arrays)
        save_json(path, record)
        print(json.dumps({"completed": candidate["key"], "tune_mse": record["tune_objective"],
                          "seconds": record["seconds"]}), flush=True)
        records.append(record)
    write_csv(out / "tuning.csv", [{k: v for k, v in r.items() if k not in ["solver", "structure", "options"]}
                                  for r in records])
    selected = []
    for family in cfg["families"]:
        eligible = [r for r in records if r["family"] == family["name"]
                    and r["solver"].get("converged", True)]
        if not eligible:
            raise RuntimeError(f'No converged setting for {family["name"]}; inspect solver diagnostics')
        selected.append(min(eligible, key=lambda r: (r["tune_objective"], r["key"])))
    save_json(out / "selected.json", selected)
    return selected


def retrieval_arrays(data, ids, cfg):
    split = data.test
    image_rows = np.arange(len(split.image_ids))
    if cfg.get("smoke_test_images"):
        image_rows = image_rows[:cfg["smoke_test_images"]]
    text_rows = np.flatnonzero(np.isin(split.parents, image_rows))
    return (split.activations["image"][image_rows][:, ids["image"]].toarray(),
            split.activations["text"][text_rows][:, ids["text"]].toarray(),
            np.searchsorted(image_rows, split.parents[text_rows]))


def evaluate_selected(cfg, out, data, ids, moments, selected):
    ni = len(ids["image"])
    fit = moments["fit"]
    xx, c, yy = split_blocks(fit.standardized(fit), ni)
    x, y, parents = retrieval_arrays(data, ids, cfg)
    root = out / "evaluation"
    root.mkdir(exist_ok=True)
    records = []
    dataset = cfg.get("evaluation_dataset", "paired evaluation images" if cfg.get("data_mode") == "paired"
                      else "held-out COCO")
    for chosen in iter_progress(selected, f"Evaluate selected methods on {dataset}", unit="methods"):
        dest = root / (chosen["family"] + ".json")
        if dest.exists():
            records.append(json.loads(dest.read_text()))
            continue
        with np.load(out / "candidates" / (chosen["key"] + ".npz")) as saved:
            p, forward, reverse = saved["mapping"], saved["forward"], saved["reverse"]
        record = {"family": chosen["family"], "key": chosen["key"],
                  "options": chosen["options"], "structure": chosen["structure"], "prediction": {}}
        coefficients = {"weighted_forward": forward, "weighted_reverse": reverse}
        for direction, coef in [("image_to_text", forward), ("text_to_image", reverse)]:
            r2, mse = r2_and_error(coef, moments["test"], fit, ni, direction == "text_to_image")
            record["prediction"][direction] = {"r2": r2, "mse": mse,
                "mean_r2": float(np.nanmean(r2)), "mean_mse": float(mse.mean()),
                "targets": len(mse), "constant_test_targets": int(np.sum(~np.isfinite(r2)))}
        with progress_task(f'Retrieval {chosen["family"]}', unit="queries"):
            record["retrieval"] = mapping_retrieval(x, y, p, parents, device=cfg["device"],
                                                   chunk_size=cfg["retrieval_chunk"])
        if cfg.get("support_regression", True):
            for direction, a, b, mask in [("forward", xx, c, p > 0),
                                            ("reverse", yy, c.T, p.T > 0)]:
                best = None
                for penalty in cfg["ridge_penalties"]:
                    with progress_task(f'Support regression {chosen["family"]} {direction}', unit="fit"):
                        model = fit_nonnegative_ridge(a, b, mask, penalty,
                            max_iter=cfg["ridge_max_iter"], tolerance=cfg["ridge_tolerance"], device=cfg["device"])
                    _, errors = r2_and_error(model.coefficients, moments["tune"], fit, ni, direction == "reverse")
                    if model.converged and (best is None or errors.mean() < best[0]):
                        best = (float(errors.mean()), penalty, model)
                if best is None:
                    raise RuntimeError(f'No converged support regression: {chosen["family"]}/{direction}')
                coefficients["support_" + direction] = best[2].coefficients
                r2, mse = r2_and_error(best[2].coefficients, moments["test"], fit, ni, direction == "reverse")
                record["prediction"]["support_" + direction] = {
                    "penalty": best[1], "tune_mse": best[0], "r2": r2, "mse": mse,
                    "mean_r2": float(np.nanmean(r2)), "mean_mse": float(mse.mean()),
                    "iterations": best[2].iterations, "kkt_residual": best[2].kkt_residual}
        np.savez_compressed(root / (chosen["family"] + "_coefficients.npz"), **coefficients)
        save_json(dest, record)
        records.append(record)
    return records


def dense_references(cfg, out, data, ids, moments):
    """CCA and rectangular orthogonal alignment are separate dense-space references."""
    dest = out / "dense_references.json"
    if dest.exists():
        return
    ni = len(ids["image"])
    fit = moments["fit"]
    xx, c, yy = split_blocks(fit.standardized(fit), ni)
    x, y, parents = retrieval_arrays(data, ids, cfg)
    x = (x - fit.mean[:ni]) / fit.scale[:ni]
    y = (y - fit.mean[ni:]) / fit.scale[ni:]
    # Zero-pad the smaller representation before square orthogonal Procrustes.
    # Keep the unmatched orthogonal complement so source norms remain unchanged.
    u, _, vt = linalg.svd(c, full_matrices=True)
    width = max(c.shape)
    xi = np.pad(x @ u, ((0, 0), (0, width - ni)))
    yt = np.pad(y @ vt.T, ((0, 0), (0, width - c.shape[1])))
    records = {"procrustes": paired_retrieval(xi, yt, parents,
                                               chunk_size=cfg["retrieval_chunk"], device=cfg["device"])}
    for ridge in cfg.get("cca_penalties", [0.01]):
        a = linalg.cholesky(xx + ridge * np.eye(ni), lower=True)
        b = linalg.cholesky(yy + ridge * np.eye(yy.shape[0]), lower=True)
        cross = linalg.solve_triangular(a, c, lower=True)
        cross = linalg.solve_triangular(b, cross.T, lower=True).T
        u, _, vt = linalg.svd(cross, full_matrices=False)
        for requested in cfg.get("cca_dimensions", [64, 256]):
            dim = min(requested, min(c.shape))
            wi = linalg.solve_triangular(a.T, u[:, :dim], lower=False)
            wt = linalg.solve_triangular(b.T, vt.T[:, :dim], lower=False)
            records[f"cca_{dim}_ridge_{ridge}"] = paired_retrieval(
                x @ wi, y @ wt, parents, chunk_size=cfg["retrieval_chunk"], device=cfg["device"])
    save_json(dest, {"results": records,
                    "protocol": "Fit-centered standardized activations; zero-padded square orthogonal Procrustes preserves all dimensions. CCA/procrustes are dense representation references, not positive correspondence graphs."})


def removal_evaluation(cfg, out, data, ids, moments, selected):
    root = out / "removal"
    root.mkdir(exist_ok=True)
    split = data.test
    ni = len(ids["image"])
    scale_i, scale_t = moments["fit"].scale[:ni], moments["fit"].scale[ni:]
    models = {}
    for chosen in selected:
        with np.load(out / "evaluation" / (chosen["family"] + "_coefficients.npz")) as saved:
            models[chosen["family"]] = {k: saved[k] for k in saved.files}
    cids = split.ids[:cfg.get("smoke_categories", len(split.ids))]
    for cid in iter_progress(cids, "Evaluate category removal responses", unit="categories"):
        dest = root / f"{cid}.json"
        if dest.exists():
            continue
        ir, im = split.removed("image", cid)
        tr, tm = split.removed("text", cid)
        valid = np.asarray(split.presence[split.parents[tr], split.columns[cid]], bool)
        tr, tm = tr[valid], tm[valid]
        if not len(tr):
            save_json(dest, {"category_id": cid, "name": data.names[cid], "rows": 0, "records": []})
            continue
        parents = split.parents[tr]
        positions = np.searchsorted(ir, parents)
        if not np.array_equal(ir[positions], parents):
            raise ValueError(f"Missing image removal rows for category {cid}")
        dx = ((split.activations["image"][parents] - im[positions])[:, ids["image"]]
              .multiply(1 / scale_i)).toarray()
        dy = ((split.activations["text"][tr] - tm)[:, ids["text"]]
              .multiply(1 / scale_t)).toarray()
        unique, first, inv, counts = np.unique(parents, return_index=True, return_inverse=True, return_counts=True)
        # Average the captions mentioning this category within each parent image.
        # Both observed and shuffled evaluation therefore use one response per image.
        grouped_y = np.zeros((len(unique), dy.shape[1]))
        np.add.at(grouped_y, inv, dy)
        dy = grouped_y / counts[:, None]
        dx = dx[first]
        rng = np.random.default_rng(cfg["seed"] + int(cid))
        order = rng.permutation(len(unique))
        mapped = np.empty(len(unique), int)
        mapped[order] = np.roll(order, 1)
        records = []
        for family, coefficients in models.items():
            for kind, coef in coefficients.items():
                reverse = kind.endswith("reverse")
                a, b = (dy, dx) if reverse else (dx, dy)
                pred = a @ coef
                energy = np.sum(b**2) / len(unique)
                error = np.sum((pred - b)**2) / len(unique)
                # Move whole parent-image response groups; no sample crosses category boundaries.
                shuffled_pred = (a[mapped] @ coef) if len(unique) > 1 else None
                shuffled_error = (np.sum((shuffled_pred - b)**2) / len(unique)
                                  if shuffled_pred is not None else None)
                records.append({"family": family, "readout": kind,
                    "direction": "text_to_image" if reverse else "image_to_text",
                    "relative_mse": float(error / energy) if energy > 1e-12 else None,
                    "shuffled_relative_mse": float(shuffled_error / energy)
                    if shuffled_error is not None and energy > 1e-12 else None,
                    "target_change_energy": float(energy)})
        save_json(dest, {"category_id": cid, "name": data.names[cid],
                         "kind": "object" if cid < 91 else "background",
                         "rows": len(tr), "images": len(unique), "records": records})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    skip_removal = skips_removal(cfg)
    for key in ["source_run", "output"]:
        cfg[key] = str((args.config.parent / cfg[key]).resolve())
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    stages = ["cache_and_moments", "fit_and_tune", "heldout_evaluation", "dense_references"]
    if not skip_removal:
        stages.append("removal")
    stages.append("report")
    with ProgressReporter(out, stages, [], cfg.get("progress_interval_seconds", 10)):
        with stage_progress("cache_and_moments"):
            check_manifest(cfg, out, Path(cfg["source_run"]))
            data_class = PairedStudyData if cfg.get("data_mode") == "paired" else StudyData
            data = data_class(cfg["source_run"], {"fit_and_tune": "train2017", "evaluation": "val2017",
                "seed": cfg["seed"], "tune_image_fraction": cfg["tune_image_fraction"]})
            ids, moments = prepare(data, cfg, out)
        with stage_progress("fit_and_tune"):
            selected = fit_candidates(cfg, out, moments, len(ids["image"]))
        with stage_progress("heldout_evaluation"):
            evaluate_selected(cfg, out, data, ids, moments, selected)
        with stage_progress("dense_references"):
            dense_references(cfg, out, data, ids, moments)
        if not skip_removal:
            with stage_progress("removal"):
                removal_evaluation(cfg, out, data, ids, moments, selected)
        with stage_progress("report"):
            from .report import report
            report(out)


if __name__ == "__main__":
    main()
