"""Hold common directions fixed and vary their contribution to cosine retrieval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
import yaml

from experiments.corpus_comparison.isolate import score_diagnostics
from mm_sae.analysis.data import save_json
from mm_sae.analysis.mapping_evaluation import paired_retrieval
from mm_sae.io import sha256, write_csv
from mm_sae.metrics.regression import Moments
from mm_sae.progress import ProgressReporter, progress_task, stage_progress


def weights_from_fit(fit, ni, wi, wt):
    covariance = fit.standardized(fit)
    vi = np.sum(wi * (covariance[:ni, :ni] @ wi), axis=0)
    vt = np.sum(wt * (covariance[ni:, ni:] @ wt), axis=0)
    cross = np.sum(wi * (covariance[:ni, ni:] @ wt), axis=0)
    if np.any(vi <= 0) or np.any(vt <= 0):
        raise ValueError("Every selected direction must vary in training")
    rho = cross / np.sqrt(vi * vt)
    return vi, vt, rho


def run(cfg):
    source, root = Path(cfg["source_run"]), Path(cfg["output"])
    out = root / "output-weighting"
    out.mkdir(exist_ok=True)
    transforms = [root / f"native__fit_{name}__scale_{name}__{method}.npz"
                  for name in cfg["mapping_runs"] for method in ("cca", "cross_svd")]
    paths = [Path(p) / "moments.npz" for p in cfg["mapping_runs"].values()]
    paths += transforms + [source / "activations/val2017" / f"{s}.npz" for s in ("image", "text")]
    paths += [source / "index/val2017/parents.npy"]
    signature = dict(config=cfg, sources={str(p): sha256(p) for p in paths}, code=sha256(Path(__file__)))
    receipt = out / "manifest.json"
    if receipt.exists() and json.loads(receipt.read_text()) != signature:
        raise ValueError("Input changes require a new output directory")
    save_json(receipt, signature)
    with ProgressReporter(out, ["evaluate"], [], 10), stage_progress("evaluate"):
        values = {s: sparse.load_npz(source / "activations/val2017" / f"{s}.npz") for s in ("image", "text")}
        parents = np.load(source / "index/val2017/parents.npy")
        for name, mapping in cfg["mapping_runs"].items():
            with np.load(Path(mapping) / "moments.npz") as z:
                fit = Moments(int(z["fit_n"]), z["fit_mean"], z["fit_second"])
                ids = {s: z[s + "_ids"] for s in ("image", "text")}
            ni = len(ids["image"])
            for method in ("cca", "cross_svd"):
                with np.load(root / f"native__fit_{name}__scale_{name}__{method}.npz") as z:
                    wi, wt = z["image"], z["text"]
                vi, vt, rho = weights_from_fit(fit, ni, wi, wt)
                write_csv(out / f"{name}__{method}__fit_coordinates.csv", [
                    dict(coordinate=i, image_variance=float(a), text_variance=float(b), correlation=float(c))
                    for i, (a, b, c) in enumerate(zip(vi, vt, rho, strict=True))])
                image = ((values["image"][:, ids["image"]].toarray() - fit.mean[:ni]) / fit.scale[:ni]) @ wi
                text = ((values["text"][:, ids["text"]].toarray() - fit.mean[ni:]) / fit.scale[ni:]) @ wt
                for weighting in ("unit_training_variance", "unit_variance_times_sqrt_training_correlation"):
                    key = f"{name}__{method}__{weighting}"
                    dest = out / f"{key}.json"
                    if dest.exists():
                        continue
                    with progress_task(key, unit="conditions"):
                        weight = np.ones_like(rho) if weighting == "unit_training_variance" else np.sqrt(np.clip(rho, 0, 1))
                        xi, yt = image * (weight / np.sqrt(vi)), text * (weight / np.sqrt(vt))
                        retrieval = paired_retrieval(xi, yt, parents, device=cfg["device"], chunk_size=cfg["chunk_size"])
                        diagnostics, contributions = score_diagnostics(xi, yt, parents, cfg["chunk_size"], cfg["device"])
                        write_csv(out / f"{key}-contributions.csv", contributions)
                        save_json(dest, dict(corpus=name, method=method, weighting=weighting,
                                             retrieval=retrieval, diagnostics=diagnostics))
                    print(json.dumps(dict(completed=key)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    for key in ("source_run", "output"):
        cfg[key] = str((args.config.parent / cfg[key]).resolve())
    cfg["mapping_runs"] = {n: str((args.config.parent / p).resolve()) for n, p in cfg["mapping_runs"].items()}
    run(cfg)
