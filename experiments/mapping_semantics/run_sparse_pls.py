"""Fit sparse PLS on saved training moments and evaluate frozen COCO retrieval."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import yaml

from experiments.mapping_ablation.run import evaluate, load_inputs, product
from mm_sae.analysis.mapping_ablation import preprocess
from mm_sae.analysis.sparse_pls import fit_sparse_pls
from mm_sae.io import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    for condition in config["conditions"]:
        cfg = {**config, **condition}
        for key in ("source_run", "parent_run", "projection_run", "output"):
            cfg[key] = str((args.config.parent / cfg[key]).resolve())
        out = Path(cfg["output"])
        (out / "results").mkdir(parents=True, exist_ok=True)
        key = f'sparse_pls_{cfg["support"]}'
        if (out / "results" / f"{key}.json").exists():
            continue
        _, ids, fit, values, parents = load_inputs(cfg, out)
        ni = len(ids["image"])
        covariance = fit.standardized(fit)
        initial_path = Path(cfg["projection_run"]) / "transforms/cross_svd_256.npz"
        with np.load(initial_path) as initial:
            a, b = initial["image"], initial["text"]
        started = time.monotonic()

        def progress(record):
            done = record["component"] + 1
            elapsed = time.monotonic() - started
            status = dict(state="fitting", completed=done, total=a.shape[1],
                          elapsed_seconds=elapsed, eta_seconds=elapsed / done * (a.shape[1] - done))
            atomic_json(out / "progress.json", status)
            if done % 16 == 0:
                print(json.dumps({"condition": condition["name"], **status}), flush=True)

        model = fit_sparse_pls(covariance[:ni, ni:], a, b, k=cfg["support"],
                               max_iter=cfg["max_iter"], progress=progress)
        model.metadata.update(initial_sha256=sha256(initial_path),
                              solver_sha256=sha256(Path(__file__).parents[2] /
                                                   "src/mm_sae/analysis/sparse_pls.py"))
        np.savez_compressed(out / "transform.npz", image=model.image, text=model.text,
                            image_ids=ids["image"], text_ids=ids["text"])
        atomic_json(out / "fit.json", model.metadata)
        atomic_json(out / "progress.json", dict(state="evaluating", completed=a.shape[1]))
        x = preprocess(values["image"], fit.mean[:ni], fit.scale[:ni], "standardized")
        y = preprocess(values["text"], fit.mean[ni:], fit.scale[ni:], "standardized")
        evaluate(out, cfg, parents, key=key, family="sparse_pls", experiment="sparse_support",
                 condition=f'{cfg["support"]} inputs per side per common coordinate', space="common",
                 metadata=model.metadata, image=product(x, model.image), text=product(y, model.text))
        atomic_json(out / "progress.json", dict(state="completed", elapsed_seconds=time.monotonic()-started))


if __name__ == "__main__":
    main()
