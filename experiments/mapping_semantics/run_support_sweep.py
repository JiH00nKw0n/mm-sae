"""Evaluate support-size curves on frozen SAE activations and training moments."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import yaml

from experiments.mapping_ablation.run import load_inputs, product
from mm_sae.analysis.mapping_ablation import preprocess
from mm_sae.analysis.mapping_evaluation import mapping_retrieval, paired_retrieval
from mm_sae.analysis.mapping_pruning import prune_coefficients
from mm_sae.analysis.sparse_cca import fit_sparse_cca
from mm_sae.analysis.sparse_pls import fit_sparse_pls
from mm_sae.io import atomic_json, sha256


def compact_retrieval(result):
    return {direction: {key: value for key, value in metrics.items()
                        if key not in ("ranks", "hits")}
            for direction, metrics in result.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--family", choices=["fast", "cca"], required=True)
    parser.add_argument("--condition")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    root = (args.config.parent / config["output"]).resolve()
    families = ["sinkhorn", "pls"] if args.family == "fast" else ["cca"]
    for condition in config["conditions"]:
        if args.condition and condition["name"] != args.condition:
            continue
        cfg = {**config, **condition}
        for key in ("source_run", "parent_run", "projection_run", "sinkhorn_transform"):
            cfg[key] = str((args.config.parent / cfg[key]).resolve())
        out = root / condition["name"]
        results = out / "results"
        results.mkdir(parents=True, exist_ok=True)
        jobs = []
        for family in families:
            for k in config["supports"]:
                keys = ([f"sinkhorn_text_{k}", f"sinkhorn_image_{k}"] if family == "sinkhorn"
                        else [f"{family}_{k}"])
                if not all((results / f"{key}.json").exists() for key in keys):
                    jobs.append((family, k))
        if not jobs:
            continue
        progress_path = out / f"progress-{args.family}.json"
        atomic_json(progress_path, dict(state="loading", jobs=len(jobs), updated_at=time.time()))
        load_out = out / f"inputs-{args.family}"
        load_out.mkdir(exist_ok=True)
        _, ids, fit, values, parents = load_inputs(cfg, load_out)
        ni = len(ids["image"])
        covariance = fit.standardized(fit)
        xx, cross, yy = covariance[:ni, :ni], covariance[:ni, ni:], covariance[ni:, ni:]
        x = preprocess(values["image"], fit.mean[:ni], fit.scale[:ni], "standardized")
        y = preprocess(values["text"], fit.mean[ni:], fit.scale[ni:], "standardized")
        for job_index, (family, k) in enumerate(jobs):
            started = time.monotonic()

            def status(state, **extra):
                record = dict(state=state, family=family, support=k, jobs_completed=job_index,
                              total_jobs=len(jobs), elapsed_seconds=time.monotonic()-started,
                              updated_at=time.time(), **extra)
                atomic_json(progress_path, record)
                return record

            if family == "sinkhorn":
                source = Path(cfg["sinkhorn_transform"])
                with np.load(source) as z:
                    mapping = prune_coefficients(z["mapping"], k, rule="mutual")
                assert ((mapping != 0).sum(0) <= k).all()
                assert ((mapping != 0).sum(1) <= k).all()
                status("evaluating")
                metrics = mapping_retrieval(x, y, mapping, parents, device=cfg["device"],
                                            chunk_size=cfg["retrieval_chunk"])
                for name, space in [("text", "text_projected_to_image"),
                                    ("image", "image_projected_to_text")]:
                    atomic_json(results / f"sinkhorn_{name}_{k}.json", dict(
                        key=f"sinkhorn_{name}_{k}", support=k,
                        metadata=dict(method="mutual_row_column_topk_after_sinkhorn",
                                      epsilon=.05, source=str(source), source_sha256=sha256(source)),
                        retrieval=compact_retrieval(metrics[space])))
            else:
                initial_path = Path(cfg["projection_run"]) / "transforms" / (
                    "cca_256.npz" if family == "cca" else "cross_svd_256.npz")
                with np.load(initial_path) as z:
                    a, b = z["image"], z["text"]
                transform_path = out / f"{family}_{k}.npz"
                metadata_path = out / f"{family}_{k}.json"
                if transform_path.exists() and metadata_path.exists():
                    with np.load(transform_path) as z:
                        wi, wt = z["image"], z["text"]
                    metadata = json.loads(metadata_path.read_text())
                else:
                    def progress(record):
                        done = record["component"] + 1
                        elapsed = time.monotonic() - started
                        event = status("fitting", completed_components=done, total_components=a.shape[1],
                                       fit_eta_seconds=elapsed / done * (a.shape[1]-done))
                        if done % 16 == 0:
                            print(json.dumps({"condition": condition["name"], **event}), flush=True)

                    if family == "cca":
                        model = fit_sparse_cca(xx, cross, yy, a.shape[1], k=k, ridge=cfg["ridge"],
                                               initial_image=a, initial_text=b,
                                               max_iter=cfg["max_iter"], progress=progress)
                    else:
                        model = fit_sparse_pls(cross, a, b, k=k, max_iter=cfg["max_iter"],
                                               progress=progress)
                    wi, wt, metadata = model.image, model.text, model.metadata
                    solver = Path(__file__).parents[2] / f"src/mm_sae/analysis/sparse_{family}.py"
                    metadata.update(initial_sha256=sha256(initial_path), solver_sha256=sha256(solver),
                                    fit_seconds=time.monotonic()-started)
                    np.savez_compressed(transform_path, image=wi, text=wt,
                                        image_ids=ids["image"], text_ids=ids["text"])
                    atomic_json(metadata_path, metadata)
                assert ((wi != 0).sum(0) <= k).all() and ((wt != 0).sum(0) <= k).all()
                status("evaluating")
                metrics = paired_retrieval(product(x, wi), product(y, wt), parents,
                                            device=cfg["device"], chunk_size=cfg["retrieval_chunk"])
                atomic_json(results / f"{family}_{k}.json", dict(
                    key=f"{family}_{k}", support=k, metadata=metadata,
                    retrieval=compact_retrieval(metrics)))
            print(json.dumps({"condition": condition["name"], **status("job_completed")}), flush=True)
        atomic_json(progress_path, dict(state="completed", jobs_completed=len(jobs),
                                       total_jobs=len(jobs), updated_at=time.time()))


if __name__ == "__main__":
    main()
