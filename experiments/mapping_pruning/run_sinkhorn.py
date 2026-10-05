"""Sweep entropy concentration and hard support size on fixed training moments."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import yaml

from experiments.mapping_ablation.run import evaluate, load_inputs, verify_recall
from mm_sae.analysis.data import save_json
from mm_sae.analysis.mapping import fit_mapping
from mm_sae.analysis.mapping_ablation import preprocess
from mm_sae.analysis.mapping_evaluation import mapping_retrieval
from mm_sae.analysis.mapping_pruning import prune_coefficients
from mm_sae.analysis.transport_diagnostics import transport_summary
from mm_sae.io import sha256
from mm_sae.progress import ProgressReporter, iter_progress, progress_task, stage_progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    for key in ("source_run", "parent_run", "output"):
        cfg[key] = str((args.config.parent / cfg[key]).resolve())
    out = Path(cfg["output"])
    for folder in (out, out / "results", out / "transforms", out / "solvers"):
        folder.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[2]
    paths = [Path(__file__)]
    paths += [repository / "src/mm_sae/analysis" / name for name in
              ("mapping.py", "mapping_pruning.py", "transport_diagnostics.py", "mapping_evaluation.py")]
    fingerprint = {str(path): sha256(path) for path in paths}
    destination = out / "sinkhorn_sources.json"
    if destination.exists() and json.loads(destination.read_text()) != fingerprint:
        raise ValueError("Analysis code changed. Use a new output directory.")
    save_json(destination, fingerprint)
    stages = ["load", "fit_and_evaluate", "verify"]
    with ProgressReporter(out, stages, [], 10):
        with stage_progress("load"):
            _, ids, fit, values, parents = load_inputs(cfg, out)
            ni = len(ids["image"])
            x = preprocess(values["image"], fit.mean[:ni], fit.scale[:ni], "standardized")
            y = preprocess(values["text"], fit.mean[ni:], fit.scale[ni:], "standardized")
            covariance = fit.standardized(fit)
            correlation = covariance[:ni, ni:]
        with stage_progress("fit_and_evaluate"):
            jobs = [(epsilon, k) for epsilon in cfg["epsilon_grid"] for k in [None, *cfg["k_grid"]]]
            original = None
            solver = None
            last_epsilon = None
            for epsilon, k in iter_progress(jobs, "Sinkhorn epsilon and support limits", unit="conditions"):
                prefix = f'sinkhorn_e{epsilon}_k{"full" if k is None else k}'
                spaces = ("text_projected_to_image", "image_projected_to_text")
                if all((out / "results" / f'{prefix}__{space}.json').exists() for space in spaces):
                    continue
                if epsilon != last_epsilon:
                    weight_path = out / "transforms" / f'sinkhorn_e{epsilon}_full.npz'
                    solver_path = out / "solvers" / f'sinkhorn_e{epsilon}.json'
                    if weight_path.exists() and solver_path.exists():
                        with np.load(weight_path) as saved:
                            original = saved["mapping"]
                        solver = json.loads(solver_path.read_text())
                    else:
                        start = time.monotonic()

                        def log(event):
                            if event.get("status") or event.get("iteration", 0) % 1000 == 0:
                                print(json.dumps({"epsilon": epsilon, **event}), flush=True)

                        with progress_task(f"Fit Sinkhorn epsilon={epsilon}", unit="iterations"):
                            fitted = fit_mapping(correlation, "sinkhorn",
                                                 {"epsilon": epsilon, "tol": cfg["tolerance"],
                                                  "max_iter": cfg["max_iter"]}, progress=log)
                        original, solver = fitted.weights, fitted.metadata
                        solver["fit_seconds"] = time.monotonic() - start
                        save_json(solver_path, solver)
                        np.savez_compressed(weight_path, mapping=original)
                    if not solver["converged"]:
                        raise RuntimeError(f"Sinkhorn epsilon={epsilon} did not converge; inspect {solver_path}")
                    last_epsilon = epsilon
                assert original is not None and solver is not None
                mapping = prune_coefficients(original, k, rule="mutual")
                if k is not None:
                    assert ((mapping > 0).sum(0) <= k).all() and ((mapping > 0).sum(1) <= k).all()
                np.savez_compressed(out / "transforms" / (prefix + ".npz"), mapping=mapping)
                with progress_task(prefix, unit="queries"):
                    scores = mapping_retrieval(x, y, mapping, parents, device=cfg["device"],
                                               chunk_size=cfg["retrieval_chunk"])
                for space in spaces:
                    evaluate(out, cfg, parents, key=f'{prefix}__{space}', family="sinkhorn",
                             experiment="epsilon_and_pruning", condition=prefix, space=space,
                             retrieval=scores[space],
                             metadata={"epsilon": epsilon, "k": k, "solver": solver,
                                       "structure": transport_summary(mapping, original),
                                       "support_rule": "Full" if k is None else "Mutual row/column top-k",
                                       "preprocessing": "Fixed fit mean subtraction and standard deviation scaling",
                                       "projection_normalization": "Original direction-specific row/column sums",
                                       "rebalance_after_pruning": False})
        with stage_progress("verify"):
            verify_recall(out)
            reference_path = repository / "runs/mapping-mechanism-2026-10-04/sinkhorn-pruning.json"
            if reference_path.exists() and .05 in cfg["epsilon_grid"]:
                references = json.loads(reference_path.read_text())
                changes = []
                for reference in references["rows"]:
                    if reference["binary"]:
                        continue
                    k = reference["k"]
                    for space, metrics in reference["retrieval"].items():
                        path = out / "results" / f'sinkhorn_e0.05_k{"full" if k is None else k}__{space}.json'
                        current = json.loads(path.read_text())
                        for direction, metric in metrics.items():
                            changes.append({"k": k, "space": space, "direction": direction,
                                            "recall_difference": {
                                                rank: current["retrieval"][direction]["recall"][rank] - value
                                                for rank, value in metric["recall"].items()}})
                save_json(out / "reference_differences.json", changes)
    print(f"Completed {out}", flush=True)


if __name__ == "__main__":
    main()
