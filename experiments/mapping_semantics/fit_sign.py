"""Fit signed and nonnegative sparse regressions using frozen support masks.

This stage reads only fit moments. It does not evaluate retrieval or select
regularization, support, or a direction using validation or test observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from mm_sae.analysis.signed_mapping import fixed_support_sign_fits, fit_objective
from mm_sae.io import atomic_json, sha256
from mm_sae.metrics.regression import Moments

PENALTY = 0.01
TOLERANCE = 1e-7


def support_digest(support: np.ndarray) -> str:
    """Hash a support mask independently of the stored coefficient signs."""
    digest = hashlib.sha256(np.asarray(support.shape, dtype="<i8").tobytes())
    digest.update(np.packbits(support, bitorder="little").tobytes())
    return digest.hexdigest()


def fit_sign_experiment(
    moments_path: Path, support_paths: dict[int, Path], output: Path,
    *, penalty: float = PENALTY, tolerance: float = TOLERANCE,
) -> dict:
    """Save independently fitted directional coefficients and audit metadata."""
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("penalty must be finite and positive")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if not support_paths:
        raise ValueError("At least one fixed support is required")
    moments_path, output = moments_path.resolve(), output.resolve()
    support_paths = {k: path.resolve() for k, path in support_paths.items()}
    root = Path(__file__).resolve().parents[2]
    code_paths = [Path(__file__).resolve(), root / "src/mm_sae/analysis/signed_mapping.py",
                  root / "src/mm_sae/analysis/mapping_evaluation.py", root / "src/mm_sae/metrics/regression.py"]
    source_hashes = {str(path): sha256(path) for path in [moments_path, *support_paths.values()]}
    code_hashes = {str(path): sha256(path) for path in code_paths}
    signature = {"source_hashes": source_hashes, "code_hashes": code_hashes,
                 "penalty": penalty, "tolerance": tolerance,
                 "support_budgets": sorted(support_paths)}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["signature"] != signature:
        raise ValueError("Fit inputs or implementation changed. Choose a new output directory.")
    with np.load(moments_path, allow_pickle=False) as saved:
        image_ids, text_ids = saved["image_ids"], saved["text_ids"]
        fit = Moments(int(saved["fit_n"]), saved["fit_mean"], saved["fit_second"])
    ni, nt = len(image_ids), len(text_ids)
    if len(fit.mean) != ni + nt:
        raise ValueError("Feature IDs and fit moments have inconsistent dimensions")
    covariance = fit.standardized(fit)
    ii, it, tt = covariance[:ni, :ni], covariance[:ni, ni:], covariance[ni:, ni:]
    manifest = {"signature": signature, "experiment": "fixed_support_coefficient_sign",
                "model_class": "sparse linear regression", "penalty": penalty, "tolerance": tolerance,
                "penalty_selection": "fixed before fitting; no validation or test selection",
                "fit_caption_pairs": fit.n, "image_features": ni, "text_features": nt,
                "moments_arrays_read": ["image_ids", "text_ids", "fit_n", "fit_mean", "fit_second"],
                "preprocessing": "subtract fit mean and divide by fit standard deviation",
                "zero_variance_scale": 1.0, "intercept_in_standardized_coordinates": 0.0,
                "coefficient_orientation": "predictor features by target features",
                "directions_fitted_independently": True,
                "cpu_thread_environment": {key: os.environ.get(key) for key in
                                           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                            "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
                "conditions": []}
    for budget, support_path in sorted(support_paths.items()):
        if budget < 1:
            raise ValueError("Support budgets must be positive")
        with np.load(support_path, allow_pickle=False) as saved:
            direct = saved["image_by_text"]
        if direct.shape != (ni, nt) or not np.isfinite(direct).all():
            raise ValueError("Source transform must have finite image-by-text feature dimensions")
        support = direct != 0
        if np.any(support.sum(axis=0) > budget) or np.any(support.sum(axis=1) > budget):
            raise ValueError("Source support exceeds the declared per-feature incident-edge budget")
        fits = fixed_support_sign_fits(fit, ni, support, penalty=penalty, tolerance=tolerance)
        for constraint, directions in fits.items():
            key = f"procrustes_support_{budget}_{constraint}"
            path = output / (key + ".npz")
            record = {"key": key, "model_class": "sparse linear regression",
                      "label": f"{constraint} sparse linear regression on fixed support of degree at most {budget}",
                      "sign_constraint": constraint, "support_budget": budget,
                      "support_source": str(support_path), "support_source_sha256": sha256(support_path),
                      "allowed_support_sha256": support_digest(support),
                      "allowed_support_count": int(support.sum()), "penalty": penalty, "tolerance": tolerance,
                      "source_hashes": source_hashes, "code_hashes": code_hashes,
                      "coefficient_orientation": "predictor features by target features",
                      "preprocessing": manifest["preprocessing"], "directions": {}}
            for direction, result in directions.items():
                xx, xy, yy, allowed = ((ii, it, tt, support) if direction == "image_to_text"
                                      else (tt, it.T, ii, support.T))
                coefficients = result.coefficients
                if np.any(coefficients[~allowed] != 0):
                    raise RuntimeError("A fitted coefficient lies outside the fixed support")
                if constraint == "nonnegative" and np.any(coefficients < 0):
                    raise RuntimeError("A nonnegative fit returned a negative coefficient")
                if not result.converged:
                    raise RuntimeError(f"Fit failed its optimality check for {key}/{direction}")
                record["directions"][direction] = {
                    "shape": list(coefficients.shape), "allowed_support_count": int(allowed.sum()),
                    "actual_nonzero_count": int(np.count_nonzero(coefficients)),
                    "negative_count": int(np.count_nonzero(coefficients < 0)),
                    "zeroed_allowed_count": int(np.count_nonzero(allowed & (coefficients == 0))),
                    "converged": bool(result.converged), "iterations": result.iterations,
                    "optimality_residual": result.kkt_residual,
                    "optimality_residual_definition": ("maximum absolute gradient on allowed support"
                                                       if constraint == "signed" else
                                                       "maximum projected gradient on allowed support"),
                    "solver": "local Cholesky normal equations" if constraint == "signed" else
                              "local nonnegative least squares on Cholesky factors",
                    **fit_objective(coefficients, xx, xy, yy, penalty),
                }
            np.savez_compressed(path, **{direction: fit.coefficients for direction, fit in directions.items()},
                                allowed_support_image_by_text=support, image_ids=image_ids, text_ids=text_ids,
                                fit_mean=fit.mean, fit_scale=fit.scale)
            record["coefficients_path"] = str(path)
            record["coefficients_sha256"] = sha256(path)
            atomic_json(output / (key + ".json"), record)
            manifest["conditions"].append(record)
            print(json.dumps({"completed": key, "allowed_support_count": int(support.sum()),
                              "fits_converged": True}), flush=True)
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moments", type=Path, default=Path("runs/mapping-suite-2026-10-04/moments.npz"))
    parser.add_argument("--support-dir", type=Path,
                        default=Path("runs/mapping-pruning-2026-10-04/transforms"))
    parser.add_argument("--support-grid", "--supports-grid", type=int, nargs="+", default=[8, 16],
                        help="Fixed per-feature support budgets to load from --support-dir")
    parser.add_argument("--penalty", type=float, default=PENALTY,
                        help="Fixed positive ridge penalty, identical across sign constraints")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE,
                        help="Positive optimality tolerance, identical across sign constraints")
    parser.add_argument("--output", type=Path, default=Path("runs/mapping-semantics-2026-10-04/sign-fit"))
    args = parser.parse_args()
    fit_sign_experiment(args.moments, {k: args.support_dir / f"procrustes_{k}.npz" for k in args.support_grid},
                        args.output, penalty=args.penalty, tolerance=args.tolerance)


if __name__ == "__main__":
    main()
