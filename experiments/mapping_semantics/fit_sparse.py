"""Fit sparse ridge CCA from saved training moments without reading test moments."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Set these before importing numerical libraries when this file is run directly.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")

import numpy as np  # noqa: E402
from scipy import linalg  # noqa: E402

from mm_sae.analysis.sparse_cca import fit_sparse_cca  # noqa: E402
from mm_sae.io import sha256  # noqa: E402
from mm_sae.metrics.regression import Moments  # noqa: E402


_gemm = linalg.get_blas_funcs(("gemm",), dtype=np.float64)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moments", type=Path, default=Path("runs/mapping-suite-2026-10-04/moments.npz"))
    parser.add_argument("--initial", type=Path,
                        default=Path("runs/mapping-ablation-2026-10-04/transforms/cca_256.npz"))
    parser.add_argument("--output", type=Path, default=Path("runs/mapping-semantics-2026-10-04/sparse-fit"))
    parser.add_argument("--supports", nargs="+", type=int, default=[8, 16, 4])
    parser.add_argument("--dimensions", type=int, default=256)
    parser.add_argument("--ridge", type=float, default=.01)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--swap-steps", type=int, default=2)
    parser.add_argument("--candidate-pool", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.moments) as saved:
        image_ids, text_ids = saved["image_ids"], saved["text_ids"]
        fit = Moments(int(saved["fit_n"]), saved["fit_mean"], saved["fit_second"])
    covariance = fit.standardized(fit)
    ni = len(image_ids)
    xx, cross, yy = covariance[:ni, :ni], covariance[:ni, ni:], covariance[ni:, ni:]
    with np.load(args.initial) as saved:
        initial_image = saved["image"][:, :args.dimensions]
        initial_text = saved["text"][:, :args.dimensions]
    provenance = {"moments_path": str(args.moments.resolve()), "moments_sha256": sha256(args.moments),
                  "initial_path": str(args.initial.resolve()), "initial_sha256": sha256(args.initial),
                  "fit_n": fit.n, "accessed_moment_fields": ["image_ids", "text_ids", "fit_n", "fit_mean",
                                                           "fit_second"],
                  "solver_source_sha256": sha256(Path(__file__).resolve().parents[2] /
                                                  "src/mm_sae/analysis/sparse_cca.py")}
    for k in args.supports:
        destination = args.output / f"sparse_cca_k{k}.npz"
        metadata_path = args.output / f"sparse_cca_k{k}.json"
        if destination.exists() or metadata_path.exists():
            raise FileExistsError(f"Refusing to overwrite completed sparse fit {destination}")
        started = time.monotonic()

        def progress(record):
            if record["component"] % 16 == 0:
                print(json.dumps({"k": k, "component": record["component"],
                                  "elapsed_seconds": time.monotonic() - started,
                                  "iterations": record["iterations"], "converged": record["converged"],
                                  "objective": record["residual_objective"]}), flush=True)

        model = fit_sparse_cca(xx, cross, yy, args.dimensions, k=k, ridge=args.ridge,
                               initial_image=initial_image, initial_text=initial_text,
                               max_iter=args.max_iter, swap_steps=args.swap_steps,
                               candidate_pool=args.candidate_pool, progress=progress)
        model.metadata.update(provenance)
        model.metadata["fit_seconds"] = time.monotonic() - started
        for side, coefficients, covariance_side in (("image", model.image, xx), ("text", model.text, yy)):
            metric = covariance_side + args.ridge * np.eye(len(covariance_side))
            gram = _gemm(1., coefficients.T, _gemm(1., metric, coefficients))
            offdiag = gram - np.diag(np.diag(gram))
            model.metadata[side + "_max_absolute_component_covariance"] = float(np.max(np.abs(offdiag)))
            model.metadata[side + "_coefficient_rank"] = int(np.linalg.matrix_rank(coefficients))
        np.savez_compressed(destination, image=model.image, text=model.text,
                            raw_text=model.text * np.asarray(
                                model.metadata["output_text_orientation_factors"])[None, :],
                            residual_objectives=model.singular_values, image_ids=image_ids, text_ids=text_ids)
        metadata_path.write_text(json.dumps(model.metadata, indent=2) + "\n")
        print(json.dumps({"completed": str(destination), "seconds": model.metadata["fit_seconds"],
                          "converged_components": model.metadata["converged_components"]}), flush=True)


if __name__ == "__main__":
    main()
