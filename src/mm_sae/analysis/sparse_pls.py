"""Sequential cardinality-constrained PLS-SVD in original activation coordinates.

This exploratory hard-threshold solver is not a reproduction of L1 sparse PLS.
It constrains each loading to k inputs and unit Euclidean norm. Components are
extracted with rank-one cross-covariance deflation, without mutual orthogonality.
"""

from collections.abc import Callable

import numpy as np
from scipy import linalg

from mm_sae.analysis.mapping_ablation import Projection


def _unit_topk(v, k):
    result = np.zeros_like(v)
    selected = np.argsort(-np.abs(v), kind="stable")[:k]
    result[selected] = v[selected]
    norm = np.linalg.norm(result)
    if norm == 0:
        result[0] = 1
    else:
        result /= norm
    return result


def fit_sparse_pls(cross, initial_image, initial_text, *, k=16, max_iter=1000,
                   tolerance=1e-7, progress: Callable[[dict], None] | None = None):
    cross = np.asarray(cross, dtype=np.float64)
    initial_image = np.asarray(initial_image, dtype=np.float64)
    initial_text = np.asarray(initial_text, dtype=np.float64)
    if cross.ndim != 2 or not np.isfinite(cross).all():
        raise ValueError("Expected finite cross covariance matrix")
    if k < 1 or max_iter < 1 or tolerance <= 0:
        raise ValueError("Positive support size, iteration limit and tolerance required")
    if (initial_image.shape[0] != cross.shape[0]
            or initial_text.shape[0] != cross.shape[1]
            or initial_image.shape[1] != initial_text.shape[1]):
        raise ValueError("Initial loadings must match cross covariance and component count")
    residual = cross.copy(order="F")
    image, text = np.zeros_like(initial_image), np.zeros_like(initial_text)
    gemv = linalg.get_blas_funcs(("gemv",), dtype=np.float64)[0]
    records = []
    for component in range(image.shape[1]):
        u = _unit_topk(initial_image[:, component], k)
        v = _unit_topk(initial_text[:, component], k)
        objective = float(u @ gemv(1., residual, v))
        if objective < 0:
            v = -v
            objective = -objective
        converged = False
        history = [objective]
        for iteration in range(1, max_iter + 1):
            old_u, old_v = u != 0, v != 0
            u = _unit_topk(gemv(1., residual, v), k)
            v = _unit_topk(gemv(1., residual, u, trans=1), k)
            updated = float(u @ gemv(1., residual, v))
            if updated < objective - 1e-9 * max(1., abs(objective)):
                raise RuntimeError("Alternating sparse PLS objective decreased")
            history.append(updated)
            converged = (np.array_equal(old_u, u != 0) and np.array_equal(old_v, v != 0)
                         and updated - objective <= tolerance * max(1., abs(objective)))
            objective = updated
            if converged:
                break
        raw_covariance = float(u @ gemv(1., cross, v))
        sign = 1 if raw_covariance >= 0 else -1
        image[:, component], text[:, component] = u, sign * v
        residual -= objective * np.outer(u, v)
        record = dict(component=component, iterations=iteration, converged=bool(converged),
                      objective=objective, objective_history=history,
                      original_covariance=abs(raw_covariance), output_text_sign=sign)
        records.append(record)
        if progress is not None:
            progress(record)
    return Projection(image, text, np.array([r["objective"] for r in records]), {
        "method": "hard_cardinality_sparse_pls_svd", "k": k,
        "dimensions": image.shape[1], "normalization": "unit_Euclidean_loading_norm",
        "deflation": "C_res -= objective * outer(u, v)",
        "sparse_components_are_orthogonal": False,
        "optimizer": "alternating_exact_topk_updates_local_solution",
        "initialization": "dense_PLS_SVD_columns",
        "output_orientation": "nonnegative_original_training_covariance",
        "max_iter": max_iter, "tolerance": tolerance,
        "converged_components": sum(r["converged"] for r in records), "components": records,
    })
