"""Delete frozen signed coefficients without refitting or changing their scale."""

from typing import Literal

import numpy as np
from scipy import linalg


def prune_coefficients(values, k: int | None, *, rule: Literal["column", "mutual"]):
    """Keep absolute top-k entries per column, or at both ends of every edge.

    Ties favor smaller indices. Original signs and surviving coefficients are
    preserved. Column pruning limits inputs to each common component; mutual
    pruning limits both degrees of a direct cross-modal coefficient matrix.
    """
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError("Expected a nonempty finite matrix")
    if rule not in {"column", "mutual"}:
        raise ValueError("Unknown support rule")
    if k is None:
        return matrix.copy()
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer or None")
    order = np.argsort(-np.abs(matrix), axis=0, kind="stable")
    support = np.zeros_like(matrix, dtype=bool)
    support[order[:k], np.arange(matrix.shape[1])[None, :]] = True
    if rule == "mutual":
        order = np.argsort(-np.abs(matrix), axis=1, kind="stable")
        rows = np.zeros_like(matrix, dtype=bool)
        rows[np.arange(matrix.shape[0])[:, None], order[:, :k]] = True
        support &= rows
    return np.where(support, matrix, 0)


def procrustes_matrix(cross):
    """Return image-by-text coefficients without singular-value weighting."""
    u, _, vt = linalg.svd(np.asarray(cross, dtype=float), full_matrices=False)
    gemm = linalg.get_blas_funcs(("gemm",), dtype=np.float64)[0]
    return gemm(1., u, vt)


def coefficient_summary(values):
    matrix = np.asarray(values)
    nonzero = matrix != 0
    return {
        "nonzero_coefficients": int(nonzero.sum()),
        "total_coefficients": int(matrix.size),
        "negative_coefficients": int((matrix < 0).sum()),
        "max_per_row": int(nonzero.sum(axis=1).max()),
        "max_per_column": int(nonzero.sum(axis=0).max()),
        "covered_rows": int(nonzero.any(axis=1).sum()),
        "covered_columns": int(nonzero.any(axis=0).sum()),
    }
