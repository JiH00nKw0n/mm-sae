"""Exact sparse row replacement without perturbing any unedited activation."""

import numpy as np
from scipy import sparse
from typing import cast


def take_rows(matrix: sparse.csr_matrix, rows) -> sparse.csr_matrix:
    """A row-index vector preserves two dimensions; SciPy's index annotation also allows scalars."""
    indices = np.asarray(rows)
    if indices.ndim != 1 or (indices.size and indices.dtype.kind not in "iu"):
        raise ValueError("Row indices must be a vector of integers, not a boolean mask or floats")
    indices = indices.astype(np.int64, copy=False)
    return cast(sparse.csr_matrix, matrix[indices])


def activation_changes(original, changed):
    """Count changes in positive TopK support and retain continuous change magnitude."""
    common = original.multiply(changed.sign()).getnnz(axis=1)
    return {
        "turned_off": original.getnnz(axis=1) - common,
        "turned_on": changed.getnnz(axis=1) - common,
        "sum_abs_activation_change": np.asarray(abs(changed - original).sum(axis=1)).ravel(),
    }


def replace_rows(original: sparse.csr_matrix, row_ids, replacements: sparse.csr_matrix) -> sparse.csr_matrix:
    if original.shape is None or replacements.shape is None:
        raise ValueError("Sparse matrices require explicit shapes")
    row_ids = np.asarray(row_ids, dtype=np.int64)
    if len(np.unique(row_ids)) != len(row_ids) or replacements.shape != (len(row_ids), original.shape[1]):
        raise ValueError("Replacement rows must be unique and have matching dimensions")
    mapping = np.arange(original.shape[0])
    mapping[row_ids] = original.shape[0] + np.arange(len(row_ids))
    return take_rows(sparse.csr_matrix(sparse.vstack([original, replacements], format="csr")), mapping)
