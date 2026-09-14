"""Exact sparse row replacement without perturbing any unedited activation."""

import numpy as np
from scipy import sparse


def activation_changes(original, changed):
    """Count changes in positive TopK support and retain continuous change magnitude."""
    common = original.multiply(changed.sign()).getnnz(axis=1)
    return {
        "turned_off": original.getnnz(axis=1) - common,
        "turned_on": changed.getnnz(axis=1) - common,
        "sum_abs_activation_change": np.asarray(abs(changed - original).sum(axis=1)).ravel(),
    }


def replace_rows(original, row_ids, replacements):
    row_ids = np.asarray(row_ids, dtype=np.int64)
    if len(np.unique(row_ids)) != len(row_ids) or replacements.shape != (len(row_ids), original.shape[1]):
        raise ValueError("Replacement rows must be unique and have matching dimensions")
    mapping = np.arange(original.shape[0])
    mapping[row_ids] = original.shape[0] + np.arange(len(row_ids))
    return sparse.vstack([original, replacements], format="csr")[mapping]
