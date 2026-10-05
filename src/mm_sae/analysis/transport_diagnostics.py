"""Distinguish exact support from concentration of nonnegative transport weights."""

import numpy as np


def transport_summary(weights, original=None):
    matrix = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Expected nonempty finite nonnegative weights")
    reference = matrix if original is None else np.asarray(original, dtype=np.float64)
    if (reference.shape != matrix.shape or not np.isfinite(reference).all()
            or (reference < 0).any() or reference.sum() <= 0):
        raise ValueError("Original transport must have positive finite mass and the same shape")
    support = matrix > 0
    result = {
        "edge_count": int(support.sum()), "density": float(support.mean()),
        "numeric_zero_count": int((~support).sum()),
        "retained_mass": float(matrix.sum() / reference.sum()),
    }
    for side, rows in (("image", matrix), ("text", matrix.T)):
        sums = rows.sum(1)
        nonempty = sums > 0
        count = (rows > 0).sum(1)
        squared = np.sum(rows**2, axis=1)
        effective = np.divide(sums**2, squared, out=np.zeros_like(sums), where=squared > 0)
        descending = np.sort(rows, axis=1)[:, ::-1]
        cumulative = np.cumsum(descending, axis=1)
        count95 = np.where(nonempty, 1 + (cumulative < .95 * sums[:, None]).sum(1), 0)
        result.update({
            side + "_coverage": float(nonempty.mean()),
            side + "_covered_count": int(nonempty.sum()),
            side + "_degree_max": int(count.max()),
            side + "_degree_mean": float(count.mean()),
            side + "_effective_degree_mean": float(effective.mean()),
            side + "_mass95_degree_mean": float(count95.mean()),
        })
    return result
