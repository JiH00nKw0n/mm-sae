"""Fit-only transforms for controlled SAE correspondence retrieval comparisons.

Every projection accepts an already selected set of SAE coordinates. Fit moments
must describe centered, standardized training activations; held-out moments never
enter these functions. Image and text coefficients have the same column count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import linalg, sparse


@dataclass
class Projection:
    image: np.ndarray
    text: np.ndarray
    singular_values: np.ndarray
    metadata: dict[str, Any]


def _matrix(value, name: str) -> np.ndarray:
    result = np.asarray(value.toarray() if sparse.issparse(value) else value, dtype=np.float64)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite two-dimensional matrix")
    return result


def preprocess(
    values, fit_mean, fit_scale, mode: Literal["raw", "centered", "standardized"] = "raw",
) -> np.ndarray:
    """Apply fixed training statistics, without estimating any evaluation statistic.

    Raw values pass through unchanged, centered values subtract the fit mean, and
    standardized values additionally divide by the fit standard deviation. Input
    arrays are never modified. Zero fit variance is rejected for standardization;
    the experiment must exclude those coordinates using training data only.
    """
    array = _matrix(values, "values")
    mean, scale = np.asarray(fit_mean, dtype=float), np.asarray(fit_scale, dtype=float)
    if mean.shape != (array.shape[1],) or scale.shape != mean.shape:
        raise ValueError("Fit mean and scale must have one entry per feature")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
        raise ValueError("Fit statistics must be finite")
    if mode == "raw":
        return array.copy()
    if mode == "centered":
        return array - mean
    if mode == "standardized":
        if np.any(scale <= 0):
            raise ValueError("Standardization requires positive fit standard deviations")
        return (array - mean) / scale
    raise ValueError(f"Unknown preprocessing mode: {mode}")


def _within_moments(value, width: int, name: str) -> np.ndarray:
    result = _matrix(value, name)
    if result.shape != (width, width):
        raise ValueError(f"{name} must be square and match its feature dimension")
    if not np.allclose(result, result.T, atol=1e-10, rtol=1e-9):
        raise ValueError(f"{name} must be symmetric")
    return (result + result.T) / 2


def fit_common_projection(
    xx, cross, yy, dimensions: int | None = None, *, whiten: bool = False, ridge: float = 0.01,
) -> Projection:
    """Fit matched-rank cross-SVD or regularized CCA from the same fit moments.

    Without whitening, use the left/right singular vectors of the cross moment.
    With whitening, first Cholesky-whiten both within-modality moment matrices,
    adding ``ridge * identity`` exactly as the existing CCA reference does. The
    returned CCA coefficients undo the Cholesky factors. Singular values are not
    multiplied into either representation in either condition.

    ``dimensions=None`` retains min(image width, text width) common directions.
    A larger integer is clipped to that same width, matching the existing CCA
    convention. Unlike full square Procrustes, cross-SVD drops the orthogonal
    complement of the larger modality even at its full common rank.
    """
    cross = _matrix(cross, "cross")
    ni, nt = cross.shape
    if min(ni, nt) < 1:
        raise ValueError("Projection requires at least one feature on each side")
    xx = _within_moments(xx, ni, "xx")
    yy = _within_moments(yy, nt, "yy")
    if dimensions is not None and (isinstance(dimensions, bool)
                                  or not isinstance(dimensions, (int, np.integer)) or dimensions < 1):
        raise ValueError("dimensions must be a positive integer or None")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    dim = min(ni, nt) if dimensions is None else min(int(dimensions), ni, nt)
    whitened = cross
    if whiten:
        left = linalg.cholesky(xx + ridge * np.eye(ni), lower=True)
        right = linalg.cholesky(yy + ridge * np.eye(nt), lower=True)
        whitened = linalg.solve_triangular(left, cross, lower=True)
        whitened = linalg.solve_triangular(right, whitened.T, lower=True).T
    u, values, vt = linalg.svd(whitened, full_matrices=False)
    image, text = u[:, :dim], vt.T[:, :dim]
    if whiten:
        image = linalg.solve_triangular(left.T, image, lower=False)
        text = linalg.solve_triangular(right.T, text, lower=False)
    return Projection(image, text, values[:dim], {
        "method": "ridge_cca" if whiten else "cross_svd",
        "dimensions_requested": dimensions,
        "dimensions": dim,
        "whiten": whiten,
        "ridge": float(ridge) if whiten else 0.0,
        "singular_value_weighting": False,
        "fit_moments_only": True,
    })


def group_projection(u, v, xx, yy, *, normalize_variance: bool = False) -> Projection:
    """Use fixed sparse factors as paired common group coordinates.

    Each factor column is divided by its L1 sum, making the group an activation
    weighted average. This removes the arbitrary ``u[:, k] *= a; v[:, k] /= a``
    scale ambiguity. Optionally divide each side by its group standard deviation
    estimated from that side's fit moments. Evaluation values are not centered or
    scaled again here; call ``preprocess`` before applying these coefficients.

    A group absent or with zero fit variance on either side is removed from both
    sides. Metadata records retained and removed original group indices. Returned
    coefficients may have zero columns; the caller must skip retrieval explicitly
    if no group survives. Covariance matrices must be positive semidefinite along
    every candidate group direction.
    """
    u, v = _matrix(u, "u"), _matrix(v, "v")
    if u.shape[1] != v.shape[1]:
        raise ValueError("Image and text factors must contain the same number of groups")
    if np.any(u < 0) or np.any(v < 0):
        raise ValueError("Group factors must be nonnegative")
    xx = _within_moments(xx, u.shape[0], "xx")
    yy = _within_moments(yy, v.shape[0], "yy")
    sum_i, sum_t = u.sum(axis=0), v.sum(axis=0)
    image = np.divide(u, sum_i, out=np.zeros_like(u), where=sum_i > 0)
    text = np.divide(v, sum_t, out=np.zeros_like(v), where=sum_t > 0)
    # scipy BLAS avoids platform-specific numpy matmul flag warnings on macOS.
    gemm = linalg.blas.get_blas_funcs(["gemm"], dtype=np.float64)[0]
    variance_i = np.sum(image * gemm(1.0, xx, image), axis=0)
    variance_t = np.sum(text * gemm(1.0, yy, text), axis=0)
    tolerance = np.finfo(float).eps * 64
    if np.any(variance_i < -tolerance) or np.any(variance_t < -tolerance):
        raise ValueError("Fit moments imply a negative group variance")
    missing = (sum_i == 0) | (sum_t == 0)
    constant = (variance_i <= tolerance) | (variance_t <= tolerance)
    keep = ~(missing | constant)
    image, text = image[:, keep], text[:, keep]
    if normalize_variance:
        image = image / np.sqrt(variance_i[keep])
        text = text / np.sqrt(variance_t[keep])
    return Projection(image, text, np.empty(0), {
        "method": "group_weighted_average",
        "normalize_fit_variance": normalize_variance,
        "groups_requested": u.shape[1],
        "dimensions": int(keep.sum()),
        "retained_group_indices": np.flatnonzero(keep).tolist(),
        "removed_missing_side_indices": np.flatnonzero(missing).tolist(),
        "removed_zero_fit_variance_indices": np.flatnonzero(constant & ~missing).tolist(),
        "fit_variance_image": variance_i[keep].tolist(),
        "fit_variance_text": variance_t[keep].tolist(),
        "fit_moments_only": True,
    })
