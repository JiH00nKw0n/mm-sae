"""Training-only annotation moments and fixed semantic-coordinate projections."""

from __future__ import annotations

import numpy as np
from scipy import linalg, sparse

_gemm = linalg.get_blas_funcs(("gemm",), dtype=np.float64)[0]


def annotation_moments(activations, labels, feature_mean, feature_scale, *, weights=None):
    """Cross covariance of standardized activations and binary concept labels.

    Integer image weights can reproduce the paired-caption population without
    repeating the image activation matrix. All inputs must be training rows.
    The observed input mean is returned to audit the supplied normalization.
    """
    x = sparse.csr_matrix(activations, dtype=np.float64)
    y = sparse.csr_matrix(labels, dtype=np.float64)
    xshape, yshape = x.shape, y.shape
    if xshape is None or yshape is None:
        raise ValueError("Activation and label arrays must have known shapes")
    mean, scale = np.asarray(feature_mean, float), np.asarray(feature_scale, float)
    w = np.ones(xshape[0]) if weights is None else np.asarray(weights, float)
    if xshape[0] != yshape[0] or mean.shape != (xshape[1],) or scale.shape != mean.shape:
        raise ValueError("Activation, label and normalization dimensions must agree")
    if w.shape != (xshape[0],) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Weights must be finite, nonnegative and have positive total")
    if not np.isfinite(x.data).all() or not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Activations and normalization must be finite")
    if np.any(scale <= 0) or not np.all((y.data == 0) | (y.data == 1)):
        raise ValueError("Scales must be positive and labels binary")
    total = float(w.sum())
    weighted_y = y.multiply(w[:, None]).tocsr()
    label_mean = np.asarray(weighted_y.sum(axis=0)).ravel() / total
    observed_mean = np.asarray(x.T @ w).ravel() / total
    xy = (x.T @ weighted_y).toarray() / total
    # True covariance uses the observed mean. A separate returned offset allows
    # projection around the exact training mean if a reference differs slightly.
    xy = (xy - np.outer(observed_mean, label_mean)) / scale[:, None]
    return dict(cross=xy, label_mean=label_mean, label_variance=label_mean * (1-label_mean),
                observed_mean=observed_mean, standardized_mean=(observed_mean-mean)/scale,
                total_weight=total)


def standardize_concept_weights(coefficients, covariance, *, tolerance=1e-14):
    """Give every nonconstant training prediction unit variance, retaining zeros."""
    b, cov = np.asarray(coefficients, float), np.asarray(covariance, float)
    if b.ndim != 2 or cov.shape != (b.shape[0], b.shape[0]):
        raise ValueError("Coefficient and covariance dimensions must agree")
    variance = np.einsum("ij,ij->j", b, _gemm(1.0, cov, b))
    if np.any(variance < -tolerance) or not np.isfinite(variance).all():
        raise ValueError("Invalid fitted prediction variance")
    scale = np.sqrt(np.maximum(variance, 0))
    active = scale > tolerance
    normalized = np.divide(b, scale[None, :], out=np.zeros_like(b), where=active[None, :])
    return normalized, scale, active


def project_concepts(raw_activations, feature_mean, feature_scale, coefficients):
    """Project inference activations only; this function accepts no annotations."""
    x = np.asarray(raw_activations, dtype=np.float64)
    b = np.asarray(coefficients, dtype=np.float64)
    mean, scale = np.asarray(feature_mean), np.asarray(feature_scale)
    if x.ndim != 2 or b.ndim != 2 or x.shape[1] != b.shape[0]:
        raise ValueError("Incompatible activation and coefficient dimensions")
    if mean.shape != (x.shape[1],) or scale.shape != mean.shape or np.any(scale <= 0):
        raise ValueError("Incompatible normalization")
    result = _gemm(1.0, (x-mean)/scale, b)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite concept predictions")
    return result
