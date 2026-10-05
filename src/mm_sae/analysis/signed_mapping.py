"""Compare coefficient signs in sparse linear regression on a fixed support.

Every regression matrix has predictors in rows and targets in columns. The
support supplied by an alignment method fixes allowed coefficients only. A
refitted matrix is a sparse linear regression, not an orthogonal alignment.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg

from mm_sae.analysis.mapping_evaluation import RidgeFit, fit_nonnegative_ridge
from mm_sae.metrics.regression import Moments


def fit_signed_ridge(xx, xy, support, penalty: float, *, tolerance: float = 1e-7) -> RidgeFit:
    """Fit unrestricted signs on allowed entries using local normal equations.

    The objective per target is half the average squared prediction error plus
    half ``penalty`` times the squared coefficient norm. A positive penalty
    makes every supported subproblem uniquely solvable for valid moments.
    Only the gradient on allowed entries participates in the residual check.
    """
    xx, xy = np.asarray(xx, dtype=np.float64), np.asarray(xy, dtype=np.float64)
    support = np.asarray(support, dtype=bool)
    if xx.ndim != 2 or xy.ndim != 2 or not np.isfinite(xx).all() or not np.isfinite(xy).all():
        raise ValueError("xx and xy must be finite two-dimensional matrices")
    if xx.shape != (xy.shape[0], xy.shape[0]) or support.shape != xy.shape:
        raise ValueError("xx, xy, and support dimensions do not agree")
    if not np.allclose(xx, xx.T, rtol=1e-9, atol=1e-10):
        raise ValueError("xx must be symmetric")
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("penalty must be finite and positive")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    gram = (xx + xx.T) / 2 + penalty * np.eye(len(xx))
    coefficients = np.zeros_like(xy)
    for target in np.flatnonzero(support.any(axis=0)):
        selected = np.flatnonzero(support[:, target])
        local = gram[np.ix_(selected, selected)]
        try:
            lower = linalg.cholesky(local, lower=True)
        except linalg.LinAlgError as exc:
            raise ValueError("Supported regularized predictor moments must be positive definite") from exc
        coefficients[selected, target] = linalg.cho_solve((lower, True), xy[selected, target])
    residual = float(np.max(np.abs((gram @ coefficients - xy)[support]), initial=0))
    limit = tolerance * max(1.0, float(np.max(np.abs(xy[support]), initial=0)))
    return RidgeFit(coefficients, residual <= limit, 1 if support.any() else 0, residual)


def fixed_support_sign_fits(
    fit: Moments, image_count: int, image_by_text_support: np.ndarray,
    *, penalty: float = 0.01, tolerance: float = 1e-7,
) -> dict[str, dict[str, RidgeFit]]:
    """Fit both sign constraints and both directions from the same fit moments.

    The two constraints receive exactly the same allowed support and penalty.
    Centering and standard deviations come only from ``fit``. Image-to-text
    and text-to-image coefficients are independently fitted, not transposes.
    No validation or test observations are accepted by this function.
    """
    size = len(fit.mean)
    support = np.asarray(image_by_text_support, dtype=bool)
    if fit.n < 2 or not 0 < image_count < size or fit.second.shape != (size, size):
        raise ValueError("Fit moments and image_count must specify both modalities")
    if support.shape != (image_count, size - image_count):
        raise ValueError("Support must have image features in rows and text features in columns")
    covariance = fit.standardized(fit)
    ii, it, tt = (covariance[:image_count, :image_count], covariance[:image_count, image_count:],
                  covariance[image_count:, image_count:])
    result: dict[str, dict[str, RidgeFit]] = {"signed": {}, "nonnegative": {}}
    for direction, xx, xy, allowed in (
        ("image_to_text", ii, it, support),
        ("text_to_image", tt, it.T, support.T),
    ):
        result["signed"][direction] = fit_signed_ridge(xx, xy, allowed, penalty, tolerance=tolerance)
        result["nonnegative"][direction] = fit_nonnegative_ridge(
            xx, xy, allowed, penalty, tolerance=tolerance, device="cpu",
            sparse_support_limit=int(np.max(allowed.sum(axis=0), initial=0)),
        )
    return result


def fit_objective(coefficients: np.ndarray, xx: np.ndarray, xy: np.ndarray,
                  yy: np.ndarray, penalty: float) -> dict[str, float]:
    """Report the exact fit-moment regression objective summed over targets."""
    squared_errors = (np.diag(yy) - 2 * np.sum(coefficients * xy, axis=0)
                      + np.sum(coefficients * (xx @ coefficients), axis=0))
    squared_norm = float(np.sum(coefficients**2))
    total = float(0.5 * (squared_errors.sum() + penalty * squared_norm))
    return {"objective_sum_over_targets": total,
            "objective_mean_over_targets": total / coefficients.shape[1],
            "fit_mean_squared_error": float(squared_errors.mean()),
            "coefficient_squared_norm": squared_norm}
