"""Standardized ridge paths from sparse moments, without dense sample matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg, sparse

_gemm = linalg.get_blas_funcs(("gemm",), dtype=np.float64)[0]


@dataclass
class Moments:
    n: int
    mean: np.ndarray
    second: np.ndarray

    @property
    def variance(self) -> np.ndarray:
        return np.maximum(np.diag(self.second) - self.mean**2, 0)

    @property
    def scale(self) -> np.ndarray:
        return np.where(self.variance > 0, np.sqrt(self.variance), 1.0)

    def standardized(self, reference: Moments) -> np.ndarray:
        """Second moments around FIT means, including evaluation mean shifts."""
        center = reference.mean
        second = (self.second - np.outer(self.mean, center)
                  - np.outer(center, self.mean) + np.outer(center, center))
        return second / np.outer(reference.scale, reference.scale)


def sparse_moments(x, y) -> Moments:
    joint = sparse.hstack([x, y], format="csr", dtype=np.float64)
    shape = joint.shape
    if shape is None or shape[0] < 2:
        raise ValueError("At least two aligned observations are required")
    joint.sum_duplicates()
    return Moments(shape[0], np.asarray(joint.mean(axis=0)).ravel(),
                   (joint.T @ joint).toarray() / shape[0])


def varying_columns(matrix) -> np.ndarray:
    matrix = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    second = np.asarray(matrix.power(2).mean(axis=0)).ravel()
    return np.flatnonzero(second - mean**2 > 32 * np.finfo(float).eps * np.maximum(second, mean**2))


@dataclass
class RidgePath:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    projected_cross: np.ndarray

    @classmethod
    def from_moments(cls, xx: np.ndarray, xy: np.ndarray) -> RidgePath:
        values, vectors = linalg.eigh(xx, check_finite=True)
        if values.min() < -1e-8:
            raise ValueError("Predictor second moment is not positive semidefinite")
        projected = _gemm(alpha=1, a=vectors, b=xy, trans_a=True)
        return cls(np.maximum(values, 0), vectors, projected)

    def coefficients(self, penalty: float) -> np.ndarray:
        if not np.isfinite(penalty) or penalty < 0:
            raise ValueError("Ridge penalty must be finite and nonnegative")
        denominator = self.eigenvalues + penalty
        if np.any(denominator <= 0):
            raise ValueError("Unregularized fit requires a nonsingular predictor matrix")
        result = _gemm(alpha=1, a=self.eigenvectors, b=self.projected_cross / denominator[:, None])
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Nonfinite ridge coefficients")
        return result


def prediction_mse(coefficients: np.ndarray, xx: np.ndarray, xy: np.ndarray,
                   yy: np.ndarray) -> np.ndarray:
    """Per-target MSE; moments must use the same fit-time standardization."""
    product = _gemm(alpha=1, a=xx, b=coefficients)
    result = (np.diag(yy) - 2 * np.sum(coefficients * xy, axis=0)
              + np.sum(coefficients * product, axis=0))
    return np.maximum(result, 0)


def score_matrices(ii: np.ndarray, it: np.ndarray, tt: np.ndarray,
                   image_penalty: float, text_penalty: float) -> dict[str, np.ndarray]:
    forward = RidgePath.from_moments(ii, it).coefficients(image_penalty)
    backward = RidgePath.from_moments(tt, it.T).coefficients(text_penalty).T
    return {"pearson": it, "ridge_image_to_text": forward,
            "ridge_text_to_image": backward, "ridge_bidirectional_mean": (forward + backward) / 2}
