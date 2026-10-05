"""Exact Pearson correlations under specified stratum masses, using sparse observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class SparseColumn:
    rows: np.ndarray
    values: np.ndarray


def sparse_columns(matrix) -> list[SparseColumn]:
    matrix = sparse.csc_matrix(matrix, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return [
        SparseColumn(matrix.indices[left:right], matrix.data[left:right])
        for left, right in zip(matrix.indptr[:-1], matrix.indptr[1:])
    ]


@dataclass
class GroupedMoments:
    counts: np.ndarray
    sums: np.ndarray
    products: np.ndarray

    def correlations(self, masses: np.ndarray) -> np.ndarray:
        """Each row in stratum g has weight masses[g] / counts[g]; NaN means undefined."""
        masses = np.asarray(masses, dtype=np.float64)
        if masses.shape != self.counts.shape or np.any(masses < 0) or not np.isclose(masses.sum(), 1):
            raise ValueError("Stratum masses must be nonnegative, sum to one, and match counts")
        if np.any((masses > 0) & (self.counts == 0)):
            raise ValueError("Cannot assign positive mass to an empty stratum")
        weights = np.divide(masses, self.counts, out=np.zeros_like(masses), where=self.counts > 0)
        mean = weights @ self.sums
        second = np.einsum("g,gij->ij", weights, self.products)
        covariance = second - np.outer(mean, mean)
        variance = np.diag(covariance)
        tolerance = 32 * np.finfo(float).eps * np.maximum(np.diag(second), mean**2)
        valid = variance > tolerance
        denominator = np.sqrt(np.maximum(variance[:, None] * variance[None, :], 0))
        result = np.full_like(covariance, np.nan)
        np.divide(covariance, denominator, out=result, where=valid[:, None] & valid[None, :])
        return np.clip(result, -1, 1)


def grouped_moments(
    columns: list[SparseColumn], groups: np.ndarray, counts: np.ndarray
) -> GroupedMoments:
    """Accumulate moments without dropping zero activations. Negative group IDs are excluded.

    Counts include ALL retained observations, including rows where every feature is zero.
    SparseColumn coordinates must be unique and sorted, as returned by sparse_columns().
    """
    n_groups, n_features = len(counts), len(columns)
    sums = np.zeros((n_groups, n_features))
    products = np.zeros((n_groups, n_features, n_features))

    def accumulate(rows, values):
        selected = groups[rows]
        keep = selected >= 0
        return np.bincount(selected[keep], weights=values[keep], minlength=n_groups)

    for i, column in enumerate(columns):
        sums[:, i] = accumulate(column.rows, column.values)
        products[:, i, i] = accumulate(column.rows, column.values**2)
        for j in range(i):
            common, left, right = np.intersect1d(
                column.rows, columns[j].rows, assume_unique=True, return_indices=True
            )
            value = accumulate(common, column.values[left] * columns[j].values[right])
            products[:, i, j] = products[:, j, i] = value
    return GroupedMoments(np.asarray(counts), sums, products)


def balanced_binary_masses(rho: float) -> np.ndarray:
    """Group order 00, 01, 10, 11; both marginal prevalences are 0.5 and Pearson rho is exact."""
    if not np.isfinite(rho) or not -1 <= rho <= 1:
        raise ValueError("Binary correlation must lie in [-1, 1]")
    return np.array([1 + rho, 1 - rho, 1 - rho, 1 + rho]) / 4


def independent_binary_masses(counts: np.ndarray) -> np.ndarray:
    """Remove binary dependence, retaining the empirical marginals; order 00, 01, 10, 11."""
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape != (4,) or not np.all(np.isfinite(counts)) or np.any(counts < 0) or counts.sum() <= 0:
        raise ValueError("Expected four finite nonnegative stratum counts with positive total")
    masses = counts / counts.sum()
    a, b = masses[2] + masses[3], masses[1] + masses[3]
    return np.array([(1 - a) * (1 - b), (1 - a) * b, a * (1 - b), a * b])


def binary_correlation(masses: np.ndarray) -> float:
    """Pearson correlation of two binary labels in strata 00, 01, 10, 11."""
    masses = np.asarray(masses, dtype=np.float64)
    if masses.shape != (4,) or np.any(masses < 0) or not np.isclose(masses.sum(), 1):
        raise ValueError("Expected four nonnegative stratum masses summing to one")
    a, b = masses[2] + masses[3], masses[1] + masses[3]
    variance_product = a * (1 - a) * b * (1 - b)
    if variance_product <= 0:
        return float("nan")
    return float((masses[3] - a * b) / np.sqrt(variance_product))
