"""Fit small SAE feature sets to annotated concepts using fit moments only.

Each concept is a separate squared-loss ridge regression of its centered label
on centered SAE activations. Forward selection adds the feature giving the
largest exact reduction in the regularized objective after all coefficients
in the expanded set are refitted. This is a deterministic greedy procedure,
not a globally optimal sparse classifier or a logistic regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import linalg


@dataclass(frozen=True)
class ConceptSetFit:
    """Coefficients and fit-only diagnostics for each allowed set size.

    Coefficient rows are feature indices and columns are concepts. With
    centered predictor rows ``x``, the predicted label is ``x @ coefficients
    + label_mean``. The function never standardizes labels or clips scores.
    Objective histories contain the no-feature objective followed by the
    objective after each greedy step. After an early stop the objective stays
    constant and subsequent gains are zero. Dense fits use the same penalty.
    """

    coefficients: dict[int, np.ndarray]
    dense_coefficients: np.ndarray
    selected_features: list[list[int]]
    objective_history: np.ndarray
    gain_history: np.ndarray
    label_mean: np.ndarray
    label_variance: np.ndarray
    penalty: float


def _validate_moments(xx, xy, label_mean, label_variance, penalty, min_gain):
    xx = np.asarray(xx, dtype=np.float64)
    xy = np.asarray(xy, dtype=np.float64)
    mean = np.asarray(label_mean, dtype=np.float64)
    variance = np.asarray(label_variance, dtype=np.float64)
    arrays = (xx, xy, mean, variance)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Predictor and label moments must be finite")
    if (xx.ndim != 2 or xy.ndim != 2 or mean.ndim != 1 or variance.ndim != 1
            or xx.shape != (xy.shape[0], xy.shape[0])
            or mean.shape != (xy.shape[1],) or variance.shape != mean.shape):
        raise ValueError("Predictor and label moment dimensions do not agree")
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("penalty must be finite and positive")
    if not np.isfinite(min_gain) or min_gain < 0:
        raise ValueError("min_gain must be finite and nonnegative")
    if not np.allclose(xx, xx.T, rtol=1e-9, atol=1e-10):
        raise ValueError("xx must be symmetric")
    xx = (xx + xx.T) / 2
    scale = max(1.0, float(np.max(np.abs(xx), initial=0)))
    if len(xx) and linalg.eigvalsh(xx, subset_by_index=[0, 0])[0] < -1e-9 * scale:
        raise ValueError("xx must be positive semidefinite")
    if np.any(variance < 0):
        raise ValueError("Label variance must be nonnegative")
    covariance_limit = np.maximum(np.diag(xx), 0)[:, None] * variance[None, :]
    if np.any(xy**2 > covariance_limit + 1e-9 * np.maximum(1.0, covariance_limit)):
        raise ValueError("Feature-label crosscovariance exceeds the covariance bound")
    if np.any(np.abs(xy[:, variance == 0]) > 1e-12):
        raise ValueError("A constant label must have zero feature-label crosscovariance")
    return xx, xy, mean, variance


def fit_concept_sets(
    xx: np.ndarray,
    xy: np.ndarray,
    label_mean: np.ndarray,
    label_variance: np.ndarray,
    *,
    budgets: Sequence[int] = (1, 4, 8, 16),
    penalty: float = 0.01,
    min_gain: float = 1e-12,
) -> ConceptSetFit:
    """Fit signed, annotation-guided feature sets at nested support budgets.

    ``xx`` is the centered predictor covariance, ``xy`` is the crosscovariance
    between those predictors and centered labels, and ``label_variance`` is
    the per-concept label variance. All moments must use the same fit samples
    and normalization. The objective is half the mean squared prediction
    error plus half ``penalty`` times the squared coefficient norm.

    Features may occur in several concept sets. At each step, the squared
    residual crosscovariance divided by the regularized conditional predictor
    variance gives twice the objective reduction. Exact ties choose the
    smallest feature index. A gain no greater than ``min_gain`` ends selection
    for that concept. Budgets larger than the feature count use all available
    features. A constant label always receives an empty set and zero weights.
    """
    requested = tuple(budgets)
    if not requested or any(isinstance(b, (bool, np.bool_)) or not isinstance(b, (int, np.integer))
                            or b < 1 for b in requested):
        raise ValueError("budgets must contain positive integer feature counts")
    ordered = sorted(set(int(b) for b in requested))
    xx, xy, mean, variance = _validate_moments(xx, xy, label_mean, label_variance, penalty, min_gain)
    feature_count, concept_count = xy.shape
    max_budget = max(ordered)
    gram = xx + penalty * np.eye(feature_count)
    diagonal = np.diag(gram)
    coefficients = {budget: np.zeros_like(xy) for budget in ordered}
    dense = (linalg.solve(gram, xy, assume_a="pos") if feature_count else np.zeros_like(xy))
    dense[:, variance == 0] = 0
    dense_objective = 0.5 * (variance - np.einsum("ic,ic->c", xy, dense))
    if np.any(dense_objective < -1e-9 * np.maximum(1.0, variance)):
        raise ValueError("Predictor and label moments imply a negative ridge objective")
    histories = np.repeat((variance / 2)[:, None], max_budget + 1, axis=1)
    gains = np.zeros((concept_count, max_budget), dtype=np.float64)
    sequences: list[list[int]] = []
    for concept in range(concept_count):
        selected: list[int] = []
        sequences.append(selected)
        if variance[concept] == 0 or feature_count == 0:
            continue
        response = xy[:, concept]
        beta = np.empty(0, dtype=np.float64)
        objective = float(variance[concept] / 2)
        for step in range(1, min(max_budget, feature_count) + 1):
            if selected:
                block = gram[:, selected]
                lower = linalg.cholesky(gram[np.ix_(selected, selected)], lower=True)
                solved = linalg.cho_solve((lower, True), block.T)
                residual = response - np.einsum("ij,j->i", block, beta)
                conditional = diagonal - np.einsum("ij,ji->i", block, solved)
            else:
                residual, conditional = response, diagonal
            eligible = np.ones(feature_count, dtype=bool)
            eligible[selected] = False
            if np.any(conditional[eligible] <= 0):
                raise ValueError("Regularized conditional predictor variance must be positive")
            candidate_gains = np.full(feature_count, -np.inf)
            candidate_gains[eligible] = 0.5 * residual[eligible] ** 2 / conditional[eligible]
            winner = int(np.argmax(candidate_gains))
            if candidate_gains[winner] <= min_gain:
                break
            selected.append(winner)
            beta = linalg.solve(gram[np.ix_(selected, selected)], response[selected], assume_a="pos")
            updated = float(0.5 * (variance[concept] - np.einsum("i,i->", response[selected], beta)))
            actual_gain = objective - updated
            if actual_gain < -1e-10 * max(1.0, abs(objective)):
                raise ArithmeticError("Refitting a larger support unexpectedly increased the objective")
            gains[concept, step - 1] = actual_gain
            objective = updated
            histories[concept, step:] = objective
            if step in coefficients:
                coefficients[step][selected, concept] = beta
        for budget in ordered:
            if budget >= len(selected):
                coefficients[budget][selected, concept] = beta
    return ConceptSetFit(coefficients, dense, sequences, histories, gains, mean.copy(), variance.copy(), penalty)
