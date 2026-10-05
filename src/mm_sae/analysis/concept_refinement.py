"""Improve fixed-size concept regressions with exact one-feature swaps.

The objective and input moments match ``concept_sets.fit_concept_sets``.
Each pass scans every selected-feature removal and every unselected-feature
addition, chooses the globally best single swap, and refits all coefficients.
This finds a one-swap local optimum when convergence is reported, not a global
optimum over every feature subset. It does not use evaluation samples.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import linalg

from mm_sae.analysis.concept_sets import _validate_moments

_gemm = linalg.get_blas_funcs(("gemm",), dtype=np.float64)[0]


@dataclass(frozen=True)
class ConceptSwap:
    """One accepted exchange, in original predictor-coordinate indices."""

    removed_feature: int
    added_feature: int
    objective: float
    gain: float
    predicted_gain: float


@dataclass(frozen=True)
class ConceptRefinementTrace:
    """The complete optimization record for one concept.

    ``objective_history`` starts after refitting the initial support, followed
    by each accepted swap. ``initial_objective`` evaluates the supplied weights;
    ``refit_gain`` separates any fixed-support correction from swap gains.
    ``best_remaining_gain`` is evaluated by an exhaustive final one-swap scan.
    """

    concept: int
    initial_support: tuple[int, ...]
    final_support: tuple[int, ...]
    initial_objective: float
    refit_gain: float
    objective_history: tuple[float, ...]
    swaps: tuple[ConceptSwap, ...]
    converged: bool
    hit_pass_limit: bool
    best_remaining_gain: float


@dataclass(frozen=True)
class ConceptRefinementFit:
    coefficients: np.ndarray
    traces: tuple[ConceptRefinementTrace, ...]
    label_mean: np.ndarray
    label_variance: np.ndarray
    penalty: float
    max_passes: int
    tolerance: float


RefinementCallback = Callable[[int, int, ConceptRefinementTrace], None]


def _refit(gram: np.ndarray, response: np.ndarray, support: list[int]):
    if not support:
        return np.empty(0), np.empty((0, 0))
    local = gram[np.ix_(support, support)]
    factor = linalg.cho_factor(local, lower=True, check_finite=False)
    beta = linalg.cho_solve(factor, response[support], check_finite=False)
    inverse = linalg.cho_solve(factor, np.eye(len(support)), check_finite=False)
    return beta, inverse


def _best_swap(
    gram: np.ndarray,
    response: np.ndarray,
    support: list[int],
    beta: np.ndarray,
    inverse: np.ndarray,
) -> tuple[float, int | None, int | None]:
    """Return the exact best objective gain among every one-drop/one-add move.

    A Schur-complement identity evaluates all supports S-minus-r-plus-j without
    separately solving their normal equations. Ties prefer the lowest removed
    feature index and then the lowest added feature index. Support must be sorted.
    """
    if not support or len(support) == len(response):
        return 0.0, None, None
    eligible = np.ones(len(response), dtype=bool)
    eligible[support] = False
    candidates = np.flatnonzero(eligible)
    block = gram[np.ix_(candidates, np.asarray(support, dtype=np.intp))]
    # BLAS avoids NumPy matmul overflow warnings on some Accelerate builds.
    projected = _gemm(alpha=1.0, a=block, b=inverse)
    residual = response[candidates] - np.einsum("ij,j->i", block, beta)
    conditional = np.diag(gram)[candidates] - np.einsum("ij,ij->i", projected, block)
    inverse_diag = np.diag(inverse)
    if np.any(inverse_diag <= 0):
        raise ArithmeticError("A regularized support inverse must have positive diagonal")
    reduced_residual = residual[:, None] + projected * (beta / inverse_diag)[None, :]
    reduced_variance = conditional[:, None] + projected**2 / inverse_diag[None, :]
    if not np.isfinite(reduced_variance).all() or np.any(reduced_variance <= 0):
        raise ArithmeticError("Regularized conditional predictor variance must be positive")
    drop_cost = 0.5 * beta**2 / inverse_diag
    gains = (0.5 * reduced_residual**2 / reduced_variance - drop_cost[None, :]).T
    if not np.isfinite(gains).all():
        raise ArithmeticError("Swap gains must be finite")
    row, col = np.unravel_index(int(np.argmax(gains)), gains.shape)
    return float(gains[row, col]), support[row], int(candidates[col])


def refine_concept_supports(
    xx: np.ndarray,
    xy: np.ndarray,
    label_mean: np.ndarray,
    label_variance: np.ndarray,
    initial_coefficients: np.ndarray,
    *,
    penalty: float = 0.01,
    max_passes: int = 10,
    tolerance: float = 1e-10,
    callback: RefinementCallback | None = None,
) -> ConceptRefinementFit:
    """Refine each concept's supplied support with exact signed ridge swaps.

    Rows of the coefficient matrix are predictor features and columns are
    concepts. The supplied nonzero coefficients define each initial support;
    no new budget is imposed and the support never grows. Predictor covariance
    ``xx``, predictor-label crosscovariance ``xy``, and the label moments must
    all come from the same centered fit data. The objective is half mean squared
    error plus half ``penalty`` times squared coefficient norm.

    ``max_passes`` bounds the number of accepted swaps for each concept. A final
    exhaustive scan distinguishes a true one-swap local optimum (within
    ``tolerance``) from termination at the bound. The callback runs once after
    each concept and receives completed count, total count, and its trace.
    """
    if (isinstance(max_passes, (bool, np.bool_))
            or not isinstance(max_passes, (int, np.integer)) or max_passes < 0):
        raise ValueError("max_passes must be a nonnegative integer")
    xx, xy, mean, variance = _validate_moments(
        xx, xy, label_mean, label_variance, penalty, tolerance,
    )
    supplied = np.asarray(initial_coefficients, dtype=np.float64)
    if supplied.shape != xy.shape or not np.isfinite(supplied).all():
        raise ValueError("initial_coefficients must be finite and match xy dimensions")
    gram = xx + penalty * np.eye(len(xx))
    coefficients = np.zeros_like(xy)
    traces: list[ConceptRefinementTrace] = []
    for concept in range(xy.shape[1]):
        response = xy[:, concept]
        original = supplied[:, concept]
        support = np.flatnonzero(original).tolist()
        initial_support = tuple(support)
        initial_objective = float(0.5 * variance[concept]
                                  - np.einsum("i,i->", original, response)
                                  + 0.5 * np.einsum("i,ij,j->", original, gram, original))
        beta, inverse = _refit(gram, response, support)
        objective = float(0.5 * (variance[concept] - np.einsum("i,i->", response[support], beta)))
        if objective < -1e-9 * max(1.0, float(variance[concept])):
            raise ValueError("Predictor and label moments imply a negative ridge objective")
        refit_gain = initial_objective - objective
        if refit_gain < -1e-10 * max(1.0, abs(initial_objective)):
            raise ArithmeticError("Refitting the initial support increased the objective")
        history = [objective]
        swaps: list[ConceptSwap] = []
        # The extra scan after the final accepted swap makes convergence exact.
        while True:
            gain, removed, added = _best_swap(gram, response, support, beta, inverse)
            if gain <= tolerance or len(swaps) >= max_passes:
                break
            if removed is None or added is None:
                raise ArithmeticError("A positive swap gain must identify both features")
            updated_support = sorted(feature for feature in support if feature != removed)
            updated_support.append(added)
            updated_support.sort()
            updated_beta, updated_inverse = _refit(gram, response, updated_support)
            updated = float(0.5 * (variance[concept]
                                  - np.einsum("i,i->", response[updated_support], updated_beta)))
            actual_gain = objective - updated
            if not np.isclose(actual_gain, gain, rtol=1e-7, atol=1e-10):
                raise ArithmeticError("Predicted swap gain disagrees with the refitted objective")
            if actual_gain < 0:
                raise ArithmeticError("A support swap unexpectedly increased the objective")
            swaps.append(ConceptSwap(removed, added, updated, actual_gain, gain))
            support, beta, inverse, objective = updated_support, updated_beta, updated_inverse, updated
            history.append(objective)
        coefficients[support, concept] = beta
        if not np.isfinite(beta).all():
            raise ArithmeticError("Refitted coefficients must be finite")
        converged = gain <= tolerance
        trace = ConceptRefinementTrace(
            concept=concept, initial_support=initial_support, final_support=tuple(support),
            initial_objective=initial_objective, refit_gain=refit_gain,
            objective_history=tuple(history), swaps=tuple(swaps), converged=converged,
            hit_pass_limit=not converged and len(swaps) >= max_passes,
            best_remaining_gain=max(0.0, gain),
        )
        traces.append(trace)
        if callback is not None:
            callback(concept + 1, xy.shape[1], trace)
    return ConceptRefinementFit(
        coefficients, tuple(traces), mean.copy(), variance.copy(), penalty, int(max_passes), tolerance,
    )
