"""Locally optimize cardinality-constrained ridge CCA in original coordinates.

The covariance metrics remain the full within-modality fit covariances plus the
ridge. Sparsity is imposed on the returned SAE coefficients, never on whitened
coordinates. The nonconvex solver makes no claim of globally optimal supports.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import linalg

from mm_sae.analysis.mapping_ablation import Projection, _matrix, _within_moments, fit_common_projection


_dot = linalg.get_blas_funcs(("dot",), dtype=np.float64)[0]
_gemv = linalg.get_blas_funcs(("gemv",), dtype=np.float64)[0]
_gemm = linalg.get_blas_funcs(("gemm",), dtype=np.float64)[0]


def _mv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _gemv(1., a, b)


def _solve_support(metric, linear, support):
    local = metric[np.ix_(support, support)]
    inverse = linalg.cho_solve(linalg.cho_factor(local, lower=True), np.eye(len(support)))
    solution = _mv(inverse, linear[support])
    objective_squared = max(float(_dot(linear[support], solution)), 0.)
    return inverse, solution, objective_squared


def _update_side(metric, linear, current, k, *, swap_steps, candidate_pool, tolerance):
    """Exactly refit a support, then accept only improving add/drop swaps.

    For fixed other-side coefficients the squared optimal objective on S is
    b_S.T @ inv(A_SS) @ b_S. Schur-complement gains rank candidate additions.
    Each shortlisted addition is paired with its best deletion, which is exact
    for that expanded support. This is a bounded local support search.
    """
    support = np.flatnonzero(current)
    if len(support) < k:
        order = np.argsort(-np.abs(linear), kind="stable")
        extra = order[~np.isin(order, support)][:k - len(support)]
        support = np.sort(np.concatenate([support, extra]))
    inverse, solution, value = _solve_support(metric, linear, support)
    accepted = 0
    for _ in range(swap_steps):
        if k == len(metric):
            break
        outside = np.flatnonzero(~np.isin(np.arange(len(metric)), support))
        coupling = metric[np.ix_(outside, support)]
        transformed = _gemm(1., coupling, inverse)
        conditional = np.diag(metric)[outside] - np.sum(transformed * coupling, axis=1)
        remainder = linear[outside] - _mv(coupling, solution)
        gains = remainder ** 2 / np.maximum(conditional, np.finfo(float).tiny)
        candidates = outside[np.argsort(-gains, kind="stable")[:candidate_pool]]
        best_support, best_value = support, value
        for candidate in candidates:
            expanded = np.append(support, candidate)
            expanded_inverse, expanded_solution, expanded_value = _solve_support(metric, linear, expanded)
            losses = expanded_solution ** 2 / np.diag(expanded_inverse)
            drop = int(np.argmin(losses))
            candidate_value = expanded_value - float(losses[drop])
            if candidate_value > best_value + tolerance * max(1., best_value):
                best_support = np.sort(np.delete(expanded, drop))
                best_value = candidate_value
        if np.array_equal(best_support, support):
            break
        support = best_support
        inverse, solution, value = _solve_support(metric, linear, support)
        accepted += 1
    result = np.zeros_like(current)
    if value > np.finfo(float).tiny:
        result[support] = solution / np.sqrt(value)
    else:
        # A zero cross moment has a flat objective, but still requires unit norm.
        result = current.copy()
        if not np.any(result):
            result[support[0]] = 1.
        result /= np.sqrt(float(_dot(result, _mv(metric, result))))
    return result, accepted


def _normalize_pruned(values, metric, k):
    order = np.argsort(-np.abs(values), kind="stable")[:k]
    result = np.zeros_like(values)
    result[order] = values[order]
    if not np.any(result):
        result[0] = 1.
    return result / np.sqrt(float(_dot(result, _mv(metric, result))))


def fit_sparse_cca(
    xx, cross, yy, dimensions: int = 256, *, k: int = 8, ridge: float = .01,
    initial_image=None, initial_text=None, max_iter: int = 100, tolerance: float = 1e-7,
    swap_steps: int = 2, candidate_pool: int = 8,
    progress: Callable[[dict], None] | None = None,
) -> Projection:
    """Fit sequential sparse rank-one ridge CCA components using fit moments only.

    At component j, maximize u.T @ C_res @ v subject to u.T @ A @ u =
    v.T @ B @ v = 1 and at most k nonzeros per side, where A = xx + ridge*I
    and B = yy + ridge*I. Alternate exact coefficient refits and improving
    support swaps. Dense ridge-CCA columns give deterministic initial supports.

    Deflate C_res by rho*(A@u)*(B@v).T, with rho = u.T @ C_res @ v. This is
    rank-one deflation in the ridge-whitened metric, expressed entirely using
    original coordinates. Sparse components need not be covariance-orthogonal.
    With unrestricted supports, exact dense initial directions, and nondegenerate
    singular values this reproduces ordinary sequential ridge CCA.

    After extraction, independently orient each returned text column to have
    nonnegative covariance with its image column in the ORIGINAL fit moments.
    This uses no labels and does not change the preceding residual optimization
    or deflation. Diagnostics retain the raw solver values and output signs.

    Convergence means small objective change and unchanged supports for this
    bounded local search. It does not certify the global cardinality optimum or
    even exhaustive one-swap optimality beyond the shortlisted candidates.
    """
    cross = _matrix(cross, "cross")
    ni, nt = cross.shape
    xx, yy = _within_moments(xx, ni, "xx"), _within_moments(yy, nt, "yy")
    for name, value in (("dimensions", dimensions), ("k", k), ("max_iter", max_iter),
                        ("swap_steps", swap_steps), ("candidate_pool", candidate_pool)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not min(ni, nt) or not np.isfinite(ridge) or ridge < 0:
        raise ValueError("Nonempty moments and finite nonnegative ridge are required")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    dimensions = min(dimensions, ni, nt)
    a, b = xx + ridge * np.eye(ni), yy + ridge * np.eye(nt)
    linalg.cholesky(a, lower=True)
    linalg.cholesky(b, lower=True)
    if (initial_image is None) != (initial_text is None):
        raise ValueError("Both initial coefficient matrices must be supplied together")
    if initial_image is None:
        initial = fit_common_projection(xx, cross, yy, dimensions, whiten=True, ridge=ridge)
        initial_image, initial_text = initial.image, initial.text
    initial_image = _matrix(initial_image, "initial_image")
    initial_text = _matrix(initial_text, "initial_text")
    if initial_image.shape != (ni, dimensions) or initial_text.shape != (nt, dimensions):
        raise ValueError("Initial coefficients must match feature counts and retained dimensions")
    image, text = np.zeros((ni, dimensions)), np.zeros((nt, dimensions))
    residual = cross.copy()
    diagnostics = []
    values = []
    for component in range(dimensions):
        u = _normalize_pruned(initial_image[:, component], a, min(k, ni))
        v = _normalize_pruned(initial_text[:, component], b, min(k, nt))
        initial_i, initial_t = u != 0, v != 0
        objective = float(_dot(u, _mv(residual, v)))
        if objective < 0:
            v = -v
            objective = -objective
        history = [objective]
        converged, swaps = False, 0
        for iteration in range(1, max_iter + 1):
            old_i, old_t = u != 0, v != 0
            u, changed_i = _update_side(a, _mv(residual, v), u, min(k, ni),
                                        swap_steps=swap_steps, candidate_pool=candidate_pool,
                                        tolerance=tolerance)
            v, changed_t = _update_side(b, _mv(residual.T, u), v, min(k, nt),
                                        swap_steps=swap_steps, candidate_pool=candidate_pool,
                                        tolerance=tolerance)
            swaps += changed_i + changed_t
            updated = float(_dot(u, _mv(residual, v)))
            if updated < objective - 1e-9 * max(1., abs(objective)):
                raise RuntimeError("Sparse CCA objective decreased beyond numerical tolerance")
            history.append(updated)
            stable_support = np.array_equal(old_i, u != 0) and np.array_equal(old_t, v != 0)
            converged = stable_support and updated - objective <= tolerance * max(1., abs(objective))
            objective = updated
            if converged:
                break
        image[:, component], text[:, component] = u, v
        values.append(objective)
        original = float(_dot(u, _mv(cross, v)))
        record = {
            "component": component, "iterations": iteration, "converged": bool(converged),
            "stop_reason": "objective_tolerance_and_stable_support" if converged else "max_iter",
            "residual_objective": objective, "original_regularized_correlation": original,
            "original_unregularized_correlation": original / np.sqrt(
                float(_dot(u, _mv(xx, u))) * float(_dot(v, _mv(yy, v))))
            if float(_dot(u, _mv(xx, u))) * float(_dot(v, _mv(yy, v))) > 0 else None,
            "objective_history": history, "accepted_support_swaps": swaps,
            "image_support": np.flatnonzero(u).tolist(), "text_support": np.flatnonzero(v).tolist(),
            "image_support_changed_count": int(np.count_nonzero(initial_i != (u != 0))),
            "text_support_changed_count": int(np.count_nonzero(initial_t != (v != 0))),
            "image_regularized_variance": float(_dot(u, _mv(a, u))),
            "text_regularized_variance": float(_dot(v, _mv(b, v))),
        }
        diagnostics.append(record)
        if progress is not None:
            progress(record)
        residual -= objective * np.outer(_mv(a, u), _mv(b, v))
    orientations = np.asarray([1 if d["original_regularized_correlation"] >= 0 else -1
                               for d in diagnostics])
    text *= orientations[None, :]
    for record, sign in zip(diagnostics, orientations, strict=True):
        record["text_output_orientation"] = int(sign)
        record["output_original_regularized_correlation"] = sign * record["original_regularized_correlation"]
        raw_correlation = record["original_unregularized_correlation"]
        record["output_original_unregularized_correlation"] = (
            sign * raw_correlation if raw_correlation is not None else None)
    return Projection(image, text, np.asarray(values), {
        "method": "cardinality_constrained_ridge_cca_with_sequential_rank_one_deflation",
        "constraint_coordinates": "selected_original_standardized_SAE_features",
        "dimensions": dimensions, "k_requested": int(k), "ridge": float(ridge),
        "max_iter": max_iter, "tolerance": tolerance, "swap_steps": swap_steps,
        "candidate_pool": candidate_pool, "initialization": "top_k_original_coefficients_of_dense_ridge_cca",
        "normalization": "unit_variance_in_full_within_modality_covariance_plus_ridge",
        "deflation": "C_res -= rho * outer((xx+ridge*I)@u, (yy+ridge*I)@v)",
        "deflation_rho": "u.T @ C_res @ raw_v_before_deflation_and_output_orientation",
        "output_text_orientation": "nonnegative_original_fit_cross_covariance_no_labels",
        "output_text_orientation_factors": orientations.tolist(),
        "output_text_flipped_components": np.flatnonzero(orientations < 0).tolist(),
        "raw_solver_diagnostics_preserved": True,
        "sparse_components_are_covariance_orthogonal": False,
        "singular_values_field": "residual_objectives_not_original_canonical_correlations",
        "optimizer": "alternating_exact_support_refit_and_schur_complement_shortlisted_add_drop_swaps",
        "optimality_claim": "local_bounded_search_only_no_global_or_exhaustive_swap_guarantee",
        "fit_moments_only": True, "singular_value_weighting": False,
        "converged_components": sum(d["converged"] for d in diagnostics), "components": diagnostics,
    })
