"""Direct alignment of a signed association matrix without activation regression.

Rows are image features and columns are text features. Returned weights retain
solver mass and scale. Any normalization for downstream scoring belongs to the
caller, since it changes the fitted solution and its constraints.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import linalg, sparse
from scipy.optimize import linear_sum_assignment, linprog, minimize
from scipy.special import logsumexp

Progress = Callable[[dict[str, Any]], None]
_gemm = linalg.get_blas_funcs(("gemm",), dtype=np.float64)[0]


@dataclass
class MappingResult:
    weights: np.ndarray
    metadata: dict[str, Any]
    u: np.ndarray | None = None
    v: np.ndarray | None = None

    @property
    def U(self) -> np.ndarray | None:
        return self.u

    @property
    def V(self) -> np.ndarray | None:
        return self.v


def _number(options: Mapping[str, Any], name: str, default: float, *, positive: bool = False) -> float:
    value = float(options.get(name, default))
    if not np.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return value


def _integer(options: Mapping[str, Any], name: str, default: int, *, minimum: int = 0) -> int:
    raw = options.get(name, default)
    value = int(raw)
    if value != raw or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def _penalty(options: Mapping[str, Any], default: float, *, positive: bool) -> float:
    raw = options.get("lambda_", options.get("lambda", options.get("regularization", default)))
    value = _number({"lambda_": raw}, "lambda_", default, positive=positive)
    if value < 0:
        raise ValueError("lambda_ must be nonnegative")
    return value


def _capacity(raw: Any, size: int, limit: int, name: str, *, integral: bool) -> np.ndarray:
    value = np.asarray(raw, dtype=np.float64)
    if value.ndim == 0:
        value = np.full(size, float(value))
    if value.shape != (size,) or not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError(f"{name} must be a nonnegative finite scalar or a vector of length {size}")
    if integral and np.any(value != np.floor(value)):
        raise ValueError(f"{name} must contain integer capacities")
    return np.minimum(value, limit)


def _emit(progress: Progress | None, method: str, **details: Any) -> None:
    if progress is not None:
        progress({"method": method, **details})


def _top_indices(values: np.ndarray, count: int) -> np.ndarray:
    """Stable ordering makes ties independent of partition implementation details."""
    return np.argsort(-values, kind="stable")[:count]


def _keep_top(values: np.ndarray, count: int) -> np.ndarray:
    if count >= values.size:
        return values
    result = np.zeros_like(values)
    indices = _top_indices(values, count)
    result[indices] = values[indices]
    return result


def _discrete(c: np.ndarray, method: str, options: Mapping[str, Any]) -> MappingResult:
    weights = np.zeros_like(c)
    if method == "hungarian":
        rows, columns = linear_sum_assignment(c, maximize=True)
        weights[rows, columns] = 1
    elif method == "greedy":
        weights[np.arange(c.shape[0]), np.argmax(c, axis=1)] = 1
    else:
        count = min(_integer(options, "k", options.get("k_image", 2)), c.shape[1])
        indices = np.argsort(-c, axis=1, kind="stable")[:, :count]
        np.put_along_axis(weights, indices, 1, axis=1)
    return MappingResult(weights, {"objective": float(np.sum(c * weights)), "converged": True,
                                   "iterations": 1, "residual": 0.0})


def _b_matching(c: np.ndarray, options: Mapping[str, Any]) -> MappingResult:
    """Solve the binary bipartite problem through its integral linear relaxation."""
    n_image, n_text = c.shape
    tau = _number(options, "tau", 0)
    image_cap = _capacity(options.get("k_image", 2), n_image, n_text, "k_image", integral=True)
    text_cap = _capacity(options.get("k_text", 2), n_text, n_image, "k_text", integral=True)
    rows, columns = np.nonzero((c > tau) & (image_cap[:, None] > 0) & (text_cap[None, :] > 0))
    count = len(rows)
    weights = np.zeros_like(c)
    budget = options.get("edge_budget")
    if budget is not None:
        budget = _integer(options, "edge_budget", count)
    metadata: dict[str, Any] = {"tau": tau, "candidate_edges": count, "edge_budget": budget,
                                "solver": "scipy.linprog highs", "optimality": "global",
                                "converged": True, "iterations": 0, "residual": 0.0,
                                "integrality_residual": 0.0, "objective": 0.0}
    if count == 0 or budget == 0:
        return MappingResult(weights, metadata)
    edge_indices = np.arange(count)
    incidence_rows = np.r_[rows, n_image + columns]
    incidence_columns = np.r_[edge_indices, edge_indices]
    capacities = np.r_[image_cap, text_cap]
    if budget is not None:
        incidence_rows = np.r_[incidence_rows, np.full(count, n_image + n_text)]
        incidence_columns = np.r_[incidence_columns, edge_indices]
        capacities = np.r_[capacities, budget]
    incidence = sparse.coo_matrix((np.ones(len(incidence_rows)), (incidence_rows, incidence_columns)),
                                  shape=(len(capacities), count)).tocsr()
    result = linprog(-(c[rows, columns] - tau), A_ub=incidence, b_ub=capacities,
                     bounds=(0, 1), method="highs")
    if not result.success:
        raise RuntimeError(f"Bipartite matching solver failed: {result.message}")
    integrality = float(np.max(np.abs(result.x - np.rint(result.x))))
    if integrality > 1e-6:
        raise RuntimeError(f"Expected an integral bipartite solution, residual was {integrality:g}")
    weights[rows, columns] = np.rint(result.x)
    if np.any(weights.sum(1) > image_cap) or np.any(weights.sum(0) > text_cap):
        raise RuntimeError("Rounded bipartite solution violates a degree bound")
    if budget is not None and weights.sum() > budget:
        raise RuntimeError("Rounded bipartite solution violates the edge budget")
    metadata.update(objective=float(np.sum((c - tau) * weights)),
                    iterations=int(result.nit), integrality_residual=integrality,
                    solver_message=str(result.message))
    return MappingResult(weights, metadata)


def _feasible_transport(weights: np.ndarray, image_cap: np.ndarray, text_cap: np.ndarray) -> np.ndarray:
    """Remove numerical constraint excess monotonically without adding mass."""
    weights = np.clip(weights, 0, 1)
    row_sum = weights.sum(1)
    scale = np.ones_like(row_sum)
    np.divide(image_cap, row_sum, out=scale, where=row_sum > image_cap)
    weights *= scale[:, None]
    col_sum = weights.sum(0)
    scale = np.ones_like(col_sum)
    np.divide(text_cap, col_sum, out=scale, where=col_sum > text_cap)
    weights *= scale[None, :]
    return weights


def _project_row_simplex(values: np.ndarray, capacities: np.ndarray) -> np.ndarray:
    """Euclidean projection onto nonnegative rows with an upper sum bound."""
    result = np.maximum(values, 0)
    result[capacities == 0] = 0
    active = (result.sum(1) > capacities) & (capacities > 0)
    if np.any(active):
        selected = values[active]
        ordered = np.sort(selected, axis=1)[:, ::-1]
        cumulative = np.cumsum(ordered, axis=1) - capacities[active, None]
        ranks = np.arange(1, values.shape[1] + 1)
        support = np.sum(ordered - cumulative / ranks > 0, axis=1) - 1
        threshold = cumulative[np.arange(len(selected)), support] / (support + 1)
        result[active] = np.maximum(selected - threshold[:, None], 0)
    return result


def _refine_transport(raw: np.ndarray, score: np.ndarray, dual: np.ndarray, penalty: float,
                      image_cap: np.ndarray, text_cap: np.ndarray, max_iter: int, tol: float,
                      progress: Progress | None, iteration_offset: int) -> tuple[np.ndarray, float, int]:
    """Warm-start Dykstra projections using the approximate row/column duals.

Each correction vector supplies a valid dual bound through the support function
of its constraint set. This distinguishes objective convergence from merely
small changes between successive feasible iterates.
    """
    n_image = len(image_cap)
    row_correction = np.broadcast_to(dual[:n_image, None] / penalty, raw.shape).copy()
    col_correction = np.broadcast_to(dual[None, n_image:] / penalty, raw.shape).copy()
    box_correction = score / penalty - row_correction - col_correction - raw
    dual_value = float("inf")
    for iteration in range(1, max_iter + 1):
        shifted = raw + row_correction
        projected = _project_row_simplex(shifted, image_cap)
        row_correction = shifted - projected
        shifted = projected + col_correction
        projected = _project_row_simplex(shifted.T, text_cap).T
        col_correction = shifted - projected
        shifted = projected + box_correction
        raw = np.clip(shifted, 0, 1)
        box_correction = shifted - raw
        # The maintained identity is score/lambda = raw + sum(corrections).
        support_value = (np.sum(image_cap * np.maximum(row_correction.max(1), 0))
                         + np.sum(text_cap * np.maximum(col_correction.max(0), 0))
                         + np.maximum(box_correction, 0).sum())
        dual_value = float(penalty * (.5 * np.sum(raw**2) + support_value))
        feasible = _feasible_transport(raw.copy(), image_cap, text_cap)
        primal = float(np.sum(score * feasible - .5 * penalty * feasible**2))
        residual = max(0.0, float(np.max(raw.sum(1) - image_cap)),
                       float(np.max(raw.sum(0) - text_cap)))
        if residual <= tol and dual_value - primal <= tol * max(1.0, abs(primal)):
            break
        if iteration % 20 == 0:
            _emit(progress, "sparse_transport", projection_iteration=iteration,
                  iterations=iteration_offset + iteration,
                  residual=residual, duality_gap=max(0.0, dual_value - primal))
    return raw, dual_value, iteration


def _sparse_transport(c: np.ndarray, options: Mapping[str, Any], progress: Progress | None) -> MappingResult:
    """Solve the strictly concave transport problem using its convex dual.

The dual has one nonnegative multiplier per row and column. Eliminating the
box-constrained primal variables gives clip((C - tau - alpha - beta)/lambda, 0, 1).
The reported duality gap bounds suboptimality of the returned feasible weights.
    """
    n_image, n_text = c.shape
    tau = _number(options, "tau", 0)
    penalty = _penalty(options, .1, positive=True)
    tol = _number(options, "tol", 1e-8, positive=True)
    max_iter = _integer(options, "max_iter", 2000, minimum=1)
    image_cap = _capacity(options.get("k_image", 2), n_image, n_text, "k_image", integral=False)
    text_cap = _capacity(options.get("k_text", 2), n_text, n_image, "k_text", integral=False)
    score = c - tau
    iteration = 0

    def objective(dual: np.ndarray) -> tuple[float, np.ndarray]:
        shifted = score - dual[:n_image, None] - dual[None, n_image:]
        weights = np.clip(shifted / penalty, 0, 1)
        value = (np.sum(shifted * weights - .5 * penalty * weights**2)
                 + np.sum(image_cap * dual[:n_image]) + np.sum(text_cap * dual[n_image:]))
        gradient = np.r_[image_cap - weights.sum(1), text_cap - weights.sum(0)]
        return float(value), gradient

    def callback(_dual: np.ndarray) -> None:
        nonlocal iteration
        iteration += 1
        if iteration % 20 == 0:
            _emit(progress, "sparse_transport", iteration=iteration, max_iter=max_iter)

    result = minimize(objective, np.zeros(n_image + n_text), method="L-BFGS-B", jac=True,
                      bounds=[(0, None)] * (n_image + n_text), callback=callback,
                      options={"maxiter": max_iter, "ftol": 1e-15, "gtol": min(tol * .1, 1e-10),
                               "maxls": 50, "maxcor": 20})
    dual = np.asarray(result.x)
    raw = np.clip((score - dual[:n_image, None] - dual[None, n_image:]) / penalty, 0, 1)
    raw_residual = max(0.0, float(np.max(raw.sum(1) - image_cap)), float(np.max(raw.sum(0) - text_cap)))
    weights = _feasible_transport(raw.copy(), image_cap, text_cap)
    primal = float(np.sum(score * weights - .5 * penalty * weights**2))
    dual_value, gradient = objective(dual)
    gap = max(0.0, dual_value - primal)
    projected_gradient = np.where(dual > 0, gradient, np.minimum(gradient, 0))
    kkt_residual = float(np.max(np.abs(projected_gradient)))
    refinement_iterations = 0
    if (raw_residual > tol or gap > tol * max(1.0, abs(primal))) and result.nit < max_iter:
        raw, dual_value, refinement_iterations = _refine_transport(
            raw, score, dual, penalty, image_cap, text_cap, max_iter - int(result.nit), tol,
            progress, int(result.nit))
        raw_residual = max(0.0, float(np.max(raw.sum(1) - image_cap)),
                           float(np.max(raw.sum(0) - text_cap)))
        weights = _feasible_transport(raw.copy(), image_cap, text_cap)
        primal = float(np.sum(score * weights - .5 * penalty * weights**2))
        gap = max(0.0, dual_value - primal)
    converged = gap <= tol * max(1.0, abs(primal)) and raw_residual <= tol
    return MappingResult(weights, {"tau": tau, "lambda_": penalty, "objective": primal,
                                   "dual_objective": dual_value, "duality_gap": gap,
                                   "converged": bool(converged),
                                   "iterations": int(result.nit) + refinement_iterations,
                                   "optimizer_iterations": int(result.nit),
                                   "projection_iterations": refinement_iterations,
                                   "residual": raw_residual,
                                   "dual_gradient_residual_before_refinement": kkt_residual,
                                   "feasibility_correction": float(np.max(np.abs(raw - weights))),
                                   "solver": "L-BFGS-B convex dual with Dykstra projection refinement",
                                   "solver_success": bool(result.success),
                                   "solver_message": str(result.message), "max_iter": max_iter,
                                   "tol": tol, "optimality": "convex global optimum within reported gap"})


def _mass(raw: Any, size: int, name: str) -> np.ndarray:
    value = np.asarray(raw, dtype=np.float64)
    if value.shape != (size,) or not np.isfinite(value).all() or np.any(value < 0) or value.sum() <= 0:
        raise ValueError(f"{name} must be a nonnegative vector of length {size} with positive total mass")
    return value / value.sum()


def _sinkhorn(c: np.ndarray, options: Mapping[str, Any], progress: Progress | None) -> MappingResult:
    epsilon = _number(options, "epsilon", .05, positive=True)
    tol = _number(options, "tol", 1e-8, positive=True)
    max_iter = _integer(options, "max_iter", 2000, minimum=1)
    image_mass = _mass(options.get("row_mass", np.ones(c.shape[0])), c.shape[0], "row_mass")
    text_mass = _mass(options.get("col_mass", np.ones(c.shape[1])), c.shape[1], "col_mass")
    ii, jj = np.flatnonzero(image_mass), np.flatnonzero(text_mass)
    # Subtracting a common constant preserves a fixed-mass transport solution.
    active = c[np.ix_(ii, jj)]
    kernel = (active - np.max(active)) / epsilon
    log_image, log_text = np.log(image_mass[ii]), np.log(text_mass[jj])
    log_v = np.zeros(len(jj))
    residual = float("inf")
    mass = np.zeros_like(active)
    for iteration in range(1, max_iter + 1):
        log_u = log_image - logsumexp(kernel + log_v[None, :], axis=1)
        log_v = log_text - logsumexp(kernel + log_u[:, None], axis=0)
        # Recenter the two dual scalings to limit numerical drift without changing mass.
        offset = float(np.mean(log_u))
        log_u -= offset
        log_v += offset
        if iteration == 1 or iteration % 10 == 0 or iteration == max_iter:
            mass = np.exp(kernel + log_u[:, None] + log_v[None, :])
            residual = max(float(np.max(np.abs(mass.sum(1) - image_mass[ii]))),
                           float(np.max(np.abs(mass.sum(0) - text_mass[jj]))))
            if residual <= tol:
                break
        if iteration % 100 == 0:
            _emit(progress, "sinkhorn", iteration=iteration, max_iter=max_iter, residual=residual)
    weights = np.zeros_like(c)
    weights[np.ix_(ii, jj)] = mass
    positive = weights > 0
    entropy = -float(np.sum(weights[positive] * (np.log(weights[positive]) - 1)))
    return MappingResult(weights, {"epsilon": epsilon, "objective": float(np.sum(c * weights)) + epsilon * entropy,
                                   "association_objective": float(np.sum(c * weights)),
                                   "converged": residual <= tol, "iterations": iteration,
                                   "residual": residual, "tol": tol, "max_iter": max_iter,
                                   "total_mass": float(weights.sum()), "row_mass": image_mass.tolist(),
                                   "col_mass": text_mass.tolist(), "solver": "log-domain Sinkhorn",
                                   "optimality": "balanced entropy-regularized transport"})


def _factor_objective(c: np.ndarray, u: np.ndarray, v: np.ndarray, penalty: float) -> float:
    error = (np.sum(c**2) - 2 * np.sum(u * _gemm(alpha=1, a=c, b=v))
             + np.sum(_gemm(alpha=1, a=u, b=u, trans_a=True)
                      * _gemm(alpha=1, a=v, b=v, trans_a=True)))
    return float(.5 * max(0.0, float(error)) + penalty * (u.sum() + v.sum()))


def _update_factor(cross: np.ndarray, other: np.ndarray, current: np.ndarray,
                   penalty: float, support: int) -> None:
    gram = _gemm(alpha=1, a=other, b=other, trans_a=True)
    for group in range(current.shape[1]):
        norm = float(gram[group, group])
        if norm <= np.finfo(float).tiny:
            current[:, group] = 0
            continue
        linear = cross[:, group] - np.sum(current * gram[:, group], axis=1) + current[:, group] * norm
        candidate = np.maximum((linear - penalty) / norm, 0)
        current[:, group] = _keep_top(candidate, support)


def _balance_factors(u: np.ndarray, v: np.ndarray) -> None:
    """Minimize each pair's L1 penalty over scales preserving its outer product."""
    image_mass, text_mass = u.sum(0), v.sum(0)
    active = (image_mass > 0) & (text_mass > 0)
    u[:, ~active] = 0
    v[:, ~active] = 0
    scale = np.sqrt(text_mass[active] / image_mass[active])
    u[:, active] *= scale
    v[:, active] /= scale


def _factorization(c: np.ndarray, options: Mapping[str, Any], progress: Progress | None) -> MappingResult:
    rank = _integer(options, "rank", 8, minimum=1)
    image_support = min(_integer(options, "k_image", min(10, c.shape[0])), c.shape[0])
    text_support = min(_integer(options, "k_text", min(10, c.shape[1])), c.shape[1])
    penalty = _penalty(options, .01, positive=False)
    tol = _number(options, "tol", 1e-7, positive=True)
    max_iter = _integer(options, "max_iter", 200, minimum=1)
    n_init = _integer(options, "n_init", 3, minimum=1)
    seed = _integer(options, "seed", 0)
    restart_objectives: list[float] = []
    initial_objectives: list[float] = []
    best: MappingResult | None = None
    total_iterations = 0
    restart_iterations: list[int] = []
    scale = np.sqrt(max(float(np.maximum(c, 0).mean()), 1e-8) / rank)
    for restart, child_seed in enumerate(np.random.SeedSequence(seed).spawn(n_init)):
        rng = np.random.default_rng(child_seed)
        u = rng.uniform(.5, 1.5, size=(c.shape[0], rank)) * scale
        v = rng.uniform(.5, 1.5, size=(c.shape[1], rank)) * scale
        for group in range(rank):
            u[:, group] = _keep_top(u[:, group], image_support)
            v[:, group] = _keep_top(v[:, group], text_support)
        history = [_factor_objective(c, u, v, penalty)]
        initial_objectives.append(history[0])
        converged = False
        relative_change = float("inf")
        for iteration in range(1, max_iter + 1):
            _update_factor(_gemm(alpha=1, a=c, b=v), v, u, penalty, image_support)
            _update_factor(_gemm(alpha=1, a=c, b=u, trans_a=True), u, v, penalty, text_support)
            _balance_factors(u, v)
            total_iterations += 1
            value = _factor_objective(c, u, v, penalty)
            previous = history[-1]
            if value > previous + 1e-9 * max(1.0, abs(previous)):
                raise RuntimeError("Exact factor block updates increased the objective")
            history.append(value)
            relative_change = abs(previous - value) / max(1.0, abs(previous))
            if iteration == 1 or iteration % 10 == 0:
                _emit(progress, "sparse_factorization", restart=restart, n_init=n_init,
                      iteration=iteration, iterations=total_iterations, max_iter=max_iter, objective=value)
            if relative_change <= tol:
                converged = True
                break
        restart_objectives.append(history[-1])
        restart_iterations.append(iteration)
        if best is None or history[-1] < best.metadata["objective"]:
            best = MappingResult(_gemm(alpha=1, a=u, b=v, trans_b=True),
                                 {"objective": history[-1], "initial_objective": history[0],
                                          "objective_history": history, "converged": converged,
                                          "iterations": iteration, "residual": relative_change,
                                          "best_restart": restart, "rank": rank, "lambda_": penalty,
                                          "k_image": image_support, "k_text": text_support,
                                          "max_iter": max_iter, "tol": tol,
                                          "solver": "exact nonnegative sparse column updates with L1 scale minimization",
                                          "optimality": "local alternating minimization, no global guarantee"},
                                 u.copy(), v.copy())
    assert best is not None
    best.metadata.update(seed=seed, n_init=n_init, restart_objectives=restart_objectives,
                         initial_objectives=initial_objectives, total_iterations=total_iterations,
                         restart_iterations=restart_iterations)
    return best


def fit_mapping(c: np.ndarray, method: str, options: Mapping[str, Any] | None = None,
                *, progress: Progress | None = None) -> MappingResult:
    """Fit nonnegative weights directly from a finite signed association matrix.

``k_image`` and ``k_text`` cap row and column degree for bipartite matching, or
row and column mass for sparse transport. For sparse factorization they instead
cap the number of nonzero image and text features in each factor column.
``k`` controls the number of selected columns per row for top-k matching.
``lambda_`` controls quadratic transport or factor L1 regularization. Sinkhorn
uses ``epsilon`` and independently normalized ``row_mass`` and ``col_mass``.
All iterative methods expose convergence diagnostics without hiding an exhausted
iteration limit. Factorization uses the original signed matrix in its loss.
    """
    c = np.asarray(c, dtype=np.float64)
    if c.ndim != 2 or not np.isfinite(c).all():
        raise ValueError("c must be a finite two-dimensional association matrix")
    aliases = {"row_argmax": "greedy", "rowtopk": "topk", "row_topk": "topk",
               "b_matching": "partial_b_matching"}
    method = aliases.get(method, method)
    allowed = {"hungarian", "greedy", "topk", "partial_b_matching", "sparse_transport",
               "sinkhorn", "sparse_factorization"}
    if method not in allowed:
        raise ValueError(f"Unknown mapping method {method!r}")
    options = {} if options is None else options
    _emit(progress, method, status="started", shape=list(c.shape))
    if c.size == 0:
        result = MappingResult(np.zeros_like(c), {"objective": 0.0, "converged": True,
                                                  "iterations": 0, "residual": 0.0, "empty": True})
    elif method in {"hungarian", "greedy", "topk"}:
        result = _discrete(c, method, options)
    elif method == "partial_b_matching":
        result = _b_matching(c, options)
    elif method == "sparse_transport":
        result = _sparse_transport(c, options, progress)
    elif method == "sinkhorn":
        result = _sinkhorn(c, options, progress)
    else:
        result = _factorization(c, options, progress)
    if not np.isfinite(result.weights).all() or not np.isfinite(result.metadata["objective"]):
        raise RuntimeError(f"{method} produced nonfinite weights or objective")
    result.metadata.update(method=method, shape=list(c.shape), total_mass=float(result.weights.sum()),
                           nonzero_edges=int(np.count_nonzero(result.weights)))
    _emit(progress, method, status="finished", converged=result.metadata["converged"],
          iterations=result.metadata.get("total_iterations", result.metadata["iterations"]),
          residual=result.metadata["residual"])
    return result
