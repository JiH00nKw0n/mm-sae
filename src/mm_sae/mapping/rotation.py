"""Orthogonal rotations of both modalities; objectives use training data only."""

import numpy as np
from scipy.linalg import polar


def row_normalize(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)


def rotate(left, right, kind, max_iter, tol):
    rank = left.shape[1]
    initial = np.eye(rank)

    def objective(r, gradient):
        a, b = left @ r, right @ r
        ac = a * a - (a * a).mean(axis=0)
        bc = b * b - (b * b).mean(axis=0)
        if kind == "covariance":
            val = 0.5 * (np.sum(ac * ac) / len(a) + np.sum(bc * bc) / len(b))
            grad = 2 * (left.T @ (a * ac) / len(a) + right.T @ (b * bc) / len(b)) if gradient else None
        else:
            val = np.sum(ac * bc) / len(a)
            grad = 2 * (left.T @ (a * bc) + right.T @ (b * ac)) / len(a) if gradient else None
        return float(val), grad

    r = initial
    # Identity is a stationary point for some symmetric loading patterns.
    history = []
    step = 1.0
    status = "iteration_limit"
    for _ in range(max_iter):
        value, g = objective(r, True)
        history.append(value)
        t = r.T @ g
        direction = r @ ((t - t.T) * 0.5)
        size = np.linalg.norm(direction)
        if size <= tol * max(abs(value), 1e-12):
            status = "stationary"
            break
        d = direction / size
        for _ in range(32):
            cand, _ = polar(r + step * d)
            new, _ = objective(cand, False)
            if new >= value + 1e-4 * step * size:
                break
            step *= 0.5
        else:
            status = "line_search_stopped"
            break
        r = cand
        if new - value <= tol * max(abs(value), 1e-12):
            status = "objective_tolerance"
            break
        step = min(2 * step, rank**0.5)
    return r, {
        "status": status,
        "iterations": len(history),
        "initial": history[0],
        "final": objective(r, False)[0],
    }
