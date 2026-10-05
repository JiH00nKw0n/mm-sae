"""Weighted sparse regression moments and explicit constraints on connections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment


@dataclass
class CrossMoments:
    mean_x: np.ndarray
    mean_y: np.ndarray
    xx: np.ndarray
    xy: np.ndarray
    yy: np.ndarray

    @classmethod
    def compute(cls, x, y, weights=None):
        x, y = sparse.csr_matrix(x, dtype=float), sparse.csr_matrix(y, dtype=float)
        assert x.shape is not None
        w = np.ones(x.shape[0]) if weights is None else np.asarray(weights, float)
        w = w / w.sum()
        return cls(np.asarray(x.T @ w).ravel(), np.asarray(y.T @ w).ravel(),
                   (x.T @ x.multiply(w[:, None])).toarray(),
                   (x.T @ y.multiply(w[:, None])).toarray(),
                   np.asarray(y.power(2).T @ w).ravel())

    def standardized(self, fit, sx, sy):
        xx = self.xx - np.outer(self.mean_x, fit.mean_x)-np.outer(fit.mean_x, self.mean_x)
        xx += np.outer(fit.mean_x, fit.mean_x)
        xy = self.xy-np.outer(self.mean_x, fit.mean_y)-np.outer(fit.mean_x, self.mean_y)
        xy += np.outer(fit.mean_x, fit.mean_y)
        yy = self.yy-2*self.mean_y*fit.mean_y+fit.mean_y**2
        return xx/sx[:, None]/sx[None, :], xy/sx[:, None]/sy[None, :], yy/sy**2


def mse_from_moments(coef, xx, xy, yy):
    return np.maximum(yy-2*np.sum(coef*xy, axis=0)+np.sum(coef*(xx @ coef), axis=0), 0)


def fit_fixed_inputs(fit, tune, sx, sy, selected, penalties):
    idx = np.asarray(selected, int)
    def subset(m):
        return CrossMoments(m.mean_x[idx], m.mean_y, m.xx[np.ix_(idx, idx)], m.xy[idx], m.yy)
    sf, st = subset(fit), subset(tune)
    g, h, _ = sf.standardized(sf, sx[idx], sy)
    tg, th, tv = st.standardized(sf, sx[idx], sy)
    best, best_error, best_penalty = None, float("inf"), None
    for penalty in sorted(penalties, reverse=True):
        b = np.linalg.solve(g+penalty*np.eye(len(idx)), h)
        error = mse_from_moments(b, tg, th, tv).mean()
        if error < best_error-1e-12:
            best, best_error, best_penalty = b, error, penalty
    coef = np.zeros_like(fit.xy)
    coef[idx] = best
    return coef, best_penalty


def forward_path(g, h, variance, candidates, penalty, counts):
    """Exact fit-MSE comparison after each possible one-coordinate ridge extension."""
    selected, result = [], {}
    candidates = np.asarray(candidates, int)
    valid = candidates[np.diag(g)[candidates] > 1e-14]
    for n in range(1, min(max(counts), len(valid))+1):
        remaining = np.setdiff1d(valid, selected)
        if selected:
            inv = np.linalg.inv(g[np.ix_(selected, selected)]+penalty*np.eye(len(selected)))
            base = inv @ h[selected]
            bridge = inv @ g[np.ix_(selected, remaining)]
            residual = h[remaining]-g[np.ix_(remaining, selected)] @ base
            denom = np.diag(g)[remaining]+penalty-np.sum(
                g[np.ix_(selected, remaining)]*bridge, axis=0)
            new = residual/np.maximum(denom, 1e-14)
            old = base[:, None]-bridge*new
            errors = variance-h[selected] @ base-new*residual-penalty*(np.sum(old**2, 0)+new**2)
        else:
            new = h[remaining]/(np.diag(g)[remaining]+penalty)
            errors = variance-2*new*h[remaining]+new**2*np.diag(g)[remaining]
        selected.append(int(remaining[np.argmin(errors)]))
        if n in counts:
            coef = np.linalg.solve(g[np.ix_(selected, selected)]+penalty*np.eye(n), h[selected])
            result[n] = (np.array(selected), coef)
    return result


def connection_models(fit, tune, sx, sy, candidates, counts, penalties):
    g, h, v = fit.standardized(fit, sx, sy)
    tg, th, tv = tune.standardized(fit, sx, sy)
    candidates = np.asarray(candidates, int)
    candidates = candidates[np.diag(g)[candidates] > 1e-14]
    ns, nt = h.shape
    edge_error = np.full((len(candidates), nt), np.inf)
    edge_coef = np.zeros_like(edge_error)
    for penalty in sorted(penalties, reverse=True):
        b = h[candidates]/(np.diag(g)[candidates, None]+penalty)
        errors = tv[None, :]-2*b*th[candidates]+b*b*np.diag(tg)[candidates, None]
        better = errors < edge_error-1e-12
        edge_error[better], edge_coef[better] = errors[better], b[better]
    models = {}
    one = np.zeros((ns, nt))
    if len(candidates):
        costs = np.column_stack([edge_error.T, np.repeat(tv[:, None], nt, axis=1)])
        targets, sources = linear_sum_assignment(costs)
        for t, s in zip(targets, sources, strict=True):
            if s < len(candidates) and edge_error[s, t] < tv[t]-1e-12:
                one[candidates[s], t] = edge_coef[s, t]
    models["one_to_one"] = one
    reusable = np.zeros_like(one)
    for target in range(nt):
        if not len(candidates):
            continue
        source = int(np.argmin(edge_error[:, target]))
        if edge_error[source, target] < tv[target]-1e-12:
            reusable[candidates[source], target] = edge_coef[source, target]
    models["reusable_one"] = reusable
    for n in counts:
        models[f"reusable_{n}"] = reusable.copy()
    best_errors = {n: mse_from_moments(reusable, tg, th, tv) for n in counts}
    from mm_sae.progress import iter_progress
    for target in iter_progress(range(nt), "Fit constrained multifeature regressions", unit="targets"):
        for penalty in sorted(penalties, reverse=True):
            path = forward_path(g, h[:, target], v[target], candidates, penalty, counts)
            for n, (idx, coef) in path.items():
                error = tv[target]-2*coef @ th[idx, target]+coef @ tg[np.ix_(idx, idx)] @ coef
                if error < best_errors[n][target]-1e-12:
                    best_errors[n][target] = error
                    models[f"reusable_{n}"][:, target] = 0
                    models[f"reusable_{n}"][idx, target] = coef
    return models


def predict(x, coef, fit, sx, sy):
    return (x @ (coef/sx[:, None])-(fit.mean_x/sx) @ coef)*sy+fit.mean_y
