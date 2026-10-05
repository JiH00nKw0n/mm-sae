"""Rank statistics and image-cluster uncertainty without changing fitted models."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from typing import Any


def auroc(y, score, weights=None):
    y = np.asarray(y, bool)
    score = np.asarray(score)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    pos, neg = w[y].sum(), w[~y].sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    sorted_score, ys, ws = score[order], y[order], w[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_score))+1]
    wp = np.add.reduceat(ws*ys, starts)
    wn = np.add.reduceat(ws*~ys, starts)
    return float(np.dot(wp, np.cumsum(wn)-wn/2)/(pos*neg))


def sparse_binary_auroc(matrix, labels):
    """Exact AUROC of every nonnegative sparse column, retaining all zero ties."""
    x = matrix.tocsc(copy=True)
    x.eliminate_zeros()
    if np.any(x.data < 0):
        raise ValueError("Requires nonnegative activations")
    labels = np.asarray(labels, bool)
    np_, nn = labels.sum(), (~labels).sum()
    result = np.full(x.shape[1], np.nan)
    if not np_ or not nn:
        return result
    for j in range(x.shape[1]):
        lo, hi = x.indptr[j:j+2]
        values, pos = x.data[lo:hi], labels[x.indices[lo:hi]]
        a, b = int(pos.sum()), int((~pos).sum())
        ranks = rankdata(values, method="average")
        wins = ranks[pos].sum()-a*(a+1)/2
        result[j] = (wins+a*(nn-b)+.5*(np_-a)*(nn-b))/(np_*nn)
    return result


def auc_interval(y, score, groups, repeats=1000, seed=42, return_samples=False):
    """Multinomial image bootstrap; aggregate tied score groups before resampling."""
    point = auroc(y, score)
    if not np.isfinite(point) or not repeats:
        return dict(auroc=point, ci_low=None, ci_high=None, bootstrap_valid=0)
    y, score, groups = np.asarray(y, bool), np.asarray(score), np.asarray(groups)
    _, inverse = np.unique(groups, return_inverse=True)
    ng = inverse.max()+1
    order = np.argsort(score, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(score[order]))+1]
    yy, gg = y[order], inverse[order]
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        counts = np.bincount(rng.integers(ng, size=ng), minlength=ng)
        w = counts[gg]
        wp = np.add.reduceat(w*yy, starts)
        wn = np.add.reduceat(w*~yy, starts)
        if wp.sum() and wn.sum():
            estimates.append(float(np.dot(wp, np.cumsum(wn)-wn/2)/(wp.sum()*wn.sum())))
    low, high = np.quantile(estimates, [.025, .975]) if estimates else [np.nan, np.nan]
    result: dict[str, Any] = dict(auroc=point, ci_low=float(low), ci_high=float(high),
                                  bootstrap_valid=len(estimates))
    if return_samples:
        result["_bootstrap"] = estimates
    return result


def distribution(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(n=0)
    return dict(n=len(values), mean=float(values.mean()), std=float(values.std()),
                min=float(values.min()), q1=float(np.quantile(values, .25)),
                median=float(np.median(values)), q3=float(np.quantile(values, .75)), max=float(values.max()))


def error_interval(errors, groups, weights=None, repeats=1000, seed=42):
    errors, groups = np.asarray(errors), np.asarray(groups)
    w = np.ones(len(errors)) if weights is None else np.asarray(weights)
    _, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=w*errors)
    masses = np.bincount(inverse, weights=w)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(repeats):
        indices = rng.integers(len(sums), size=len(sums))
        boot.append(sums[indices].sum()/masses[indices].sum())
    low, high = np.quantile(boot, [.025, .975]) if boot else [np.nan, np.nan]
    return dict(mse=float(sums.sum()/masses.sum()), mse_ci_low=float(low), mse_ci_high=float(high))
