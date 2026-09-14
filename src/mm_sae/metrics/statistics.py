"""Signed Pearson correlations and exact AUROC with all zero ties retained."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.stats import rankdata


def correlation(x, y):
    if x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("Correlation requires at least two aligned observations")
    x, y = sparse.csr_matrix(x, dtype=np.float64), sparse.csr_matrix(y, dtype=np.float64)
    n = x.shape[0]
    sx, sy = np.asarray(x.sum(0)).ravel(), np.asarray(y.sum(0)).ravel()
    qx, qy = np.asarray(x.power(2).sum(0)).ravel(), np.asarray(y.power(2).sum(0)).ravel()
    vx, vy = qx - sx * sx / n, qy - sy * sy / n
    # Numerical zero detection, not a semantic firing-rate filter.
    valid_x = vx > 32 * np.finfo(float).eps * np.maximum(qx, sx * sx / n)
    valid_y = vy > 32 * np.finfo(float).eps * np.maximum(qy, sy * sy / n)
    denominator = np.sqrt(np.maximum(vx, 0)[:, None] * np.maximum(vy, 0)[None, :])
    c = (x.T @ y).toarray() - np.outer(sx, sy) / n
    valid = valid_x[:, None] & valid_y[None, :]
    np.divide(c, denominator, out=c, where=valid)
    c[~valid] = 0  # Implementation placeholder only; validity travels with every score.
    np.clip(c, -1, 1, out=c)
    return {
        "C": c,
        "valid_image": valid_x,
        "valid_text": valid_y,
        "alive_image": x.getnnz(axis=0) > 0,
        "alive_text": y.getnnz(axis=0) > 0,
        "n": n,
    }


def paired_auroc(original, removed):
    """Standard pooled AUROC. Pairing defines the examples, not the rank comparison itself."""
    if original.shape != removed.shape or not original.shape[0]:
        raise ValueError("AUROC requires equally sized, nonempty original/removal samples")
    p, q = original.tocsc(copy=True), removed.tocsc(copy=True)
    p.eliminate_zeros()
    q.eliminate_zeros()
    if np.any(p.data < 0) or np.any(q.data < 0):
        raise ValueError("Sparse zero shortcut assumes non-negative TopK activations")
    n, features = p.shape
    auc = np.empty(features, np.float64)
    for f in range(features):
        a, b = p.data[p.indptr[f] : p.indptr[f + 1]], q.data[q.indptr[f] : q.indptr[f + 1]]
        rank = rankdata(np.concatenate([a, b]), method="average")
        nz_wins = rank[: len(a)].sum() - len(a) * (len(a) + 1) / 2
        wins = nz_wins + len(a) * (n - len(b)) + 0.5 * (n - len(a)) * (n - len(b))
        auc[f] = wins / (n * n)
    return auc


def select_representative(aucs, alive):
    candidates = np.flatnonzero(alive)
    if not len(candidates):
        return None, 0
    maximum = aucs[candidates].max()
    ties = candidates[aucs[candidates] == maximum]
    return int(ties[0]), len(ties)


def bin_summary(rows, width):
    edges = np.linspace(-1, 1, round(2 / width) + 1)
    groups = [[] for _ in range(len(edges) - 1)]
    for row in rows:
        if not row["valid"]:
            continue
        x = float(row["label_correlation"])
        b = min(
            len(groups) - 1, max(0, int(np.searchsorted(np.round(edges, 10), round(x, 10), side="right") - 1))
        )
        groups[b].append(float(row["coactivation_correlation"]))
    result = []
    for i, values in enumerate(groups):
        result.append(
            {
                "left": round(float(edges[i]), 10),
                "right": round(float(edges[i + 1]), 10),
                "right_inclusive": i == len(groups) - 1,
                "n_pairs": len(values),
                "mean": float(np.mean(values)) if values else None,
                "std_population": float(np.std(values, ddof=0)) if values else None,
            }
        )
    return result
