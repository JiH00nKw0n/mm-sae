"""Feature ranks and matched controls use only the designated training rows."""

import numpy as np
from scipy import sparse

from .evaluation import sparse_binary_auroc


def removal_ranking(original, removed, alive):
    both = sparse.vstack([original, removed], format="csr")
    labels = np.r_[np.ones(original.shape[0], bool), np.zeros(removed.shape[0], bool)]
    values = sparse_binary_auroc(both, labels)
    candidates = np.asarray(alive)[np.isfinite(values[alive])]
    order = np.lexsort((candidates, -values[candidates]))
    return candidates[order], values


def best_single(train_x, train_y, tune_x, tune_y, alive):
    fit_auc = sparse_binary_auroc(train_x, train_y)
    tune_auc = sparse_binary_auroc(tune_x, tune_y)
    signs = np.where(fit_auc >= .5, 1., -1.)
    oriented = np.where(signs > 0, tune_auc, 1-tune_auc)
    keep = np.asarray(alive)[np.isfinite(oriented[alive])]
    if not len(keep):
        return None
    fid = keep[np.lexsort((keep, -oriented[keep]))[0]]
    return int(fid), float(signs[fid])


def matched_random(selected, alive, firing, seed, nearest=20, keep_first=True):
    rng = np.random.default_rng(seed)
    chosen = [int(selected[0])] if keep_first else []
    forbidden = set(map(int, selected))
    for target in selected[1:] if keep_first else selected:
        available = np.array([f for f in alive if f not in forbidden and f not in chosen])
        if not len(available):
            return None
        distance = np.abs(np.log(np.maximum(firing[available], 1e-12))-
                          np.log(max(firing[target], 1e-12)))
        pool = available[np.argsort(distance, kind="stable")[:nearest]]
        chosen.append(int(rng.choice(pool)))
    return np.asarray(chosen, int)
