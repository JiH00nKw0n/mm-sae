"""Concept-coordinate selection without using cross-modal correspondence scores.

The single-coordinate probes optimize exactly the same mean logistic loss and L2
penalty as ``linear.fit_logistic``. Implicit sparse zeros are aggregated, not
sampled; fitting all coordinates in parallel does not combine their predictions.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import sparse
from typing import Any


def ranked(scores, alive, descending=True):
    ids = np.asarray(alive, int)
    ids = ids[np.isfinite(np.asarray(scores)[ids])]
    value = np.asarray(scores)[ids]
    return ids[np.lexsort((ids, -value if descending else value))]


def paired_mean_drop(original, removed):
    if original.shape != removed.shape:
        raise ValueError("Original and removed rows must be paired")
    if not original.shape[0]:
        return np.full(original.shape[1], np.nan)
    return np.asarray((original - removed).mean(axis=0)).ravel()


def probe_attribution(activations, labels, decoder, probe_direction):
    labels = np.asarray(labels, bool)
    if not labels.any() or labels.all():
        return np.full(activations.shape[1], np.nan)
    difference = np.asarray(activations[labels].mean(0)-activations[~labels].mean(0)).ravel()
    return difference * (np.asarray(decoder) @ np.asarray(probe_direction))


class SparseUnivariateLogistic:
    """Independent two-parameter logistic models, one for each sparse column."""

    def __init__(self, x, scale, columns, device="cpu"):
        self.columns = np.asarray(columns, int)
        coo = sparse.coo_matrix(x[:, self.columns])
        coo.sum_duplicates()
        self.device = device
        self.rows = torch.as_tensor(coo.row.astype(np.int64), device=device)
        self.cols = torch.as_tensor(coo.col.astype(np.int64), device=device)
        self.values = torch.as_tensor(coo.data / np.asarray(scale)[self.columns][coo.col],
                                      dtype=torch.float64, device=device)
        self.n, self.d = int(x.shape[0]), len(self.columns)

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float64, device=self.device)

    def sum_columns(self, values):
        out = torch.zeros(self.d, dtype=torch.float64, device=self.device)
        return out.scatter_add_(0, self.cols, values) / self.n

    def loss(self, w, b, y, penalty=0.):
        t = self.values * w[self.cols]
        base = torch.nn.functional.softplus(b)
        correction = torch.nn.functional.softplus(b[self.cols]+t)-base[self.cols]-y[self.rows]*t
        return base-y.mean()*b+self.sum_columns(correction)+penalty*w.square()/2

    def derivatives(self, w, b, y, penalty):
        p0 = torch.sigmoid(b)
        p = torch.sigmoid(b[self.cols]+w[self.cols]*self.values)
        v0, v = p0*(1-p0), p*(1-p)
        gw = self.sum_columns((p-y[self.rows])*self.values)+penalty*w
        gb = p0-y.mean()+self.sum_columns(p-p0[self.cols])
        hww = self.sum_columns(v*self.values.square())+penalty
        hwb = self.sum_columns(v*self.values)
        hbb = v0+self.sum_columns(v-v0[self.cols])
        return gw, gb, hww, hwb, hbb

    def fit_rank(self, y, tune, tune_y, penalties, maxiter=150, tolerance=1e-7,
                 callback=None) -> dict[str, Any]:
        if np.unique(y).size < 2 or np.unique(tune_y).size < 2:
            return dict(status="missing_fit_or_tune_class", ranking=[])
        y, ty = self.tensor(y), tune.tensor(tune_y)
        w = torch.zeros(self.d, dtype=torch.float64, device=self.device)
        b = torch.full_like(w, float(torch.logit(y.mean())))
        best = torch.full_like(w, float("inf"))
        chosen_w, chosen_b, chosen_penalty = w.clone(), b.clone(), w.clone()
        trace = []
        for penalty in sorted(penalties, reverse=True):
            if callback:
                callback(penalty)
            for iteration in range(maxiter):
                gw, gb, hww, hwb, hbb = self.derivatives(w, b, y, penalty)
                active = torch.maximum(gw.abs(), gb.abs()) > tolerance
                if not bool(active.any()):
                    break
                determinant = (hww*hbb-hwb.square()).clamp_min(1e-25)
                dw = torch.where(active, (hbb*gw-hwb*gb)/determinant, 0.)
                db = torch.where(active, (hww*gb-hwb*gw)/determinant, 0.)
                old = self.loss(w, b, y, penalty)
                step = torch.ones_like(w)
                for _ in range(45):
                    nw, nb = w-step*dw, b-step*db
                    new = self.loss(nw, nb, y, penalty)
                    accepted = new <= old-1e-4*step*(gw*dw+gb*db)+1e-14
                    if bool(accepted.all()):
                        break
                    step = torch.where(accepted, step, step/2)
                else:
                    raise RuntimeError("Single-coordinate Newton line search did not converge")
                w, b = nw, nb
            gw, gb, *_ = self.derivatives(w, b, y, penalty)
            error = float(torch.maximum(gw.abs(), gb.abs()).max())
            if error > tolerance*1.05:
                raise RuntimeError(f"Single-coordinate probes did not converge: {error}")
            loss = tune.loss(w, b, ty)
            improve = loss < best-1e-12
            best = torch.where(improve, loss, best)
            chosen_w = torch.where(improve, w, chosen_w)
            chosen_b = torch.where(improve, b, chosen_b)
            chosen_penalty = torch.where(improve, penalty, chosen_penalty)
            trace.append(dict(penalty=penalty, iterations=iteration+1, gradient_max=error))
        losses = best.cpu().numpy()
        order = np.lexsort((self.columns, losses))
        return dict(status="ok", ranking=self.columns[order], tune_logloss=losses[order],
                    coefficient=chosen_w.cpu().numpy()[order],
                    uncentered_bias=chosen_b.cpu().numpy()[order],
                    penalty=chosen_penalty.cpu().numpy()[order], optimization=trace)
