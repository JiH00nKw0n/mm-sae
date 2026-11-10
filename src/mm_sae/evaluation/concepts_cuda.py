"""Scalar logistic probes on every observation. Compress only label-independent smooth sums.
Each nonconstant bin integrates interpolation polynomials of degree 12. Bins spanning more
than one logit unit near the sigmoid transition trigger direct evaluation on all original
values in the affected bins. Final selected test losses always use the original scores directly.
"""

import numpy as np
import torch
from scipy import sparse
from mm_sae.evaluation.concepts import detection_metrics, entropy_scores, macro_detection


def as_gpu(x):
    return torch.as_tensor(x, dtype=torch.float64, device="cuda")


class EmpiricalDistribution:
    """Integrate loss and Newton derivatives over one fixed coordinate distribution."""

    def __init__(self, x, repeated=None, multiplicity=0):
        self.raw = x
        self.n = len(x) + multiplicity
        if not len(x):
            x = np.array([repeated])
            multiplicity -= 1
            self.raw = x
        n = len(x)
        self.exact_bins = {}
        self.sorted = None
        count = 192
        degree = 12
        lo, hi = x.min(), x.max()
        index = np.minimum(((x - lo) * (count / max(hi - lo, 1e-100))).astype(int), count - 1)
        lows = np.full(count, np.inf)
        highs = np.full(count, -np.inf)
        np.minimum.at(lows, index, x)
        np.maximum.at(highs, index, x)
        unique = np.unique(index)
        middle = np.zeros(count)
        radius = np.zeros(count)
        middle[unique] = (highs[unique] + lows[unique]) / 2
        radius[unique] = (highs[unique] - lows[unique]) / 2
        local = np.divide(x - middle[index], radius[index], out=np.zeros(n), where=radius[index] > 0)
        t0 = np.ones(n)
        t1 = local
        moments = [np.bincount(index, weights=v, minlength=count) for v in (t0, t1)]
        for _ in range(2, degree + 1):
            t0, t1 = t1, 2 * local * t1 - t0
            moments.append(np.bincount(index, weights=t1, minlength=count))
        theta = np.pi * (np.arange(degree + 1) + 0.5) / (degree + 1)
        cs = np.cos(theta[:, None] * np.arange(degree + 1)[None, :])
        cs[:, 1:] *= 2
        weights = (np.stack(moments).T @ cs.T) / (degree + 1)
        nodes = middle[:, None] + radius[:, None] * np.cos(theta)
        # Combine repeated constant nodes exactly.
        nodes0 = nodes[unique].ravel()
        w0 = weights[unique].ravel() / self.n
        if multiplicity:
            nodes0 = np.append(nodes0, repeated)
            w0 = np.append(w0, multiplicity / self.n)
        nodes1, idx = np.unique(nodes0, return_inverse=True)
        self.x = as_gpu(nodes1)[:, None]
        self.mass = as_gpu(np.bincount(idx, weights=w0))[:, None]
        self.low = as_gpu(lows[unique])[:, None]
        self.high = as_gpu(highs[unique])[:, None]
        self.bin_nodes = nodes[unique]
        self.bin_mass = weights[unique] / self.n
        self.bounds = np.stack([lows[unique], highs[unique]], axis=1)

    def calc(self, w, b, hessian):
        def calc_nodes(x, mass, w, b):
            z = x * w + b
            # use stable softplus, including its small positive tail
            loss = (mass * (torch.clamp_min(z, 0) + torch.log1p(torch.exp(-z.abs())))).sum(0)
            if not hessian:
                return loss[None, :]
            p = z.sigmoid()
            g = p * mass
            h = g * (-z).sigmoid()
            return torch.stack([loss, (g * x).sum(0), g.sum(0), (h * x * x).sum(0), (h * x).sum(0), h.sum(0)])

        values = calc_nodes(self.x, self.mass, w, b)
        a = self.low * w + b
        c = self.high * w + b
        bad = (abs(c - a) > 1) & (torch.minimum(a, c) < 38) & (torch.maximum(a, c) > -38)
        if bad.any():
            if self.sorted is None:
                self.sorted = np.sort(self.raw)
            for i in torch.nonzero(bad.any(1)).flatten().cpu().tolist():
                if i not in self.exact_bins:
                    left, right = self.bounds[i]
                    tiny = self.sorted[
                        self.sorted.searchsorted(left) : self.sorted.searchsorted(right, side="right")
                    ]
                    ux, counts = np.unique(tiny, return_counts=True)
                    self.exact_bins[i] = (
                        as_gpu(ux)[:, None],
                        as_gpu(counts / self.n)[:, None],
                        as_gpu(self.bin_nodes[i])[:, None],
                        as_gpu(self.bin_mass[i])[:, None],
                    )
                raw, mass, nodes, weight = self.exact_bins[i]
                indices = torch.nonzero(bad[i]).flatten()
                for group in indices.split(64):
                    corr = -calc_nodes(nodes, weight, w[group], b[group])
                    for start in range(0, len(raw), 16000):
                        sl = slice(start, start + 16000)
                        corr += calc_nodes(raw[sl], mass[sl], w[group], b[group])
                    values[:, group] += corr
        return values


def fit_logistic(model, rate, xy):
    """Use E[softplus(w x + b)] - w E[x y] - b E[y] for each binary label."""
    w = torch.zeros_like(rate)
    b = torch.logit(rate)
    for _ in range(80):
        f, gw, gb, ww, wb, bb = model.calc(w, b, True)
        f = f - w * xy - b * rate
        gw -= xy
        gb -= rate
        ww += 1e-10
        bb += 1e-10
        det = (ww * bb - wb * wb).clamp_min(1e-24)
        dw = (bb * gw - wb * gb) / det
        db = (ww * gb - wb * gw) / det
        scale = torch.ones_like(w)
        attempts = 0
        while True:
            nw = w - scale * dw
            nb = b - scale * db
            new = model.calc(nw, nb, False)[0] - nw * xy - nb * rate
            reject = new > f + 1e-14
            attempts += 1
            if not reject.any() or attempts == 24:
                break
            scale[reject] *= 0.5
        w = torch.where(reject, w, nw)
        b = torch.where(reject, b, nb)
        if (f - new).abs().max().item() < 1e-9:
            break
    return w, b


def _sparse_distributions(train, val, j):
    # The implicit zeros have identical logits; a single node represents all of them exactly.
    xt, xv = (m.data[m.indptr[j] : m.indptr[j + 1]].astype(float) for m in (train, val))
    n = train.shape[0]
    mean = xt.sum() / n
    sd = np.sqrt(max(0.0, (xt @ xt) / n - mean**2))
    if sd < 1e-10:
        sd = 1.0
    common = -mean / sd
    return (
        mean,
        sd,
        EmpiricalDistribution((xt - mean) / sd, common, n - len(xt)),
        EmpiricalDistribution((xv - mean) / sd, common, val.shape[0] - len(xv)),
    )


def evaluate(scores, labels, concepts, split_ids, minimum, diagnostics):
    train_idx, val_idx, test_idx = (split_ids[name] for name in ("train", "val", "test"))
    labels = sparse.csr_matrix(labels, dtype=np.float64)
    counts = [np.asarray(labels[ids].sum(0)).ravel() for ids in (train_idx, val_idx, test_idx)]
    keep = (
        (counts[0] >= minimum)
        & (len(train_idx) - counts[0] >= minimum)
        & (counts[1] > 0)
        & (counts[1] < len(val_idx))
    )
    excluded = {str(c): "insufficient train or validation support" for c in np.asarray(concepts)[~keep]}
    concepts = np.asarray(concepts)[keep]
    labels = sparse.csr_matrix(labels[:, keep])
    counts = [v[keep] for v in counts]
    rates = [v / len(ix) for v, ix in zip(counts, (train_idx, val_idx, test_idx))]
    if not len(concepts):
        raise ValueError("No eligible concepts; check labels, split sizes and minimum support")
    k = len(concepts)
    rank = scores.shape[1]
    sums = []
    for ix in (train_idx, val_idx):
        m = np.zeros((k, rank))
        for i in range(0, len(ix), 8192):
            r = ix[i : i + 8192]
            product = labels[r].T @ scores[r].astype(float)
            m += product.toarray() if sparse.issparse(product) else product
        sums.append(m / len(ix))
    w = np.empty((rank, k))
    b = w.copy()
    loss = w.copy()
    split_scores = None
    if sparse.issparse(scores):
        split_scores = [sparse.csc_matrix(scores[idx]) for idx in (train_idx, val_idx, test_idx)]
        split_scores[0].sum_duplicates()
        split_scores[0].eliminate_zeros()
    train_rate, val_rate = rates[:2]
    prior_bias = np.log(train_rate) - np.log1p(-train_rate)
    prior_loss = -val_rate * np.log(train_rate) - (1 - val_rate) * np.log1p(-train_rate)
    for j in range(rank):
        if split_scores is not None:
            train = split_scores[0]
            column = train.data[train.indptr[j] : train.indptr[j + 1]]
            if len(column) == 0 or (len(column) == len(train_idx) and np.ptp(column) == 0):
                w[j], b[j], loss[j] = 0, prior_bias, prior_loss
                continue
            mean, sd, tr, va = _sparse_distributions(split_scores[0], split_scores[1], j)
        else:
            raw = scores[:, j].astype(float)
            train_column = raw[train_idx]
            if np.ptp(train_column) == 0:
                w[j], b[j], loss[j] = 0, prior_bias, prior_loss
                continue
            mean = train_column.mean()
            sd = train_column.std()
            if sd < 1e-10:
                sd = 1.0
            x = (raw - mean) / sd
            tr = EmpiricalDistribution(x[train_idx])
            va = EmpiricalDistribution(x[val_idx])
        for start in range(0, k, 128):
            sl = slice(start, start + 128)
            p, q = [as_gpu(v[sl]) for v in rates[:2]]
            a, c = fit_logistic(tr, p, as_gpu((sums[0][sl, j] - mean * rates[0][sl]) / sd))
            validation_loss = (
                va.calc(a, c, False)[0] - a * as_gpu((sums[1][sl, j] - mean * rates[1][sl]) / sd) - c * q
            )
            w[j, sl] = (a / sd).cpu().numpy()
            b[j, sl] = (c - a * mean / sd).cpu().numpy()
            loss[j, sl] = validation_loss.cpu().numpy()
    if not np.isfinite(loss).all():
        raise ValueError("Nonfinite validation loss in CUDA probe evaluation")
    best = loss.argmin(0)
    probes = {}
    test_y = sparse.csc_matrix(labels[test_idx])
    for c, (name, j) in enumerate(zip(concepts, best)):
        positives = test_y.indices[test_y.indptr[c] : test_y.indptr[c + 1]]
        raw = split_scores[2][:, j].toarray().ravel() if split_scores is not None else scores[test_idx, j]
        z = raw.astype(float) * w[j, c] + b[j, c]
        v = float((np.logaddexp(0, z).sum() - z[positives].sum()) / len(z))
        probes[str(name)] = dict(
            axis=int(j),
            weight=w[j, c],
            bias=b[j, c],
            val_loss=loss[j, c],
            test_loss=v,
            **detection_metrics(z, positives),
            positives={name: int(count[c]) for name, count in zip(("train", "val", "test"), counts)},
        )
    gain = np.maximum(0, prior_loss - loss)
    result = dict(
        macro_test_loss=float(np.mean([v["test_loss"] for v in probes.values()])),
        **macro_detection(probes),
        eligible=k,
        excluded=excluded,
        split_sizes={name: len(ids) for name, ids in split_ids.items()},
        concepts=probes,
    )
    if diagnostics:
        result["diagnostics"] = entropy_scores(gain)
    return result, dict(validation_loss=loss, importance=gain, concepts=concepts)
