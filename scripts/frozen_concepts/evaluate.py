"""Evaluate frozen CC3M mappings on the November 11 populations using CUDA."""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from mm_sae.evaluation.concepts import entropy_scores
from mm_sae.evaluation.concepts_cuda import fit_logistic, as_gpu
from fast_distribution import Distribution
from inputs import coefficients, source


def csr(a):
    a = a.tocsr()
    a.sort_indices()
    return torch.sparse_csr_tensor(
        torch.tensor(a.indptr, device="cuda"),
        torch.tensor(a.indices, device="cuda"),
        torch.tensor(a.data, dtype=torch.float64, device="cuda"),
        size=a.shape,
    )


def run(work, out, cfg, ds, method):
    start = time.time()
    scope = cfg["datasets"][ds]
    X, Y, terms, split = source(work, ds)
    if scope["prefix"]:
        keep = np.char.startswith(terms.astype(str), scope["prefix"])
        Y = Y[:, keep]
        terms = terms[keep]
    if (
        len(terms) != scope["concepts"]
        or hashlib.sha256(split.tobytes()).hexdigest() != scope["split_sha256"]
    ):
        raise ValueError(f"Unexpected evaluation population: {ds}")
    xs = []
    cross = []
    rates = []
    for s in ("train", "val"):
        ix = np.flatnonzero(split == s)
        xx = X[ix]
        yy = Y[ix].astype(float)
        xs.append(csr(xx))
        cross.append(as_gpu((yy.T @ xx.astype(float)).toarray() / len(ix)))
        rates.append(as_gpu(np.asarray(yy.mean(0)).ravel()))
    p, q = rates
    null = (-q * p.log() - (1 - q) * torch.log1p(-p)).cpu().numpy()
    test = np.flatnonzero(split == "test")
    xt = csr(X[test])
    yt = Y[test].tocsc()
    del X, Y, xx, yy
    B = as_gpu(coefficients(work, method, scope["side"], cfg["methods"][method][scope["side"]]))
    rank = B.shape[1]
    sums = [cs @ B for cs in cross]
    dest = out / "all-probes" / method / ds
    dest.mkdir(parents=True, exist_ok=False)
    for j0 in range(0, rank, 8):
        path = dest / f"{j0:04d}.npz"
        W = B[:, j0 : j0 + 8].contiguous()
        v0, v1 = [(a @ W).T.contiguous() for a in xs]
        mu = v0.mean(1)
        sd = torch.sqrt(((v0 - mu[:, None]) ** 2).mean(1))
        sd = torch.where(sd < 1e-10, 1.0, sd)
        u0 = (v0 - mu[:, None]) / sd[:, None]
        u1 = (v1 - mu[:, None]) / sd[:, None]
        losses = []
        weights = []
        bias = []
        active = []
        for i in range(len(W.T)):
            j = i + j0
            if v0[i].min() == v0[i].max():
                w = torch.zeros_like(p)
                b = torch.logit(p)
                vv = null
                active.append(False)
            else:
                offset = (-mu[i] / sd[i]).item()
                dt, dv = Distribution(u0[i], offset), Distribution(u1[i], offset)
                w, b = fit_logistic(dt, p, (sums[0][:, j] - mu[i] * p) / sd[i])
                vl = dv.calc(w, b, False)[0] - w * (sums[1][:, j] - mu[i] * q) / sd[i] - b * q
                vv = vl.cpu().numpy()
                active.append(True)
            losses.append(vv)
            weights.append((w / sd[i]).cpu().numpy())
            bias.append((b - w * mu[i] / sd[i]).cpu().numpy())
        loss = np.stack(losses)
        assert np.isfinite(loss).all()
        g = np.maximum(0, null - loss)
        g[~np.array(active)] = 0
        with path.with_suffix(".tmp").open("wb") as f:
            np.savez_compressed(
                f, validation_loss=loss, importance=g, weight=weights, bias=bias, concepts=terms
            )
        path.with_suffix(".tmp").replace(path)
        if j0 % 128 == 0:
            print(ds, method, j0, "/", rank, "seconds", round(time.time() - start, 1), flush=True)
    blocks = [dict(np.load(dest / f"{j:04d}.npz")) for j in range(0, rank, 8)]
    vl = np.concatenate([t["validation_loss"] for t in blocks])
    ax = vl.argmin(0)
    idx = np.arange(len(ax))
    w = np.concatenate([t["weight"] for t in blocks])[ax, idx]
    b = np.concatenate([t["bias"] for t in blocks])[ax, idx]
    best = vl[ax, idx]
    stats = entropy_scores(np.concatenate([block["importance"] for block in blocks]))
    final = out / "complete-probes" / method / ds
    final.mkdir(parents=True, exist_ok=True)
    (final / "diagnostics.json").write_text(json.dumps(stats, indent=2))
    rows = []
    for j in range(0, len(terms), 32):
        jj = slice(j, j + 32)
        z = (xt @ B[:, ax[jj]].contiguous()).cpu().numpy() * w[jj] + b[jj]
        labels = yt[:, jj].toarray().astype(bool)
        for i in range(labels.shape[1]):
            c = j + i
            y = labels[:, i]
            pred = z[:, i] >= 0
            tp = int((y & pred).sum())
            fp = int((~y & pred).sum())
            fn = int((y & ~pred).sum())
            ce = np.logaddexp(0, np.where(y, -z[:, i], z[:, i])).mean()
            pr = p[c].item()
            rows.append(
                dict(
                    method=method,
                    dataset=ds,
                    concept=terms[c],
                    axis=int(ax[c]),
                    n=len(y),
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    tn=len(y) - tp - fp - fn,
                    precision=tp / max(1, tp + fp),
                    recall=tp / (tp + fn),
                    f1=2 * tp / max(1, 2 * tp + fp + fn),
                    ce=ce,
                    prior_ce=-np.where(y, np.log(pr), np.log1p(-pr)).mean(),
                )
            )
    np.savez_compressed(final / "best.npz", concepts=terms, axis=ax, weight=w, bias=b, validation_loss=best)
    with (final / "records.tmp").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (final / "records.tmp").replace(final / "per-concept.csv")
    print("DONE", ds, method, "seconds", round(time.time() - start, 1), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work", type=Path, required=True, help="Root containing the frozen weights and evaluation packs"
    )
    parser.add_argument("--output", type=Path, required=True, help="New directory for experiment outputs")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs/report/frozen_uncapped.json",
    )
    parser.add_argument("--method", help="Evaluate only this fitted mapping")
    parser.add_argument("--dataset", help="Evaluate only this annotation dataset")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg["concept_feature_budget"] is not None:
        parser.error("This experiment uses all fitted coefficients")
    if args.method and args.method not in cfg["methods"]:
        parser.error("Expected one of " + ", ".join(cfg["methods"]))
    if args.dataset and args.dataset not in cfg["datasets"]:
        parser.error("Expected one of " + ", ".join(cfg["datasets"]))
    for ds in [args.dataset] if args.dataset else cfg["datasets"]:
        for method in [args.method] if args.method else cfg["methods"]:
            run(args.work, args.output, cfg, ds, method)


if __name__ == "__main__":
    main()
