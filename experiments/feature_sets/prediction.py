"""Original-activation prediction with fixed target coordinates."""

import numpy as np
from safetensors.numpy import load_file
from scipy import sparse

from mm_sae.analysis.evaluation import error_interval
from mm_sae.analysis.regression import CrossMoments, fit_fixed_inputs, predict
from mm_sae.analysis.selection import matched_random
from .context import get_context


def prediction_metrics(prediction, target, groups, target_scale, repeats, seed, decoder=None, weights=None):
    target = target.toarray() if sparse.issparse(target) else np.asarray(target)
    error = prediction-target
    if not np.all(np.isfinite(error)):
        raise FloatingPointError("Nonfinite regression predictions")
    w = np.ones(len(target)) if weights is None else np.asarray(weights)
    mean = np.average(target, axis=0, weights=w)
    variance = np.average((target-mean)**2, axis=0, weights=w)
    mse = np.average(error**2, axis=0, weights=w)
    valid = variance > 1e-14
    r2 = np.full(len(variance), np.nan)
    r2[valid] = 1-mse[valid]/variance[valid]
    normalized_rows = np.mean((error/target_scale)**2, axis=1)
    interval = error_interval(normalized_rows, groups, w, repeats, seed)
    result = dict(r2_mean=float(np.mean(r2[valid])) if valid.any() else None,
                  r2_valid_targets=int(valid.sum()), targets=len(valid), observations=len(target),
                  images=len(np.unique(groups)), normalized_mse=interval["mse"],
                  normalized_mse_ci_low=interval["mse_ci_low"],
                  normalized_mse_ci_high=interval["mse_ci_high"])
    if decoder is not None:
        gram = decoder @ decoder.T
        row_error = np.sum((error @ gram)*error, axis=1)
        centered = target-mean
        reference = np.average(np.sum((centered @ gram)*centered, axis=1), weights=w)
        result["decoder_contribution_mse"] = float(np.average(row_error, weights=w))
        result["decoder_contribution_relative_mse"] = (
            result["decoder_contribution_mse"]/reference if reference > 0 else None)
    return result, r2, mse


def cooccurring(ctx, cid):
    fit = ctx.data.fit["image"]
    p = ctx.data.train.presence[fit]
    col = ctx.data.train.columns[cid]
    a = np.asarray(p[:, col], float)
    candidates = [c for c in ctx.ids if c != cid and len(ctx.features("image", c, 1))]
    means = np.asarray(p.mean(0)).ravel()
    joint = a @ np.asarray(p, float)/len(a)
    denom = np.sqrt(a.var()*means*(1-means))
    rho = np.divide(joint-a.mean()*means, denom, out=np.zeros_like(means), where=denom > 0)
    candidates = [c for c in candidates if rho[ctx.data.train.columns[c]] > 0]
    return max(candidates, key=lambda c: rho[ctx.data.train.columns[c]]) if candidates else None


def raw_job(job):
    side, cid = job
    other = "text" if side == "image" else "image"
    ctx = get_context()
    path = ctx.file("raw_prediction", f"{side}_{cid}")
    if path.exists():
        return str(path)
    target_ids = ctx.features(other, cid, ctx.o["original_rq2"]["target_set_size"])
    if not len(target_ids):
        ctx.save("raw_prediction", f"{side}_{cid}", dict(records=[], status="no_target"))
        return str(path)
    decoder = load_file(ctx.data.source / "models" / other / "model.safetensors")["W_dec"][target_ids]
    sy = ctx.scale[other][target_ids]
    top = ctx.features(side, cid, max(ctx.counts))
    variants = [("top_n", top)]
    co = cooccurring(ctx, cid)
    if co is not None:
        variants.append(("cooccurring", ctx.features(side, co, max(ctx.counts))))
    for seed in ctx.o["controls"]["random_seeds"]:
        random = matched_random(top, ctx.alive[side], ctx.firing[side], seed,
                                ctx.o["controls"]["random_nearest_candidates"], keep_first=False)
        if random is not None:
            variants.append((f"random_{seed}", random))
    records, models = [], {}
    for condition, pool in variants:
        if not len(pool):
            continue
        parts = []
        for split, keep in ctx.data.samples("text"):
            rows = np.flatnonzero(keep)
            ix, iy = (split.parents[rows], rows) if side == "image" else (rows, split.parents[rows])
            parts.append((split.activations[side][ix][:, pool],
                          split.activations[other][iy][:, target_ids], split.parents[rows]))
        fit = CrossMoments.compute(parts[0][0], parts[0][1])
        tune = CrossMoments.compute(parts[1][0], parts[1][1])
        sx = ctx.scale[side][pool]
        baseline_errors = None
        for n in ctx.counts:
            if n > len(pool):
                continue
            coef, penalty = fit_fixed_inputs(fit, tune, sx, sy, np.arange(n), ctx.penalties)
            prediction = predict(parts[2][0], coef, fit, sx, sy)
            metrics, r2, mse = prediction_metrics(prediction, parts[2][1], parts[2][2], sy,
                ctx.repeats if condition == "top_n" else 0, ctx.seed, decoder=decoder)
            error_rows = np.mean(((prediction-parts[2][1].toarray())/sy)**2, axis=1)
            if n == 1:
                baseline_errors = error_rows
            if baseline_errors is not None and condition == "top_n":
                delta = error_interval(error_rows-baseline_errors, parts[2][2], repeats=ctx.repeats,
                                       seed=ctx.seed)
                metrics.update(normalized_mse_change_from_n1=delta["mse"],
                               normalized_mse_change_ci_low=delta["mse_ci_low"],
                               normalized_mse_change_ci_high=delta["mse_ci_high"])
            records.append(dict(**ctx.metadata(side, cid), direction=f"{side}_to_{other}",
                condition=condition, n=n, **metrics, cooccurring_category=co if condition == "cooccurring" else None))
            models[f"{condition}_{n}"] = dict(input_features=pool, target_features=target_ids,
                coefficients=coef, fit_mean_x=fit.mean_x, fit_mean_y=fit.mean_y,
                scale_x=sx, scale_y=sy, penalty=penalty, per_target_r2=r2, per_target_mse=mse)
    ctx.save("raw_prediction", f"{side}_{cid}", dict(records=records, models=models))
    return str(path)
