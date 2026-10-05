"""Shared, non-duplicated latent coordinates under three connection constraints."""

import numpy as np
from scipy import sparse

from mm_sae.analysis.data import save_json
from mm_sae.analysis.evaluation import error_interval
from mm_sae.analysis.regression import CrossMoments, connection_models, fit_fixed_inputs, predict
from mm_sae.analysis.selection import matched_random
from mm_sae.progress import iter_progress
from .prediction import prediction_metrics


def build_intervention_split(ctx, split, root):
    dest = root / split.name
    if (dest / "complete.json").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    images, texts, groups, categories, weights = [], [], [], [], []
    for cid in iter_progress(ctx.ids, f"Align removal changes {split.name}", unit="categories"):
        ir, im = split.removed("image", cid)
        tr, tm = split.removed("text", cid)
        keep = split.presence[split.parents[tr], split.columns[cid]]
        tr, tm = tr[keep], tm[keep]
        if not len(tr):
            continue
        parents = split.parents[tr]
        positions = np.searchsorted(ir, parents)
        assert np.array_equal(ir[positions], parents)
        di = split.activations["image"][ir]-im
        dt = split.activations["text"][tr]-tm
        images.append(di[positions][:, ctx.alive["image"]])
        texts.append(dt[:, ctx.alive["text"]])
        groups.append(parents)
        categories.append(np.full(len(tr), cid))
        counts = np.bincount(parents, minlength=len(split.image_ids))
        weights.append(1/counts[parents]/np.sum(counts > 0))
    if not groups:
        raise ValueError("No category-present-and-mentioned intervention pairs")
    sparse.save_npz(dest / "image.npz", sparse.vstack(images, format="csr"))
    sparse.save_npz(dest / "text.npz", sparse.vstack(texts, format="csr"))
    for name, values in [("groups", groups), ("categories", categories), ("weights", weights)]:
        np.save(dest / f"{name}.npy", np.concatenate(values))
    save_json(dest / "complete.json", dict(rows=sum(map(len, groups))))


def shuffle_image_groups(groups, categories, seed):
    """Map each image to another image of the same concept, moving all its captions together."""
    rng = np.random.default_rng(seed)
    index = np.arange(len(groups))
    for cid in np.unique(categories):
        rows = np.flatnonzero(categories == cid)
        unique, first, inverse = np.unique(groups[rows], return_index=True, return_inverse=True)
        if len(unique) < 2:
            continue
        order = rng.permutation(len(unique))
        mapped = np.empty(len(unique), int)
        mapped[order] = np.roll(order, 1)
        index[rows] = rows[first[mapped[inverse]]]
    return index


def intervention_analysis(ctx):
    root = ctx.out / "intervention"
    root.mkdir(exist_ok=True)
    for split in [ctx.data.train, ctx.data.test]:
        build_intervention_split(ctx, split, root)
    pools = {s: np.unique(np.concatenate([ctx.features(s, c, max(ctx.counts)) for c in ctx.ids]))
             for s in ["image", "text"]}
    for side in ["image", "text"]:
        other = "text" if side == "image" else "image"
        target_positions = np.searchsorted(ctx.alive[other], pools[other])
        source_positions = np.searchsorted(ctx.alive[side], pools[side])
        sx, sy = ctx.scale[side][ctx.alive[side]], ctx.scale[other][pools[other]]
        for condition in ["aligned", "shuffled"]:
            dest = root / f"{side}_{condition}.json"
            if dest.exists():
                continue
            samples = []
            for partition in ["fit", "tune", "test"]:
                split = ctx.data.test if partition == "test" else ctx.data.train
                folder = root / split.name
                groups = np.load(folder / "groups.npy")
                categories = np.load(folder / "categories.npy")
                if partition == "test":
                    keep = np.ones(len(groups), bool)
                else:
                    keep = ctx.data.tune_images[groups]
                    if partition == "fit":
                        keep = ~keep
                gs, cs = groups[keep], categories[keep]
                # Recompute weights within each partition so every category sums to one.
                w = np.zeros(len(gs))
                for cid in np.unique(cs):
                    rows = np.flatnonzero(cs == cid)
                    _, inverse, counts = np.unique(gs[rows], return_inverse=True, return_counts=True)
                    w[rows] = 1/(len(counts)*counts[inverse])
                xi = sparse.load_npz(folder / "image.npz")[keep]
                xt = sparse.load_npz(folder / "text.npz")[keep]
                if condition == "shuffled":
                    xi = xi[shuffle_image_groups(gs, cs, ctx.seed)]
                x, y = (xi, xt) if side == "image" else (xt, xi)
                samples.append((x, y[:, target_positions], gs, w))
            fit = CrossMoments.compute(samples[0][0], samples[0][1], samples[0][3])
            tune = CrossMoments.compute(samples[1][0], samples[1][1], samples[1][3])
            from mm_sae.progress import progress_task
            with progress_task(f"Select {side} {condition} connections", unit="targets"):
                models = connection_models(fit, tune, sx, sy, source_positions, ctx.counts, ctx.penalties)
            records, coefficients = [], {}
            reference_errors = {}
            common = np.all(np.array([np.any(b != 0, axis=0) for b in models.values()]), axis=0)
            for name, coef in iter_progress(list(models.items()),
                                            f"Evaluate {side} {condition} connections", unit="models"):
                prediction = predict(samples[2][0], coef, fit, sx, sy)
                metrics, r2, mse = prediction_metrics(prediction, samples[2][1], samples[2][2], sy,
                    ctx.repeats, ctx.seed, weights=samples[2][3])
                errors = np.mean(((prediction-samples[2][1].toarray())/sy)**2, axis=1)
                if name in {"one_to_one", "reusable_one"}:
                    reference_errors[name] = errors
                for reference, baseline in reference_errors.items():
                    delta = error_interval(errors-baseline, samples[2][2], weights=samples[2][3],
                                           repeats=ctx.repeats, seed=ctx.seed)
                    metrics[f"mse_change_from_{reference}"] = delta["mse"]
                    metrics[f"mse_change_from_{reference}_ci_low"] = delta["mse_ci_low"]
                    metrics[f"mse_change_from_{reference}_ci_high"] = delta["mse_ci_high"]
                nonzero = np.sum(coef != 0, axis=0)
                records.append(dict(direction=f"{side}_to_{other}", condition=condition, model=name,
                    **metrics, linked_targets=int(np.sum(nonzero > 0)),
                    common_linked_targets=int(common.sum()),
                    common_linked_r2=float(np.nanmean(r2[common])) if common.any() else None,
                    total_connections=int(nonzero.sum()), per_target_r2=r2, per_target_mse=mse))
                coefficients[name] = coef
            # Randomize selected source identities per target; refit coefficients using the same moments.
            for seed in ctx.o["controls"]["random_seeds"] if condition == "aligned" else []:
                for n in ctx.counts:
                    reference = models[f"reusable_{n}"]
                    random_coef = np.zeros_like(reference)
                    for target in range(reference.shape[1]):
                        chosen = np.flatnonzero(reference[:, target])
                        if not len(chosen):
                            continue
                        replacement = matched_random(ctx.alive[side][chosen], ctx.alive[side],
                            ctx.firing[side], seed+target, ctx.o["controls"]["random_nearest_candidates"],
                            keep_first=False)
                        if replacement is None:
                            continue
                        pos = np.searchsorted(ctx.alive[side], replacement)
                        small_fit = CrossMoments(fit.mean_x, fit.mean_y[target:target+1],
                            fit.xx, fit.xy[:, target:target+1], fit.yy[target:target+1])
                        small_tune = CrossMoments(tune.mean_x, tune.mean_y[target:target+1],
                            tune.xx, tune.xy[:, target:target+1], tune.yy[target:target+1])
                        coef, _ = fit_fixed_inputs(small_fit, small_tune, sx, sy[target:target+1],
                                                  pos, ctx.penalties)
                        random_coef[:, target] = coef[:, 0]
                    prediction = predict(samples[2][0], random_coef, fit, sx, sy)
                    metrics, _, _ = prediction_metrics(prediction, samples[2][1], samples[2][2], sy,
                        0, ctx.seed, weights=samples[2][3])
                    records.append(dict(direction=f"{side}_to_{other}", condition=f"random_{seed}",
                                        model=f"reusable_{n}", **metrics))
            np.savez_compressed(root / f"{side}_{condition}_models.npz",
                **coefficients, source_ids=ctx.alive[side], target_ids=pools[other],
                mean_x=fit.mean_x, mean_y=fit.mean_y, scale_x=sx, scale_y=sy)
            save_json(dest, dict(records=records, category_weighting="equal_category_equal_image"))
