"""Feature selection, linear concept probes, and removal readouts."""

import json
from pathlib import Path

import numpy as np

from mm_sae.analysis.evaluation import auroc, auc_interval
from mm_sae.analysis.linear import LinearScore, fit_logistic, identity_score
from mm_sae.analysis.selection import best_single, matched_random, removal_ranking
from .context import get_context


def select_job(job):
    side, cid = job
    ctx = get_context()
    path = ctx.file("selection", f"{side}_{cid}")
    if path.exists():
        return str(path)
    prepared = ctx.o["selection"].get("prepared_directory")
    if prepared:
        record = json.loads((Path(prepared) / f"{side}_{cid}.json").read_text())
        if record["metric"] != ctx.o["selection"]["metric"]:
            raise ValueError("Prepared ranking metric differs from configured selector")
        ctx.save("selection", f"{side}_{cid}", record)
        return str(path)
    data = ctx.data
    rows, changed = data.train.removed(side, cid)
    keep = data.fit[side][rows]
    ranking, auc = removal_ranking(data.train.activations[side][rows[keep]],
                                  changed[keep], ctx.alive[side])
    ctx.save("selection", f"{side}_{cid}", dict(**ctx.metadata(side, cid),
        ranking=ranking, auroc=auc[ranking], fit_pairs=int(keep.sum())))
    return str(path)


def evaluate(model, sample, ctx, intervals=True):
    x, y, groups = sample
    if model is None:
        return dict(auroc=None, status="no_two_classes_in_fit_or_tune")
    score = model.predict(x)
    stats = auc_interval(y, score, groups, ctx.repeats if intervals else 0, ctx.seed, return_samples=True)
    point = stats["auroc"]
    assert point is not None
    return dict(**stats, observations=len(y), positives=int(np.sum(y)),
                images=len(np.unique(groups)), status="ok" if np.isfinite(point) else "one_class")


def concept_job(job):
    side, cid = job
    ctx = get_context()
    path = ctx.file("concepts", f"{side}_{cid}")
    if path.exists():
        return str(path)
    removal = ctx.removal_data(side, cid)
    originals = [(split.activations[side][mask], split.labels(side, cid)[mask],
                  split.groups[side][mask]) for split, mask in ctx.data.samples(side)]
    train_x, train_y, _ = originals[0]
    tune_x, tune_y, _ = originals[1]
    mean, scale = ctx.mean[side], ctx.scale[side]
    records, models, probes = [], {}, {}
    single = best_single(train_x, train_y, tune_x, tune_y, ctx.alive[side])
    if single:
        fid, sign = single
        model = identity_score(fid, mean, scale)
        model.weight *= sign
        records.append(dict(task="concept_presence", condition="best_single", n=1,
                            **evaluate(model, originals[2], ctx), **ctx.metadata(side, cid)))
        probes["best_single"] = model.record()
    for n in ctx.counts:
        chosen = ctx.features(side, cid, n)
        if len(chosen) < n:
            records.append(dict(task="removal", condition="learned", n=n, status="too_few_candidates",
                                **ctx.metadata(side, cid)))
            continue
        variants = [("top_n", chosen)]
        if n > 1:
            for seed in ctx.o["controls"]["random_seeds"]:
                random = matched_random(chosen, ctx.alive[side], ctx.firing[side], seed,
                                        ctx.o["controls"]["random_nearest_candidates"])
                if random is not None:
                    variants.append((f"random_{seed}", random))
        for name, features in variants:
            condition = "learned" if name == "top_n" else name
            model = (identity_score(int(features[0]), mean, scale) if n == 1 else fit_logistic(
                removal[0][0], removal[0][1], removal[1][0], removal[1][1], features,
                mean, scale, ctx.penalties, nonnegative=True))
            models[f"{condition}_{n}"] = model.record() if model else None
            for task, sample in [("removal", removal[2]), ("removal_score_presence", originals[2])]:
                records.append(dict(task=task, condition=condition, n=n,
                                    **evaluate(model, sample, ctx, intervals=name == "top_n"),
                                    **ctx.metadata(side, cid)))
            if model:
                records[-2]["tune_auroc"] = auroc(removal[1][1], model.predict(removal[1][0]))
                records[-2]["effective_features"] = int(np.sum(model.weight != 0))
            probe = fit_logistic(train_x, train_y, tune_x, tune_y, features, mean, scale, ctx.penalties)
            probes[f"{name}_{n}"] = probe.record() if probe else None
            records.append(dict(task="concept_presence", condition=name, n=n,
                                **evaluate(probe, originals[2], ctx, intervals=name == "top_n"),
                                **ctx.metadata(side, cid)))
        equal = LinearScore(chosen, mean[chosen], scale[chosen], np.ones(n)/n, 0., 0.)
        models[f"equal_{n}"] = equal.record()
        for task, sample in [("removal", removal[2]), ("removal_score_presence", originals[2])]:
            records.append(dict(task=task, condition="equal", n=n,
                                **evaluate(equal, sample, ctx, intervals=False), **ctx.metadata(side, cid)))
    add_paired_changes(records, "n1_same_condition")
    ctx.save("concepts", f"{side}_{cid}", dict(records=records, models=models, probes=probes))
    return str(path)


def pair_job(job):
    side, a, b = job
    ctx = get_context()
    path = ctx.file("shared_pairs", f"{side}_{a}_{b}")
    if path.exists():
        return str(path)
    samples = []
    for split, mask in ctx.data.samples(side):
        aa, bb = split.labels(side, a), split.labels(side, b)
        keep = mask & (aa != bb)
        samples.append((split.activations[side][keep], aa[keep], split.groups[side][keep]))
    meta = dict(side=side, category_a=a, category_b=b, name_a=ctx.data.names[a],
                name_b=ctx.data.names[b])
    records, models = [], {}
    shared = ctx.features(side, a, 1)
    conditions = [("shared_one", 1, shared)]
    for n in ctx.counts:
        union = np.union1d(ctx.features(side, a, n), ctx.features(side, b, n))
        conditions.append(("union", n, union))
        best = best_single(samples[0][0], samples[0][1], samples[1][0], samples[1][1], union)
        if best:
            model = identity_score(best[0], ctx.mean[side], ctx.scale[side])
            model.weight *= best[1]
            records.append(dict(**meta, condition="best_in_union", n=n, coordinates=1,
                                **evaluate(model, samples[2], ctx)))
    best = best_single(samples[0][0], samples[0][1], samples[1][0], samples[1][1], ctx.alive[side])
    if best:
        model = identity_score(best[0], ctx.mean[side], ctx.scale[side])
        model.weight *= best[1]
        records.append(dict(**meta, condition="best_all", n=1, coordinates=1,
                            **evaluate(model, samples[2], ctx)))
    for condition, n, features in conditions:
        model = fit_logistic(samples[0][0], samples[0][1], samples[1][0], samples[1][1],
                             features, ctx.mean[side], ctx.scale[side], ctx.penalties)
        models[f"{condition}_{n}"] = model.record() if model else None
        records.append(dict(**meta, condition=condition, n=n, coordinates=len(features),
                            **evaluate(model, samples[2], ctx)))
    add_paired_changes(records, "best_all")
    ctx.save("shared_pairs", f"{side}_{a}_{b}", dict(records=records, models=models))
    return str(path)


def add_paired_changes(records, reference):
    for record in records:
        baseline = next((r for r in records if (
            r.get("task") == record.get("task") and r["condition"] == record["condition"] and r["n"] == 1
            if reference == "n1_same_condition" else r["condition"] == reference)), None)
        if baseline is None or record.get("auroc") is None or baseline.get("auroc") is None:
            continue
        record["auroc_change_reference"] = reference
        record["auroc_change"] = record["auroc"]-baseline["auroc"]
        a, b = record.get("_bootstrap", []), baseline.get("_bootstrap", [])
        if len(a) and len(a) == len(b):
            low, high = np.quantile(np.asarray(a)-b, [.025, .975])
            record["auroc_change_ci_low"], record["auroc_change_ci_high"] = float(low), float(high)
    for record in records:
        record.pop("_bootstrap", None)
