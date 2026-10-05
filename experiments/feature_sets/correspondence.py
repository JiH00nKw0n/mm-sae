"""RQ1 over frozen concept readouts, with pair-specific independence controls."""

import csv
import itertools
import json

import numpy as np
from scipy import sparse

from mm_sae.analysis.data import save_json
from mm_sae.analysis.evaluation import distribution
from mm_sae.io import write_csv
from mm_sae.metrics.reweighting import (
    binary_correlation, grouped_moments, independent_binary_masses, sparse_columns,
)
from mm_sae.metrics.statistics import correlation
from mm_sae.progress import iter_progress


def project_scores(ctx, side, n, condition, split=None):
    split = ctx.data.test if split is None else split
    columns, coordinates, weights, invalid = [], [], [], []
    for j, cid in enumerate(ctx.ids):
        model = ctx.readout(side, cid, n, condition)
        if model is None:
            invalid.append(cid)
            continue
        coordinates.extend(model.features)
        columns.extend([j]*len(model.features))
        weights.extend(model.weight/model.scale)
    projection = sparse.csc_matrix((weights, (coordinates, columns)),
                                   shape=(len(ctx.mean[side]), len(ctx.ids)))
    # Remove each score's constant intercept: both Pearson correlation and AUROC are invariant.
    x = split.activations[side] @ projection
    if side == "image":
        x = x[split.parents]
    return x.tocsc(), invalid


def summarize_pairs(rows, edges, requested_categories):
    summaries, distributions = [], []
    for direction in ["image_to_text", "text_to_image"]:
        directed = [r for r in rows if r["direction"] == direction]
        groups = ["all", "object_object", "object_background", "background_object", "background_background"]
        groups += sorted({r.get("quality_group", "missing") for r in directed})
        for kind in groups:
            selected = directed if kind == "all" else [
                r for r in directed if r["types"] == kind or r.get("quality_group") == kind]
            for condition in ["original", "controlled"]:
                valid = [r for r in selected if r[condition+"_same"] is not None
                         and r[condition+"_other"] is not None]
                higher = [r for r in valid if r[condition+"_other"] > r[condition+"_same"]]
                anchors = {r["anchor_id"] for r in selected}
                positive = {r["anchor_id"] for r in higher}
                incomplete = {r["anchor_id"] for r in selected if
                              r[condition+"_same"] is None or r[condition+"_other"] is None}
                unresolved = incomplete-positive
                summaries.append(dict(direction=direction, group=kind, condition=condition,
                    requested_categories=len(anchors), categories_with_higher=len(positive),
                    unresolved_categories=len(unresolved),
                    category_percent=(100*len(positive)/len(anchors) if anchors and not unresolved else None),
                    category_percent_lower_bound=100*len(positive)/len(anchors) if anchors else None,
                    requested_pairs=len(selected), valid_pairs=len(valid), higher_pairs=len(higher),
                    pair_percent=100*len(higher)/len(valid) if valid else None,
                    tied_pairs=sum(r[condition+"_other"] == r[condition+"_same"] for r in valid)))
        for lower, upper in zip(edges[:-1], edges[1:]):
            selected = [r for r in directed if r["annotation_correlation"] is not None and
                        lower <= r["annotation_correlation"] and
                        (r["annotation_correlation"] < upper or upper == 1)]
            # Before/after distributions use the exact same evaluable pairs.
            paired = [r for r in selected if r["original_other"] is not None and r["controlled_other"] is not None]
            for condition in ["original", "controlled"]:
                distributions.append(dict(direction=direction, left=lower, right=upper, condition=condition,
                    requested_pairs=len(selected), **distribution([r[condition+"_other"] for r in paired])))
    return summaries, distributions


def legacy_check(ctx):
    path = ctx.out / "legacy_reproduction.json"
    if path.exists():
        return
    reps = json.loads((ctx.data.source / "representatives.json").read_text())
    ids = ctx.data.train.ids
    ii, tt = [reps[str(c)]["image"] for c in ids], [reps[str(c)]["text"] for c in ids]
    x = ctx.data.train.activations["image"][ctx.data.train.parents][:, ii]
    y = ctx.data.train.activations["text"][:, tt]
    actual = correlation(x, y)["C"]
    with np.load(ctx.data.source / "panel.npz") as archive:
        expected = archive["C"][np.ix_(ii, tt)]
    assert np.allclose(actual, expected, atol=1e-10, rtol=1e-10)
    image_count = int(np.sum(actual > np.diag(actual)[:, None])-np.sum(np.diag(actual) > np.diag(actual)))
    text_count = int(np.sum(actual.T > np.diag(actual)[:, None]))
    save_json(path, dict(categories=len(ids), directed_pairs=len(ids)*(len(ids)-1),
                        image_exceeding_pairs=image_count, text_exceeding_pairs=text_count,
                        maximum_absolute_error=float(np.max(np.abs(actual-expected)))))


def analyze(ctx, split=None, folder="correspondence", counts=None, conditions=None):
    root = ctx.out / folder
    root.mkdir(exist_ok=True)
    if ctx.o["multi_feature_rq1"]["legacy_n1_reproduction"]:
        legacy_check(ctx)
    split = ctx.data.test if split is None else split
    quality = {}
    for side in ["image", "text"]:
        for cid in ctx.ids:
            entries = json.loads(ctx.file("concepts", f"{side}_{cid}").read_text())["records"]
            value = next((r.get("tune_auroc") for r in entries if r["task"] == "removal" and
                          r["condition"] == "learned" and r["n"] == 1), None)
            quality[side, cid] = value
    presence = split.presence[split.parents][:, [split.columns[c] for c in ctx.ids]]
    annotation = correlation(sparse.csr_matrix(presence), sparse.csr_matrix(presence))
    rho = annotation["C"]
    for condition, n in itertools.product(conditions or ["learned", "equal"], counts or ctx.counts):
        mark = root / f"{condition}_{n}.json"
        if mark.exists():
            continue
        xi, missing_i = project_scores(ctx, "image", n, condition, split)
        xt, missing_t = project_scores(ctx, "text", n, condition, split)
        original = correlation(xi, xt)
        c = original["C"].copy()
        c[~original["valid_image"], :] = np.nan
        c[:, ~original["valid_text"]] = np.nan
        cols_i, cols_t = sparse_columns(xi), sparse_columns(xt)
        pairs = []
        pair_ids = list(itertools.combinations(range(len(ctx.ids)), 2))
        for a, b in iter_progress(pair_ids, f"RQ1 {condition} n={n}", unit="category pairs"):
            group = presence[:, a].astype(int)*2+presence[:, b]
            counts = np.bincount(group, minlength=4)
            image_group = (split.presence[:, split.columns[ctx.ids[a]]].astype(int)*2+
                           split.presence[:, split.columns[ctx.ids[b]]])
            image_counts = np.bincount(image_group, minlength=4)
            masses = independent_binary_masses(counts)
            valid_control = np.all(counts[masses > 0] > 0)
            adjusted = bool(rho[a, b] > 0)
            controlled = np.full((4, 4), np.nan)
            ess, maximum_weight, controlled_rho = None, None, None
            if adjusted and valid_control:
                moments = grouped_moments([cols_i[a], cols_t[a], cols_i[b], cols_t[b]], group, counts)
                controlled = moments.correlations(masses)
                weights = np.divide(masses, counts, out=np.zeros(4), where=counts > 0)
                ess = float(1/np.sum(counts*weights**2))
                maximum_weight = float(weights.max()*len(group))
                controlled_rho = binary_correlation(masses)
                assert abs(controlled_rho) < 1e-10
            for anchor, rival, ia, ta, ib, tb in [(a, b, 0, 1, 2, 3), (b, a, 2, 3, 0, 1)]:
                for direction in ["image_to_text", "text_to_image"]:
                    raw_other = c[anchor, rival] if direction == "image_to_text" else c[rival, anchor]
                    ctrl_other = controlled[ia, tb] if direction == "image_to_text" else controlled[ib, ta]
                    kinds = ["object" if ctx.ids[k] < 91 else "background" for k in [anchor, rival]]
                    q = quality["image" if direction == "image_to_text" else "text", ctx.ids[anchor]]
                    if q is None:
                        qgroup = "auroc_missing"
                    elif q < .5:
                        qgroup = "auroc_below_0.5"
                    else:
                        lower = min(.9, np.floor(q*10)/10)
                        qgroup = f"auroc_{lower:.1f}_{lower+.1:.1f}"
                    pairs.append(dict(direction=direction, anchor_id=ctx.ids[anchor], rival_id=ctx.ids[rival],
                        types="_".join(kinds), annotation_correlation=float(rho[a, b]),
                        quality_group=qgroup, n1_tune_anchor_auroc=q,
                        original_same=float(c[anchor, anchor]), original_other=float(raw_other),
                        controlled_same=float(controlled[ia, ta]) if adjusted else float(c[anchor, anchor]),
                        controlled_other=float(ctrl_other) if adjusted else float(raw_other),
                        adjusted=adjusted, control_estimable=valid_control,
                        minimum_stratum_images=int(image_counts.min()), effective_caption_rows=ess,
                        maximum_relative_caption_weight=maximum_weight, controlled_annotation_rho=controlled_rho,
                        stratum_images_00=int(image_counts[0]), stratum_images_01=int(image_counts[1]),
                        stratum_images_10=int(image_counts[2]), stratum_images_11=int(image_counts[3])))
        from mm_sae.analysis.data import clean
        pairs = clean(pairs)
        summary, distributions = summarize_pairs(pairs, ctx.o["multi_feature_rq1"]["annotation_bins"], ctx.ids)
        write_csv(root / f"{condition}_{n}_pairs.csv", pairs)
        write_csv(root / f"{condition}_{n}_summary.csv", summary)
        write_csv(root / f"{condition}_{n}_distributions.csv", distributions)
        save_json(mark, dict(condition=condition, n=n, summary=summary, distributions=distributions,
                             missing_image=missing_i, missing_text=missing_t))


def load_pairs(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))
