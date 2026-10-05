"""Compare original SAE rankings with pairwise independence at the original category prevalences."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from mm_sae.io import atomic_json, sha256, write_csv
from mm_sae.metrics.reweighting import (
    binary_correlation, grouped_moments, independent_binary_masses, sparse_columns,
)
from mm_sae.metrics.statistics import correlation
from mm_sae.progress import ProgressReporter, iter_progress, stage_progress


def rate_summary(rows, name):
    valid = [r for r in rows if r["status"] == "included"]
    result = {"group": name, "requested_ordered_pairs": len(rows), "paired_ordered_pairs": len(valid),
              "excluded_ordered_pairs": len(rows) - len(valid)}
    if not valid:
        return result
    before = np.array([r["original_other"] > r["original_same"] for r in valid])
    after = np.array([r["independent_other"] > r["independent_same"] for r in valid])
    result.update(
        original_exceeds=int(before.sum()), independent_exceeds=int(after.sum()),
        original_percent=float(100 * before.mean()), independent_percent=float(100 * after.mean()),
        change_percentage_points=float(100 * (after.mean() - before.mean())),
        resolved=int(np.sum(before & ~after)), introduced=int(np.sum(~before & after)),
        persistent=int(np.sum(before & after)), neither=int(np.sum(~before & ~after)),
        image_categories=len({r["image_category"] for r in valid}),
    )
    for condition in ["original", "independent"]:
        same = np.array([r[f"{condition}_same"] for r in valid])
        other = np.array([r[f"{condition}_other"] for r in valid])
        result[f"{condition}_mean_same"] = float(same.mean())
        result[f"{condition}_mean_other"] = float(other.mean())
        result[f"{condition}_mean_margin"] = float((other - same).mean())
        result[f"{condition}_exact_ties"] = int(np.sum(other == same))
        result[f"{condition}_exceeds_with_1e12_tolerance"] = int(np.sum(other - same > 1e-12))
    return result


def summarize(rows, minimum_images, quality_threshold):
    positive = [r for r in rows if r["original_annotation_correlation"] > 0]
    summaries = [rate_summary(positive, "positive_annotation")]
    for group in ["object_object", "object_background", "background_object", "background_background"]:
        summaries.append(rate_summary([r for r in positive if r["category_types"] == group], group))
    for minimum in minimum_images:
        summaries.append(rate_summary([r for r in positive if r["minimum_group_images"] >= minimum],
                                      f"positive_minimum_{minimum}_images"))
    for high in [False, True]:
        selected = [r for r in positive if (r["image_auroc"] >= quality_threshold) == high]
        label = "high_image_auroc" if high else "low_image_auroc"
        summaries.append(rate_summary(selected, label))
        for minimum in minimum_images:
            summaries.append(rate_summary([r for r in selected if r["minimum_group_images"] >= minimum],
                                          f"{label}_minimum_{minimum}_images"))
    summaries.extend([
        rate_summary([r for r in positive if not r["shared_image"] and not r["shared_text"]],
                     "positive_distinct_representatives_in_both_modalities"),
        rate_summary(rows, "all_annotation_signs"),
        rate_summary([r for r in rows if r["original_annotation_correlation"] < 0], "negative_annotation"),
        rate_summary([r for r in rows if r["image_level_annotation_correlation"] > 0],
                     "positive_image_level_annotation_sensitivity"),
    ])
    anchors = []
    for category in sorted({r["image_category"] for r in positive if r["status"] == "included"}):
        selected = [r for r in positive if r["image_category"] == category and r["status"] == "included"]
        anchors.append({
            "image_category": category, "image_name": selected[0]["image_name"],
            "eligible_other_categories": len(selected),
            "original_higher_alternatives": sum(r["original_other"] > r["original_same"] for r in selected),
            "independent_higher_alternatives": sum(r["independent_other"] > r["independent_same"] for r in selected),
            "independent_condition_note": "Each rival has its own reweighted distribution and self-score",
        })
    return summaries, anchors


def compare_all_categories(rows, category_ids):
    """All rival categories compete; only positive label correlations are neutralized.

    Every targeted rival defines a separate reweighted distribution and corresponding
    self-score. The resulting category indicator is an OR over these pairwise comparisons,
    not the ranking of a single jointly reweighted correlation matrix.
    """
    comparisons = []
    for row in rows:
        adjusted = row["original_annotation_correlation"] > 0
        result = {**row, "positive_cooccurrence_adjusted": adjusted,
                  "status": row["status"] if adjusted else "included",
                  "independent_same": row["independent_same"] if adjusted else row["original_same"],
                  "independent_other": row["independent_other"] if adjusted else row["original_other"],
                  "independent_annotation_correlation": row["independent_annotation_correlation"] if adjusted else row["original_annotation_correlation"]}
        if not adjusted:
            for side in ["same", "other"]:
                key = f"original_{side}_caption_mention_rate"
                if key in row:
                    result[f"independent_{side}_caption_mention_rate"] = row[key]
        if result["status"] == "included":
            result["original_margin"] = result["original_other"] - result["original_same"]
            result["independent_margin"] = result["independent_other"] - result["independent_same"]
            result["margin_change"] = result["independent_margin"] - result["original_margin"]
        comparisons.append(result)
    summaries = []
    for label in ["all", "object_object", "object_background", "background_object", "background_background"]:
        selected = comparisons if label == "all" else [r for r in comparisons if r["category_types"] == label]
        summary = rate_summary(selected, label)
        summaries.append({k.replace("independent", "cooccurrence_removed"): v for k, v in summary.items()})
    anchors = []
    for category in category_ids:
        selected = [r for r in comparisons if r["image_category"] == category]
        before = sum(r["original_other"] > r["original_same"] for r in selected)
        after = sum(r["independent_other"] > r["independent_same"] for r in selected if r["status"] == "included")
        missing = sum(r["status"] != "included" for r in selected)
        anchors.append({
            "image_category": category, "image_name": selected[0]["image_name"],
            "image_type": selected[0]["category_types"].split("_")[0],
            "other_categories": len(selected), "uncomputed_other_categories": missing,
            "original_higher_alternatives": before, "cooccurrence_removed_higher_alternatives": after,
            "original_has_higher_alternative": before > 0,
            "cooccurrence_removed_has_higher_alternative": True if after > 0 else None if missing else False,
        })
    category_summaries = []
    for group in ["all", "object", "background"]:
        selected = anchors if group == "all" else [r for r in anchors if r["image_type"] == group]
        if not selected:
            continue
        before = sum(r["original_has_higher_alternative"] for r in selected)
        after = sum(r["cooccurrence_removed_has_higher_alternative"] is True for r in selected)
        unknown = sum(r["cooccurrence_removed_has_higher_alternative"] is None for r in selected)
        category_summaries.append({
            "group": group, "image_categories": len(selected), "original_exceeds": before,
            "cooccurrence_removed_exceeds": after, "unresolved_categories": unknown,
            "original_percent": 100 * before / len(selected),
            "cooccurrence_removed_percent": 100 * after / len(selected) if not unknown else None,
            "lost_all_higher_alternatives": sum(r["original_has_higher_alternative"] and r["cooccurrence_removed_has_higher_alternative"] is False for r in selected),
            "gained_a_higher_alternative": sum(not r["original_has_higher_alternative"] and r["cooccurrence_removed_has_higher_alternative"] is True for r in selected),
        })
    return comparisons, anchors, summaries, category_summaries


def bin_all_category_comparisons(comparisons, category_ids):
    """Bin by ORIGINAL annotation correlation; category rates condition on a rival in the bin."""
    edges = np.round(np.linspace(-1, 1, 11), 10)
    result = []
    for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        selected = [r for r in comparisons if left <= round(r["original_annotation_correlation"], 10)
                    and (round(r["original_annotation_correlation"], 10) < right
                         or i == len(edges) - 2 and round(r["original_annotation_correlation"], 10) <= right)]
        summary = rate_summary(selected, f"[{left:.1f},{right:.1f}{']' if i == 9 else ')'}")
        summary = {k.replace("independent", "cooccurrence_removed"): v for k, v in summary.items()}
        summary.update(left=float(left), right=float(right), right_inclusive=i == 9)
        eligible = sorted({r["image_category"] for r in selected if r["status"] == "included"})
        summary["categories_with_computable_rivals_in_bin"] = len(eligible)
        summary["total_image_categories"] = len(category_ids)
        for condition, name in [("original", "original"), ("independent", "cooccurrence_removed")]:
            exceeds = len({r["image_category"] for r in selected if r["status"] == "included"
                           and r[f"{condition}_other"] > r[f"{condition}_same"]})
            summary[f"{name}_categories_with_higher_rivals"] = exceeds
            summary[f"{name}_percent_of_eligible_categories"] = 100 * exceeds / len(eligible) if eligible else None
            summary[f"{name}_percent_of_all_categories"] = 100 * exceeds / len(category_ids)
        result.append(summary)
    return result


def plot_binned_comparison(bins, out):
    selected = [r for r in bins if r["left"] >= 0]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), layout="constrained")
    positions = np.arange(len(selected))
    for ax, value_suffix, denominator_key, title, ylabel in [
        (axes[0], "percent_of_eligible_categories", "categories_with_computable_rivals_in_bin",
         "At least one higher-scoring rival in the bin", "Image categories with a rival in the bin (%)"),
        (axes[1], "percent", "paired_ordered_pairs",
         "Individual rival scores exceeding self", "Ordered category pairs in the bin (%)"),
    ]:
        heights = []
        for offset, condition, label, color in [
            (-.19, "original", "Original", "#FFADAD"),
            (.19, "cooccurrence_removed", "Positive co-occurrence removed", "#9BF6FF"),
        ]:
            values = [r.get(f"{condition}_{value_suffix}") or 0 for r in selected]
            heights.extend(values)
            bars = ax.bar(positions + offset, values, width=.36, color=color, edgecolor="#56626B", linewidth=.6, label=label)
            ax.bar_label(bars, labels=[f"{v:.1f}" if r[denominator_key] else "N/A" for v, r in zip(values, selected)],
                         padding=4, fontsize=8)
        ax.set(xticks=positions,
               xticklabels=[f"{r['left']:.1f}–{r['right']:.1f}\nn = {r[denominator_key]:,}" for r in selected],
               title=title, ylabel=ylabel, xlabel="ORIGINAL annotation correlation (bins stay fixed)",
               ylim=(0, min(112, max(heights, default=0) * 1.25 + 3)))
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"All {bins[0]['total_image_categories']} categories considered; each positive-correlation pair reweighted separately", fontsize=12)
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(out / f"original_annotation_bins.{extension}", dpi=220)
    plt.close(fig)


def plot_all_categories(pair_summary, category_summary, out):
    if category_summary["unresolved_categories"] or not pair_summary["paired_ordered_pairs"]:
        return
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), layout="constrained")
    for ax, summary, denominator, title, ylabel in [
        (axes[0], category_summary, "image_categories", "Categories with at least one higher-scoring rival", "Image categories (%)"),
        (axes[1], pair_summary, "paired_ordered_pairs", "Individual other-category scores exceeding self", "Ordered category pairs (%)"),
    ]:
        values = [summary["original_percent"], summary["cooccurrence_removed_percent"]]
        counts = [summary["original_exceeds"], summary["cooccurrence_removed_exceeds"]]
        bars = ax.bar([0, 1], values, width=.6, color=["#FFADAD", "#9BF6FF"], edgecolor="#56626B", linewidth=.7)
        ax.bar_label(bars, labels=[f"{v:.2f}%\n{n:,} / {summary[denominator]:,}" for v, n in zip(values, counts)],
                     padding=5, fontsize=11)
        ax.set(xticks=[0, 1], xticklabels=["Original", "Positive co-occurrence\nremoved"],
               title=title, ylabel=ylabel, ylim=(0, max(values) * 1.32 + 1))
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("All categories compared; original category prevalences retained", fontsize=13)
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(out / f"all_categories_comparison.{extension}", dpi=220)
    plt.close(fig)


def save_all_category_comparison(rows, category_ids, out):
    comparisons, anchors, pair_summaries, category_summaries = compare_all_categories(rows, category_ids)
    write_csv(out / "all_category_comparisons.csv", comparisons)
    write_csv(out / "all_category_anchors.csv", anchors)
    write_csv(out / "all_category_pair_summary.csv", pair_summaries)
    write_csv(out / "all_category_anchor_summary.csv", category_summaries)
    bins = bin_all_category_comparisons(comparisons, category_ids)
    write_csv(out / "original_annotation_bins.csv", bins)
    atomic_json(out / "all_category_summary.json", {
        "definition": "All rivals included; positive original image-label correlations set to zero; nonpositive pairs unchanged",
        "scope": "OR over separately reweighted rival comparisons, not one global post-intervention matrix",
        "category_summaries": category_summaries, "pair_summaries": pair_summaries,
        "original_annotation_bins": bins,
    })
    plot_all_categories(pair_summaries[0], category_summaries[0], out)
    plot_binned_comparison(bins, out)
    return category_summaries[0], pair_summaries[0]


def report_saved_comparisons(out):
    """Update summaries from saved scores without repeating activation statistics."""
    strings = {"image_name", "other_name", "category_types", "status"}
    integers = {"pair_index", "image_category", "other_category", "image_feature", "same_text_feature",
                "other_text_feature", "minimum_group_images"}
    with (out / "scores.csv").open() as stream:
        rows = [{key: None if value == "" else value if key in strings else
                 value == "True" if value in {"True", "False"} else int(value) if key in integers else float(value)
                 for key, value in row.items()} for row in csv.DictReader(stream)]
    manifest = json.loads((out / "manifest.json").read_text())
    result = save_all_category_comparison(rows, manifest["category_ids"], out)
    atomic_json(out / "report_manifest.json", {
        "requested_comparison": "All categories; only positive co-occurrence correlations set to zero",
        "binning": "Original caption-weighted image-label Pearson correlation, width 0.2; bins fixed before adjustment",
        "score_source_sha256": sha256(out / "scores.csv"),
        "run_manifest_sha256": sha256(out / "manifest.json"),
        "report_script_sha256": sha256(Path(__file__)),
    })
    print(json.dumps({"categories": result[0], "pairs": result[1]}, indent=2), flush=True)


def plot_comparison(summaries, out):
    names = {"positive_annotation": "All", "object_object": "Object / object",
             "object_background": "Object / background", "background_object": "Background / object",
             "background_background": "Background / background"}
    selected = [r for r in summaries if r["group"] in names and r["paired_ordered_pairs"]]
    if not selected:
        return
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(11, 5.1), layout="constrained")
    positions = np.arange(len(selected))
    for offset, condition, label, color in [
        (-.19, "original", "Original distribution", "#FFADAD"),
        (.19, "independent", "Zero annotation correlation; original prevalences", "#9BF6FF"),
    ]:
        bars = ax.bar(positions + offset, [r[f"{condition}_percent"] for r in selected], width=.36,
                      color=color, edgecolor="#56626B", linewidth=.7, label=label)
        ax.bar_label(bars, labels=[f"{r[f'{condition}_percent']:.2f}%" for r in selected], padding=4, fontsize=9)
    ax.set(xticks=positions, xticklabels=[f"{names[r['group']]}\nn = {r['paired_ordered_pairs']:,}" for r in selected],
           ylabel="Other-category score > same-category score (%)",
           title="Same category pairs, fixed SAE features, full original image-caption data")
    ax.set_ylim(0, max(r[f"{c}_percent"] for r in selected for c in ["original", "independent"]) * 1.3 + 1)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_xlabel("Image category / text category; pairs with positive original annotation correlation")
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(out / f"natural_vs_independent.{extension}", dpi=220)
    plt.close(fig)


def run(args):
    run_root, out = args.run.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with ProgressReporter(out, ["load", "compare", "report"], [], interval=10):
        with stage_progress("load"):
            index = run_root / "index" / args.split
            ids = json.loads((index / "concept_ids.json").read_text())
            requested = ids if args.concept_ids is None else args.concept_ids
            if len(set(requested)) != len(requested) or not set(requested) <= set(ids):
                raise ValueError("Choose unique category IDs present in the index")
            positions = [ids.index(c) for c in requested]
            ids = requested
            names = {c["id"]: c["name"] for c in json.loads((run_root / "dataset.json").read_text())["concepts"]}
            reps = json.loads((run_root / "representatives.json").read_text())
            image_ids = [reps[str(c)]["image"] for c in ids]
            text_ids = [reps[str(c)]["text"] for c in ids]
            if any(f is None for f in image_ids + text_ids):
                raise ValueError("Requested categories must have fixed representatives")
            with (run_root / "representatives.csv").open() as stream:
                auc = {(int(r["concept_id"]), r["side"]): float(r["auroc"])
                       for r in csv.DictReader(stream) if r["split"] == args.selection_split}
            parents = np.load(index / "parents.npy")
            presence = np.load(index / "presence.npy")[:, positions].T.copy()
            n_captions = np.bincount(parents, minlength=presence.shape[1])
            if np.any(n_captions == 0):
                raise ValueError("Every cached image must have at least one caption")
            mentions = np.load(index / "mentions.npy")[:, positions]
            mention_rows = [np.flatnonzero(mentions[:, i]) for i in range(len(ids))]
            del mentions
            image = sparse.load_npz(run_root / f"activations/{args.split}/image.npz")[:, image_ids][parents]
            text = sparse.load_npz(run_root / f"activations/{args.split}/text.npz")[:, text_ids]
            full = correlation(image, text)
            x, y = sparse_columns(image), sparse_columns(text)
            del image, text
            pairs = list(itertools.combinations(range(len(ids)), 2))
            sources = [run_root / "representatives.json", run_root / "representatives.csv", run_root / "dataset.json",
                       *[index / f for f in ["parents.npy", "presence.npy", "mentions.npy", "concept_ids.json"]],
                       *[run_root / f"activations/{args.split}/{s}.npz" for s in ["image", "text"]]]
            manifest = {
                "source_run": str(run_root), "split": args.split, "selection_split": args.selection_split,
                "category_ids": ids, "images": presence.shape[1], "captions": len(parents),
                "observation_unit": "One original image-caption pair; each caption equal weight at baseline",
                "annotation": "Model-visible image category presence, repeated for each caption",
                "caption_agreement_filter": False, "group_order": ["00", "01", "10", "11"],
                "target": "Product of the two empirical binary image-label marginals; no prevalence balancing",
                "primary_pair_population": "All ordered pairs of distinct categories; only positive label correlations are neutralized",
                "paired_comparison": "Same pair population and observations before and after reweighting",
                "missing_group_policy": "Exclude from BOTH comparison conditions if a positive target mass lacks observations",
                "caption_mentions": "Used only for diagnostics; mention prevalences are not constrained",
                "quality_threshold_descriptive_only": args.quality_threshold,
                "minimum_group_images_sensitivity_only": args.minimum_images,
                "source_sha256": {str(p): sha256(p) for p in sources},
                "script_sha256": sha256(Path(__file__)),
                "moments_code_sha256": sha256(Path(__file__).resolve().parents[1] / "src/mm_sae/metrics/reweighting.py"),
            }
            atomic_json(out / "manifest.json", manifest)
        support_rows, rows = [], []
        moments_counts = np.zeros((len(pairs), 4), dtype=np.int64)
        moments_sums = np.full((len(pairs), 4, 4), np.nan)
        moments_products = np.full((len(pairs), 4, 4, 4), np.nan)
        max_baseline_error = max_marginal_error = max_target_correlation = 0.
        with stage_progress("compare"):
            for pair_index, (i, j) in enumerate(iter_progress(pairs, "Original vs independence", unit="pairs")):
                image_groups = 2 * presence[i].astype(np.int8) + presence[j]
                image_counts = np.bincount(image_groups, minlength=4)
                counts = np.bincount(image_groups, weights=n_captions, minlength=4).astype(np.int64)
                image_squared = np.bincount(image_groups, weights=n_captions.astype(float)**2, minlength=4)
                natural = counts / counts.sum()
                target = independent_binary_masses(counts)
                phi = binary_correlation(natural)
                phi_images = binary_correlation(image_counts / image_counts.sum())
                target_phi = binary_correlation(target)
                marginal_error = max(abs((natural[2] + natural[3]) - (target[2] + target[3])),
                                     abs((natural[1] + natural[3]) - (target[1] + target[3])))
                max_marginal_error = max(max_marginal_error, marginal_error)
                if np.isfinite(target_phi):
                    max_target_correlation = max(max_target_correlation, abs(target_phi))
                status = "included" if np.all(counts > 0) else "missing_group"
                support = {"pair_index": pair_index, "category_a": ids[i], "category_b": ids[j],
                           "name_a": names[ids[i]], "name_b": names[ids[j]], "status": status,
                           "original_annotation_correlation": phi, "image_level_annotation_correlation": phi_images,
                           "independent_annotation_correlation": target_phi,
                           "original_prevalence_a": float(natural[2] + natural[3]),
                           "original_prevalence_b": float(natural[1] + natural[3]),
                           "minimum_group_images": int(image_counts.min())}
                for g in range(4):
                    support.update({f"images_{g:02b}": int(image_counts[g]), f"captions_{g:02b}": int(counts[g]),
                                    f"original_mass_{g:02b}": float(natural[g]), f"independent_mass_{g:02b}": float(target[g])})
                support_rows.append(support)
                moments_counts[pair_index] = counts
                adjusted = None
                mention_rates = None
                if status == "included":
                    groups = image_groups[parents]
                    moments = grouped_moments([x[i], y[i], x[j], y[j]], groups, counts)
                    moments_sums[pair_index], moments_products[pair_index] = moments.sums, moments.products
                    original = moments.correlations(natural)
                    adjusted = moments.correlations(target)
                    reference = full["C"][np.ix_([i, j], [i, j])]
                    max_baseline_error = max(max_baseline_error,
                                             float(np.max(abs(original[np.ix_([0, 2], [1, 3])] - reference))))
                    target_row_weights = target / counts
                    support["effective_images_original"] = float(counts.sum()**2 / image_squared.sum())
                    support["effective_images_independent"] = float(1 / np.sum(target_row_weights**2 * image_squared))
                    support["maximum_weight_multiplier"] = float(np.max(target / natural))
                    mention_rates = [
                        (len(mention_rows[k]) / len(parents),
                         float(np.sum(target_row_weights * np.bincount(groups[mention_rows[k]], minlength=4))))
                        for k in [i, j]
                    ]
                for a, b, xi, ya, yb, direction in [(i, j, 0, 1, 3, 0), (j, i, 2, 3, 1, 1)]:
                    valid = status == "included" and adjusted is not None and bool(
                        np.isfinite(adjusted[xi, ya]) and np.isfinite(adjusted[xi, yb])
                        and full["valid_image"][a] and full["valid_text"][a] and full["valid_text"][b])
                    row = {"pair_index": pair_index, "image_category": ids[a], "other_category": ids[b],
                           "image_name": names[ids[a]], "other_name": names[ids[b]],
                           "category_types": "_".join("object" if ids[k] < args.stuff_start_id else "background" for k in [a, b]),
                           "image_feature": image_ids[a], "same_text_feature": text_ids[a], "other_text_feature": text_ids[b],
                           "shared_image": image_ids[a] == image_ids[b], "shared_text": text_ids[a] == text_ids[b],
                           "image_auroc": auc[ids[a], "image"], "same_text_auroc": auc[ids[a], "text"],
                           "other_text_auroc": auc[ids[b], "text"],
                           "minimum_group_images": int(image_counts.min()),
                           "original_annotation_correlation": phi, "image_level_annotation_correlation": phi_images,
                           "independent_annotation_correlation": target_phi,
                           "status": "included" if valid else status if status != "included" else "undefined_correlation",
                           "original_same": float(full["C"][a, a]), "original_other": float(full["C"][a, b]),
                           "independent_same": float(adjusted[xi, ya]) if valid and adjusted is not None else None,
                           "independent_other": float(adjusted[xi, yb]) if valid and adjusted is not None else None}
                    if valid:
                        row["original_margin"] = row["original_other"] - row["original_same"]
                        row["independent_margin"] = row["independent_other"] - row["independent_same"]
                        row["margin_change"] = row["independent_margin"] - row["original_margin"]
                    if mention_rates is not None:
                        row["original_same_caption_mention_rate"], row["independent_same_caption_mention_rate"] = mention_rates[direction]
                        row["original_other_caption_mention_rate"], row["independent_other_caption_mention_rate"] = mention_rates[1 - direction]
                    rows.append(row)
        with stage_progress("report"):
            if max_baseline_error > 1e-10 or max_marginal_error > 1e-12 or max_target_correlation > 1e-12:
                raise AssertionError("Numerical verification of original scores or independent label distribution failed")
            np.savez_compressed(out / "grouped_moments.npz", pairs=np.array(pairs), category_ids=ids,
                                counts=moments_counts, sums=moments_sums, products=moments_products)
            write_csv(out / "pair_support.csv", support_rows)
            write_csv(out / "scores.csv", rows)
            summaries, anchors = summarize(rows, args.minimum_images, args.quality_threshold)
            write_csv(out / "summary.csv", summaries)
            write_csv(out / "per_image_category.csv", anchors)
            verification = {"maximum_original_score_error": max_baseline_error,
                            "maximum_prevalence_error": max_marginal_error,
                            "maximum_absolute_target_label_correlation": max_target_correlation,
                            "all_input_hashes_unchanged": all(sha256(Path(p)) == h for p, h in manifest["source_sha256"].items())}
            if not verification["all_input_hashes_unchanged"]:
                raise AssertionError("Input source changed during analysis")
            atomic_json(out / "summary.json", {"results": summaries, "verification": verification,
                                               "total_ordered_pairs": len(rows)})
            category_main, pair_main = save_all_category_comparison(rows, ids, out)
            plot_comparison(summaries, out)
            print(json.dumps({"categories": category_main, "all_pairs": pair_main,
                              "positive_pairs_diagnostic": summaries[0], "verification": verification}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", default="train2017")
    parser.add_argument("--selection-split", default="train2017")
    parser.add_argument("--concept-ids", type=int, nargs="+")
    parser.add_argument("--stuff-start-id", type=int, default=91)
    parser.add_argument("--quality-threshold", type=float, default=.75)
    parser.add_argument("--minimum-images", type=int, nargs="+", default=[10, 50, 100, 200])
    parser.add_argument("--report-only", action="store_true", help="Regenerate all-category summaries from saved scores")
    args = parser.parse_args()
    if args.report_only:
        report_saved_comparisons(args.out.resolve())
    else:
        run(args)


if __name__ == "__main__":
    main()
