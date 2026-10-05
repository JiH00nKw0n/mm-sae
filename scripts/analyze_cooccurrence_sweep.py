"""Reweight original image-caption pairs to test fixed SAE scores under controlled co-occurrence.

No model inference, feature selection, training, caption editing, or image masking is performed.
"""

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
from mm_sae.metrics.reweighting import balanced_binary_masses, grouped_moments, sparse_columns
from mm_sae.metrics.statistics import correlation
from mm_sae.progress import ProgressReporter, iter_progress, stage_progress


def group_support(groups: np.ndarray, parents: np.ndarray):
    """Count captions and distinct images, retaining only image/caption agreement rows.

    Parents are sorted, so all retained captions from an image are contiguous. Agreement
    guarantees that these captions have the same A/B stratum within an image.
    """
    retained = np.flatnonzero(groups >= 0)
    selected, parent = groups[retained], parents[retained]
    counts = np.bincount(selected, minlength=4)
    if not len(retained):
        return counts, np.zeros(4, dtype=int), np.zeros(4)
    starts = np.r_[0, np.flatnonzero(parent[1:] != parent[:-1]) + 1]
    captions_per_image = np.diff(np.r_[starts, len(retained)])
    image_groups = selected[starts]
    image_counts = np.bincount(image_groups, minlength=4)
    squared_counts = np.bincount(image_groups, weights=captions_per_image.astype(float)**2, minlength=4)
    return counts, image_counts, squared_counts


def effective_counts(masses, counts, image_squared_counts):
    row_weights = masses / counts
    return float(1 / np.sum(masses**2 / counts)), float(1 / np.sum(row_weights**2 * image_squared_counts))


def summarize(rows, rhos, threshold, minimum_images):
    results = []
    selectors = {
        "all": lambda r: True,
        "image_auc_below_threshold": lambda r: r["image_auroc"] < threshold,
        "image_auc_at_least_threshold": lambda r: r["image_auroc"] >= threshold,
        "both_modalities_distinct_representatives": lambda r: not r["shared_image"] and not r["shared_text"],
        "shared_image_representative": lambda r: r["shared_image"],
        "shared_text_representative": lambda r: r["shared_text"],
        "all_four_groups_at_least_minimum_images": lambda r: r["minimum_group_images"] >= minimum_images,
    }
    for group in ["object_object", "object_background", "background_object", "background_background"]:
        selectors[group] = lambda r, group=group: r["category_types"] == group
    for name, selector in selectors.items():
        for rho in rhos:
            selected = [r for r in rows if r["rho"] == rho and r["valid"] and selector(r)]
            count = len(selected)
            row = {"group": name, "rho": rho, "ordered_pairs": count,
                   "image_categories": len({r["image_category"] for r in selected})}
            if count:
                wrong = np.array([r["other_score"] for r in selected])
                same = np.array([r["same_score"] for r in selected])
                margin = wrong - same
                row.update(
                    exceeds=int(np.sum(margin > 0)), exceeds_percent=float(100 * np.mean(margin > 0)),
                    exceeds_with_1e12_tolerance=int(np.sum(margin > 1e-12)),
                    exact_ties=int(np.sum(margin == 0)),
                    mean_same=float(same.mean()), mean_other=float(wrong.mean()),
                    mean_margin=float(margin.mean()), median_margin=float(np.median(margin)),
                    margin_q1=float(np.quantile(margin, .25)), margin_q3=float(np.quantile(margin, .75)),
                    median_effective_images=float(np.median([r["effective_images"] for r in selected])),
                )
            results.append(row)
    return results


def draw(summaries, out, rhos, threshold, score_rows, examples):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    overall = [r for r in summaries if r["group"] == "all" and r["ordered_pairs"]]
    if not overall:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    x = [r["rho"] for r in overall]
    axes[0].plot(x, [r["mean_same"] for r in overall], "o-", color="#226B8A", label="Same category")
    axes[0].plot(x, [r["mean_other"] for r in overall], "s-", color="#B55757", label="Other category")
    axes[0].set(title=f"Fixed feature scores ({overall[0]['ordered_pairs']:,} ordered pairs)",
                ylabel="Mean image-text activation correlation")
    for group, label, color, marker in [
        ("all", "All eligible pairs", "#34495E", "o"),
        ("image_auc_below_threshold", f"Image AUROC < {threshold:g}", "#B55757", "s"),
        ("image_auc_at_least_threshold", f"Image AUROC >= {threshold:g}", "#226B8A", "^"),
    ]:
        selected = [r for r in summaries if r["group"] == group and r["ordered_pairs"]]
        if selected:
            axes[1].plot([r["rho"] for r in selected], [r["exceeds_percent"] for r in selected],
                         marker=marker, color=color, label=f"{label} (n={selected[0]['ordered_pairs']:,})")
    axes[1].set(title="Other-category score exceeds same-category score", ylabel="Ordered pairs (%)", ylim=(0, None))
    for ax in axes:
        ax.set(xlabel="Controlled annotation correlation", xticks=rhos)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Original observations, fixed features, 50% prevalence for each category", fontsize=12)
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(out / f"cooccurrence_sweep.{extension}", dpi=220)
    plt.close(fig)

    selected_examples = []
    for a, b in examples:
        selected = [r for r in score_rows if r["image_category"] == a and r["other_category"] == b and r["valid"]]
        if selected:
            selected_examples.append(selected)
    if not selected_examples:
        return
    fig, axes = plt.subplots(1, len(selected_examples), figsize=(3.5 * len(selected_examples), 3.8),
                             squeeze=False, layout="constrained")
    for ax, selected in zip(axes[0], selected_examples):
        first = selected[0]
        x = [r["rho"] for r in selected]
        ax.plot(x, [r["same_score"] for r in selected], "o-", color="#226B8A", label="Same category")
        ax.plot(x, [r["other_score"] for r in selected], "s-", color="#B55757", label="Other category")
        ax.set(title=f"Image {first['image_name']} / text {first['other_name']}\n"
                     f"Smallest group: {first['minimum_group_images']:,} images",
               xlabel="Controlled annotation correlation", xticks=rhos, ylim=(-.15, 1))
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    axes[0, 0].set_ylabel("Image-text activation correlation")
    fig.suptitle("Examples specified before the sweep; full results include every eligible pair", fontsize=11)
    for extension in ["png", "pdf"]:
        fig.savefig(out / f"prespecified_examples.{extension}", dpi=200)
    plt.close(fig)


def run(args):
    args.run, args.out = args.run.resolve(), args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if len(args.rhos) != len(set(args.rhos)) or any(not 0 <= r < 1 for r in args.rhos):
        raise ValueError("Specify unique correlations in [0, 1); all four strata must have positive mass")
    args.rhos.sort()
    sources = [args.run / p for p in [
        "representatives.json", "representatives.csv", "dataset.json",
        f"index/{args.split}/parents.npy", f"index/{args.split}/presence.npy",
        f"index/{args.split}/mentions.npy", f"index/{args.split}/concept_ids.json",
        f"activations/{args.split}/image.npz", f"activations/{args.split}/text.npz",
    ]]
    with ProgressReporter(args.out, ["load", "sweep", "report"], [], interval=10):
        with stage_progress("load"):
            ids = json.loads((args.run / f"index/{args.split}/concept_ids.json").read_text())
            reps = json.loads((args.run / "representatives.json").read_text())
            names = {c["id"]: c["name"] for c in json.loads((args.run / "dataset.json").read_text())["concepts"]}
            with (args.run / "representatives.csv").open() as stream:
                auc = {(int(r["concept_id"]), r["side"]): float(r["auroc"])
                       for r in csv.DictReader(stream) if r["split"] == args.selection_split}
            selected_ids = ids if args.concept_ids is None else args.concept_ids
            if len(set(selected_ids)) != len(selected_ids) or not set(selected_ids) <= set(ids):
                raise ValueError("Category IDs must be unique and present in the cached index")
            columns = [ids.index(c) for c in selected_ids]
            ids = selected_ids
            image_ids = [reps[str(c)]["image"] for c in ids]
            text_ids = [reps[str(c)]["text"] for c in ids]
            if any(c is None for c in image_ids + text_ids):
                raise ValueError("All requested categories must have frozen image and text representatives")
            index = args.run / "index" / args.split
            parents = np.load(index / "parents.npy")
            if np.any(parents[1:] < parents[:-1]):
                raise ValueError("Expected cached caption rows grouped by image")
            presence = np.load(index / "presence.npy")[:, columns][parents]
            mentions = np.load(index / "mentions.npy")[:, columns]
            # -5 ensures 2*state[A]+state[B] is negative whenever either category disagrees.
            states = np.where(presence == mentions, presence.astype(np.int8), -5).T.copy()
            del presence, mentions
            image = sparse.load_npz(args.run / f"activations/{args.split}/image.npz")[:, image_ids][parents]
            text = sparse.load_npz(args.run / f"activations/{args.split}/text.npz")[:, text_ids]
            full = correlation(image, text)
            x, y = sparse_columns(image), sparse_columns(text)
            del image, text
            pairs = list(itertools.combinations(range(len(ids)), 2))
            manifest = {
                "source_run": str(args.run), "split": args.split, "selection_split": args.selection_split,
                "rhos": args.rhos, "category_ids": ids, "input_captions": len(parents),
                "category_marginals": [.5, .5], "group_order": ["00", "01", "10", "11"],
                "eligibility": "Both image categories agree with dictionary caption mentions; all 4 groups nonempty",
                "observation_weight": "Stratum mass / number of eligible captions in stratum",
                "directional_comparison": "corr(image[A], text[B]) > corr(image[A], text[A]), A != B",
                "quality_threshold_descriptive_only": args.quality_threshold,
                "minimum_images_sensitivity_only": args.minimum_images,
                "prespecified_example_ids": args.examples,
                "source_sha256": {str(p): sha256(p) for p in sources},
                "script_sha256": sha256(Path(__file__)),
                "moments_code_sha256": sha256(Path(__file__).resolve().parents[1] / "src/mm_sae/metrics/reweighting.py"),
            }
            atomic_json(args.out / "manifest.json", manifest)
        score_rows, support_rows = [], []
        counts_saved = np.zeros((len(pairs), 4), dtype=np.int64)
        sums_saved = np.full((len(pairs), 4, 4), np.nan)
        products_saved = np.full((len(pairs), 4, 4, 4), np.nan)
        with stage_progress("sweep"):
            for pair_index, (i, j) in enumerate(iter_progress(pairs, "Category pairs", unit="pairs")):
                groups = 2 * states[i] + states[j]
                counts, image_counts, image_squared = group_support(groups, parents)
                counts_saved[pair_index] = counts
                support = {"pair_index": pair_index, "category_a": ids[i], "category_b": ids[j],
                           "name_a": names[ids[i]], "name_b": names[ids[j]],
                           "eligible_captions": int(counts.sum()), "eligible_images": int(image_counts.sum()),
                           "minimum_group_images": int(image_counts.min()),
                           "status": "included" if np.all(counts > 0) else "missing_group"}
                for g in range(4):
                    support[f"captions_{g:02b}"] = int(counts[g])
                    support[f"images_{g:02b}"] = int(image_counts[g])
                support_rows.append(support)
                if not np.all(counts > 0):
                    continue
                moments = grouped_moments([x[i], y[i], x[j], y[j]], groups, counts)
                sums_saved[pair_index], products_saved[pair_index] = moments.sums, moments.products
                unweighted = moments.correlations(counts / counts.sum())
                controlled = [moments.correlations(balanced_binary_masses(r)) for r in args.rhos]
                for a, b, xi, ya, yb in [(i, j, 0, 1, 3), (j, i, 2, 3, 1)]:
                    valid = all(np.isfinite(c[xi, ya]) and np.isfinite(c[xi, yb]) for c in controlled)
                    for rho, c in zip(args.rhos, controlled):
                        n_eff, image_eff = effective_counts(balanced_binary_masses(rho), counts, image_squared)
                        row = {
                            "pair_index": pair_index, "image_category": ids[a], "other_category": ids[b],
                            "image_name": names[ids[a]], "other_name": names[ids[b]],
                            "category_types": "_".join("object" if ids[k] < args.stuff_start_id else "background" for k in [a, b]),
                            "image_feature": image_ids[a], "same_text_feature": text_ids[a], "other_text_feature": text_ids[b],
                            "shared_image": image_ids[a] == image_ids[b], "shared_text": text_ids[a] == text_ids[b],
                            "image_auroc": auc[ids[a], "image"], "same_text_auroc": auc[ids[a], "text"],
                            "other_text_auroc": auc[ids[b], "text"],
                            "minimum_group_images": int(image_counts.min()),
                            "effective_captions": n_eff, "effective_images": image_eff,
                            "full_original_same": float(full["C"][a, a]), "full_original_other": float(full["C"][a, b]),
                            "agreement_unweighted_same": float(unweighted[xi, ya]) if np.isfinite(unweighted[xi, ya]) else None,
                            "agreement_unweighted_other": float(unweighted[xi, yb]) if np.isfinite(unweighted[xi, yb]) else None,
                            "rho": rho, "valid": valid, "status": "included" if valid else "constant_feature_on_support",
                            "same_score": float(c[xi, ya]) if valid else None,
                            "other_score": float(c[xi, yb]) if valid else None,
                            "margin": float(c[xi, yb] - c[xi, ya]) if valid else None,
                        }
                        score_rows.append(row)
                if (pair_index + 1) % 500 == 0:
                    print(f"Processed {pair_index + 1:,}/{len(pairs):,} unordered category pairs", flush=True)
        with stage_progress("report"):
            np.savez_compressed(args.out / "grouped_moments.npz", pairs=np.array(pairs), category_ids=ids,
                                counts=counts_saved, sums=sums_saved, products=products_saved)
            write_csv(args.out / "pair_support.csv", support_rows)
            write_csv(args.out / "scores.csv", score_rows)
            summaries = summarize(score_rows, args.rhos, args.quality_threshold, args.minimum_images)
            write_csv(args.out / "summary.csv", summaries)
            included = sum(r["status"] == "included" for r in support_rows)
            valid_directions = sum(r["valid"] for r in score_rows if r["rho"] == args.rhos[0])
            coverage = {"total_unordered_pairs": len(pairs), "all_four_groups_present": included,
                        "excluded_missing_group": len(pairs) - included,
                        "total_ordered_pairs": 2 * len(pairs), "valid_ordered_pairs": valid_directions,
                        "excluded_constant_feature_directions": 2 * included - valid_directions}
            atomic_json(args.out / "summary.json", {"coverage": coverage, "results": summaries})
            draw(summaries, args.out, args.rhos, args.quality_threshold, score_rows, args.examples)
            print(json.dumps(coverage, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", default="train2017")
    parser.add_argument("--selection-split", default="train2017")
    parser.add_argument("--rhos", type=float, nargs="+", default=[0, .2, .4, .6, .8])
    parser.add_argument("--concept-ids", type=int, nargs="+")
    parser.add_argument("--stuff-start-id", type=int, default=91)
    parser.add_argument("--quality-threshold", type=float, default=.75)
    parser.add_argument("--minimum-images", type=int, default=50,
                        help="Descriptive sensitivity subset only; no filtering of the main result")
    parser.add_argument("--examples", type=int, nargs=2, action="append", default=[])
    run(parser.parse_args())


if __name__ == "__main__":
    main()
