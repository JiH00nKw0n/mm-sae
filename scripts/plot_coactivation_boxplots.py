"""Plot the saved RQ1 category-pair scores without recomputing any experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mm_sae.io import atomic_json, sha256, write_csv

COLORS = ["#FFADAD", "#FFD6A5", "#FDFFB6", "#CAFFBF", "#9BF6FF"]
GROUPS = {
    "object_object": "Image object / text object",
    "object_background": "Image object / text background",
    "background_object": "Image background / text object",
    "background_background": "Image background / text background",
}
EDGES = np.round(np.linspace(0, 1, 6), 10)


def summarize(rows, group):
    result, values = [], []
    for index, (left, right) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        scores = np.asarray(
            [
                r["score"]
                for r in rows
                if r["group"] == group or group == "all"
                if left <= r["annotation"]
                and (r["annotation"] < right or index == 4 and r["annotation"] <= right)
            ],
            dtype=float,
        )
        row = {
            "group": group,
            "annotation_left": float(left),
            "annotation_right": float(right),
            "right_inclusive": index == 4,
            "n_pairs": len(scores),
            **dict.fromkeys(["min", "q1", "median", "q3", "max", "mean", "std_population"]),
        }
        if len(scores):
            q1, median, q3 = np.quantile(scores, [.25, .5, .75], method="linear")
            row.update(
                min=float(scores.min()), q1=float(q1), median=float(median), q3=float(q3),
                max=float(scores.max()), mean=float(scores.mean()), std_population=float(scores.std()),
            )
        result.append(row)
        values.append(scores)
    return result, values


def draw(ax, summaries, values, colors, title):
    for index, (row, scores, color) in enumerate(zip(summaries, values, colors)):
        if not row["n_pairs"]:
            continue
        ax.bxp(
            [{"q1": row["q1"], "med": row["median"], "q3": row["q3"],
              "whislo": row["min"], "whishi": row["max"], "fliers": []}],
            positions=[index], widths=.56, patch_artist=True, showfliers=False,
            manage_ticks=False,
            boxprops={"facecolor": color, "edgecolor": "#49545C", "linewidth": 1.15},
            whiskerprops={"color": "#49545C", "linewidth": 1.05},
            capprops={"color": "#49545C", "linewidth": 1.05},
            medianprops={"color": "#202A31", "linewidth": 1.7},
        )
        if len(scores) <= 10:
            offsets = np.linspace(-.075, .075, len(scores)) if len(scores) > 1 else [0]
            ax.scatter(index + np.asarray(offsets), np.sort(scores), s=17,
                       c="#36434D", edgecolors="white", linewidths=.45, zorder=4)
    labels = [
        f"[{left:.1f}, {right:.1f}{']' if i == 4 else ')'}\nn = {row['n_pairs']:,}"
        for i, (left, right, row) in enumerate(zip(EDGES[:-1], EDGES[1:], summaries))
    ]
    ax.set(xticks=range(5), xticklabels=labels, xlim=(-.6, 4.6), ylim=(-.1, 1.0))
    ax.set_title(title, fontsize=12, fontweight="semibold", pad=13)
    ax.set_yticks(np.arange(0, 1.01, .2))
    ax.axhline(0, color="#9BA4AC", linewidth=.8, zorder=0)
    ax.grid(axis="y", color="#E6E9EC", linewidth=.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=7)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["bottom", "left"]:
        ax.spines[side].set_color("#B7BEC4")
    ax.set_xlabel("Image-annotation correlation", labelpad=9)
    ax.set_ylabel("Image–text coactivation correlation", labelpad=9)


def save(fig, stem):
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(stem.with_suffix("." + suffix), dpi=240, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--colors", nargs=5, default=COLORS)
    parser.add_argument("--stuff-start-id", type=int, default=91,
                        help="First stuff/background ID in raw COCO-Stuff PNG labels (default 91)")
    args = parser.parse_args()
    args.run, args.out = args.run.resolve(), args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.run / "rq1/experiment1/all_ordered_concept_pairs.csv"
    with source.open(newline="") as stream:
        raw = list(csv.DictReader(stream))
    rows = []
    for row in raw:
        if row["valid"] != "True":
            continue
        a, b = int(row["image_concept"]), int(row["text_concept"])
        assert a != b
        sides = ["background" if c >= args.stuff_start_id else "object" for c in [a, b]]
        rows.append({"group": "_".join(sides), "annotation": round(float(row["label_correlation"]), 10),
                     "score": float(row["coactivation_correlation"])})
    assert all(np.isfinite(r["annotation"]) and np.isfinite(r["score"]) for r in rows)
    assert all(r["label_source"] == "image_image" for r in raw)
    summaries = {group: summarize(rows, group) for group in ["all", *GROUPS]}
    with (args.run / "rq1/experiment1/bins.csv").open(newline="") as stream:
        baseline = [r for r in csv.DictReader(stream) if float(r["left"]) >= 0]
    for row, prior in zip(summaries["all"][0], baseline, strict=True):
        assert row["n_pairs"] == int(prior["n_pairs"])
        if row["n_pairs"]:
            assert np.isclose(row["mean"], float(prior["mean"]), rtol=0, atol=1e-14)
            assert np.isclose(row["std_population"], float(prior["std_population"]), rtol=0, atol=1e-14)
    for i, row in enumerate(summaries["all"][0]):
        assert row["n_pairs"] == sum(summaries[g][0][i]["n_pairs"] for g in GROUPS)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.labelcolor": "#27333C", "text.color": "#27333C"})
    note = "Box: 25th–75th percentile · line: median · whiskers: minimum–maximum"
    fig, ax = plt.subplots(figsize=(7.9, 5.1))
    draw(ax, *summaries["all"], args.colors, "Different-category feature pairs")
    fig.subplots_adjust(left=.12, right=.975, top=.88, bottom=.25)
    fig.text(.12, .068, note, fontsize=9)
    fig.text(.12, .029, "n counts ordered category pairs; dots show individual values when n ≤ 10.", fontsize=8.5)
    save(fig, args.out / "all_pairs_boxplot")
    fig, axes = plt.subplots(2, 2, figsize=(11.1, 8.6), sharey=True)
    for ax, (group, title) in zip(axes.flat, GROUPS.items()):
        draw(ax, *summaries[group], args.colors, title)
    fig.subplots_adjust(left=.08, right=.975, top=.88, bottom=.16, hspace=.52, wspace=.21)
    fig.suptitle("Object and background pairs, shown separately", fontsize=15, y=.975)
    fig.text(.08, .056, note, fontsize=9)
    fig.text(.08, .026, "Object = COCO thing; background = COCO stuff (regions/materials). Empty bins have n = 0.", fontsize=9)
    save(fig, args.out / "pair_types_boxplot")
    write_csv(args.out / "boxplot_statistics.csv", [r for summary, _ in summaries.values() for r in summary])
    metadata = {
        "source": str(source), "source_sha256": sha256(source), "colors": args.colors,
        "valid_different_category_pairs": len(rows),
        "plotted_pairs": sum(r["n_pairs"] for r in summaries["all"][0]),
        "omitted_negative_annotation_pairs": sum(r["annotation"] < 0 for r in rows),
        "quartile_definition": "numpy.quantile(method='linear'), 25%, 50%, 75%",
        "whisker_definition": "Actual minimum and maximum, not the 1.5 IQR convention",
        "binning": "Annotation correlations rounded to 10 decimals; [left,right), final bin includes 1",
        "candidate_filtering": "Only original experiment validity; no new AUROC, score, or shared-feature exclusion",
        "interpretation": "Different annotated categories need not imply independently verified different feature meanings",
        "stuff_start_id": args.stuff_start_id,
        "verification": "All counts, means, and population standard deviations reproduce the saved experiment bins",
    }
    atomic_json(args.out / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2))
    print(json.dumps({g: [r["n_pairs"] for r in summary] for g, (summary, _) in summaries.items()}))


if __name__ == "__main__":
    main()
