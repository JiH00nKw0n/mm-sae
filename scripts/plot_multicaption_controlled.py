#!/usr/bin/env python3
"""Render compact figures for the controlled multi-caption audit."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = [
    "hungarian",
    "sinkhorn",
    "procrustes",
    "sparse_pls",
    "sparse_cca",
    "cca_cov64",
    "pls",
    "cca",
    "cca_cov",
]
DISPLAY = {
    "hungarian": "Hungarian",
    "sinkhorn": "Sinkhorn",
    "procrustes": "Procrustes",
    "sparse_pls": "Sparse PLS",
    "sparse_cca": "Sparse CCA",
    "cca_cov64": "CCA rot. top-64",
    "pls": "PLS",
    "cca": "CCA",
    "cca_cov": "CCA + cov. rot.",
}
AGGREGATIONS = {
    "clip_mean_pre_sae": ("CLIP mean → SAE", "o"),
    "activation_mean_post_sae": ("SAE → activation mean", "s"),
}
COLORS = dict(zip(METHOD_ORDER, plt.cm.tab10(np.linspace(0, 0.9, len(METHOD_ORDER)))))


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.axisbelow": True,
        }
    )


def caption_count_curves(summary: list[dict], output: Path) -> None:
    selected = [
        row
        for row in summary
        if row["benchmark"] == "controlled_5000x5000" and row["control"] == "same_image"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, aggregation in zip(axes, AGGREGATIONS):
        for method in METHOD_ORDER:
            means, lows, highs = [], [], []
            for k in (1, 2, 3, 5):
                values = np.asarray(
                    [
                        float(row["mean_R@5"]) * 100
                        for row in selected
                        if row["method"] == method
                        and row["aggregation"] == aggregation
                        and int(row["k"]) == k
                    ]
                )
                means.append(values.mean())
                lows.append(np.quantile(values, 0.025))
                highs.append(np.quantile(values, 0.975))
            axis.plot((1, 2, 3, 5), means, marker="o", color=COLORS[method], label=DISPLAY[method])
            axis.fill_between((1, 2, 3, 5), lows, highs, color=COLORS[method], alpha=0.10)
        axis.set_title(AGGREGATIONS[aggregation][0])
        axis.set_xlabel("Captions from the same image (k)")
        axis.set_xticks((1, 2, 3, 5))
    axes[0].set_ylabel("Bidirectional mean R@5 (%)")
    axes[1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("More observations improve retrieval, but methods respond differently", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output / "caption_count_curves.png", bbox_inches="tight")
    plt.close(fig)


def effect_intervals(bootstrap: list[dict], output: Path) -> None:
    selected = [
        row
        for row in bootstrap
        if row["control"] == "same_image" and row["comparison"] == "k5-k1"
    ]
    fig, axis = plt.subplots(figsize=(8, 5.2))
    y = np.arange(len(METHOD_ORDER))
    offsets = {"clip_mean_pre_sae": -0.13, "activation_mean_post_sae": 0.13}
    colors = {"clip_mean_pre_sae": "#5B7DB1", "activation_mean_post_sae": "#D66A4A"}
    for aggregation, (label, marker) in AGGREGATIONS.items():
        values = {row["method"]: row for row in selected if row["aggregation"] == aggregation}
        mean = np.asarray([float(values[m]["mean_R@5_difference"]) * 100 for m in METHOD_ORDER])
        low = np.asarray([float(values[m]["ci95_low"]) * 100 for m in METHOD_ORDER])
        high = np.asarray([float(values[m]["ci95_high"]) * 100 for m in METHOD_ORDER])
        axis.errorbar(
            mean,
            y + offsets[aggregation],
            xerr=np.vstack((mean - low, high - mean)),
            fmt=marker,
            color=colors[aggregation],
            capsize=2,
            label=label,
        )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(y, [DISPLAY[m] for m in METHOD_ORDER])
    axis.set_xlabel("Paired bidirectional mean R@5 change, k=5 − k=1 (%p)")
    axis.set_title("Dense shared subspaces use additional caption evidence most", fontweight="bold")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "k1_to_k5_effects.png", bbox_inches="tight")
    plt.close(fig)


def control_comparison(bootstrap: list[dict], output: Path) -> None:
    selected = [row for row in bootstrap if row["comparison"] == "k5-k1"]
    grouped = defaultdict(list)
    for row in selected:
        grouped[(row["control"], row["aggregation"])].append(float(row["mean_R@5_difference"]) * 100)
    controls = ["same_image", "duplicate", "cross_image"]
    labels = ["Same-image\nadditional captions", "Same-caption\nduplicate", "Cross-image\nshuffled captions"]
    x = np.arange(3)
    fig, axis = plt.subplots(figsize=(8, 4.6))
    for aggregation, (label, marker) in AGGREGATIONS.items():
        offset = -0.12 if aggregation == "clip_mean_pre_sae" else 0.12
        color = "#5B7DB1" if aggregation == "clip_mean_pre_sae" else "#D66A4A"
        for index, control in enumerate(controls):
            values = np.asarray(grouped[(control, aggregation)])
            axis.scatter(
                np.full(len(values), x[index] + offset),
                values,
                alpha=0.45,
                s=20,
                color=color,
                marker=marker,
            )
            axis.scatter(x[index] + offset, values.mean(), color=color, marker=marker, s=70, edgecolor="white")
        axis.plot([], [], color=color, marker=marker, linestyle="none", label=label)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Bidirectional mean R@5 change, k=5 − k=1 (%p)")
    axis.set_title("The gain requires coherent observations of the same scene", fontweight="bold")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "control_comparison.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    summary, bootstrap = rows(args.summary), rows(args.bootstrap)
    caption_count_curves(summary, args.output)
    effect_intervals(bootstrap, args.output)
    control_comparison(bootstrap, args.output)


if __name__ == "__main__":
    main()
