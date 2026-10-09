"""Render cached Recall@5 curves; leave uncomputed support sizes disconnected."""

import csv
import colorsys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "mm-sae/runs/sparsity-sweep-2026-10-06"
ASSETS = Path("/Users/jihoonkwon/Desktop/projects/project-pages/f4750a4b/multimodal-SAE/assets/2026-10-06-mapping")
METHODS = [
    ("sinkhorn_text", "Sinkhorn (text transform)", "#FFADAD"),
    ("sinkhorn_image", "Sinkhorn (image transform)", "#FFD6A5"),
    ("pls", "Sparse PLS-SVD", "#A0C4FF"),
    ("cca", "Sparse CCA", "#BDB2FF"),
]
CONDITIONS = [("coco-coco", "COCO2017 / COCO2017"),
              ("cc3m-cc3m", "CC3M / CC3M"),
              ("cc3m-coco", "CC3M / COCO2017")]
SUPPORTS = [8, 16, 32, 64, 128]


def darker(color):
    # Reference: Robot-data section4/paper_figures.py, HSL lightness adjustment.
    rgb = [int(color[i:i+2], 16)/255 for i in (1, 3, 5)]
    h, lightness, saturation = colorsys.rgb_to_hls(*rgb)
    return colorsys.hls_to_rgb(h, max(0, lightness-.085), saturation)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "font.size": 14, "axes.labelsize": 15, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "svg.fonttype": "path"})
    fig, axes = plt.subplots(2, 3, figsize=(17.8, 6.6), sharey="row", sharex=True)
    rows = []
    completed = 0
    for column, (condition, title) in enumerate(CONDITIONS):
        for method, label, color in METHODS:
            color = darker(color)
            curves = {"image_to_text": [], "text_to_image": []}
            for k in SUPPORTS:
                path = RUN / condition / "results" / f"{method}_{k}.json"
                data = json.loads(path.read_text()) if path.exists() else None
                completed += int(data is not None)
                for direction in curves:
                    value = data["retrieval"][direction]["recall"]["5"] * 100 if data else np.nan
                    curves[direction].append(value)
                    rows.append(dict(condition=condition, method=method, direction=direction,
                                     support=k, recall_at_5_percent=None if data is None else value,
                                     source=str(path)))
            for row_index, (direction, linestyle, marker) in enumerate([("image_to_text", "-", "o"),
                                                  ("text_to_image", (0, (3, 3)), "s")]):
                ax = axes[row_index, column]
                ax.plot(SUPPORTS, curves[direction], color=color, linewidth=2.2,
                        linestyle=linestyle, marker=marker, markersize=6.5,
                        markerfacecolor=color if direction == "image_to_text" else "white",
                        markeredgewidth=.8, markeredgecolor="#555555")
        axes[0, column].set_title(title, fontsize=19, fontfamily="cmr10", pad=10)
        for ax in axes[:, column]:
            ax.set_xscale("log", base=2)
            ax.set_xticks(SUPPORTS, [str(k) for k in SUPPORTS])
            ax.tick_params(axis="both", length=4, width=1, pad=4, labelleft=True)
            ax.grid(color="#d9d9d9", linewidth=.85, alpha=.7)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_color("#222222")
                spine.set_linewidth(1.1)
            ax.set_xlim(7.2, 143)
    for row_index, direction in enumerate(("image_to_text", "text_to_image")):
        measured = [row["recall_at_5_percent"] for row in rows
                    if row["direction"] == direction and row["recall_at_5_percent"] is not None]
        lower = 5 * np.floor((min(measured) - .8)/5)
        upper = 5 * np.ceil((max(measured) + .8)/5)
        for ax in axes[row_index]:
            ax.set_ylim(lower, upper)
            ax.set_yticks(np.arange(lower, upper+.1, 5))
    axes[0, 0].set_ylabel("Image query\nRecall@5 (%)", labelpad=10)
    axes[1, 0].set_ylabel("Text query\nRecall@5 (%)", labelpad=10)
    fig.supxlabel("Maximum activations / connections", fontsize=15, y=.02)
    handles = [Line2D([0], [0], color=darker(c), lw=2.8, marker="o", markersize=6,
                      markeredgecolor="#555555", markeredgewidth=.7, label=label)
               for _, label, c in METHODS]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=13,
               handlelength=2, columnspacing=1.7, handletextpad=.6, bbox_to_anchor=(.53, 1.01))
    fig.text(.53, .905, "SAE training / Mapping training", ha="center", fontsize=12, color="#555555")
    fig.subplots_adjust(top=.80, bottom=.12, left=.08, right=.99, wspace=.19, hspace=.24)
    for suffix in ["svg", "png"]:
        fig.savefig(ASSETS / f"sparsity-recall5.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)
    (ASSETS / "sparsity-recall5-status.json").write_text(json.dumps({
        "completed": completed, "total": 60, "pending": 60-completed,
        "missing_results_are_not_interpolated": True}, indent=2))
    with (ASSETS / "sparsity-recall5.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Support sweep chart: {completed}/60 results")


if __name__ == "__main__":
    main()
