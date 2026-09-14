"""Three manuscript figures, each backed by full CSV/JSON outputs."""

import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mm_sae.io import atomic_json

COLORS = ["#217568", "#c7634b", "#a7afa9", "#8d80ab", "#decda1"]


def finish(fig, path, smoke):
    if smoke:
        fig.suptitle(
            "SMOKE TEST ONLY. Insufficient data for scientific conclusions.", fontsize=10, color="#a33"
        )
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def report(config, options):
    root = config.output / "rq1"
    figs = root / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    smoke = bool(config.data.image_limits or config.training.max_steps or config.data.source == "synthetic")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    e1 = json.loads((root / "experiment1" / "summary.json").read_text())
    bins = e1["bins"]
    fig, ax = plt.subplots(figsize=(8, 4.1))
    x = np.arange(len(bins))
    valid = [i for i, b in enumerate(bins) if b["n_pairs"]]
    if valid:
        ax.errorbar(
            valid,
            [bins[i]["mean"] for i in valid],
            yerr=[bins[i]["std_population"] for i in valid],
            fmt="o",
            capsize=4,
            color=COLORS[0],
            label="Mean ± population SD",
        )
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No defined non-self concept pairs", transform=ax.transAxes, ha="center")
    labels = [
        f"[{b['left']:.1f}, {b['right']:.1f}{']' if b['right_inclusive'] else ')'}\nn={b['n_pairs']}"
        for b in bins
    ]
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.axhline(0, color="#bbb", lw=0.7)
    ax.set(
        xlabel=f"Annotation correlation bins ({options.label_correlation})",
        ylabel="Image–text feature correlation",
    )
    finish(fig, figs / "figure1_annotation_vs_coactivation", smoke)

    e2 = json.loads((root / "experiment2" / "summary.json").read_text())
    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    methods = ["greedy", "hungarian"]
    statuses = ["same", "different", "ambiguous_or_unlabeled", "undefined_correlation", "unmatched"]
    labels = [
        "Same concept",
        "Different concept",
        "Ambiguous / unlabeled",
        "Undefined correlation",
        "Unmatched",
    ]
    bottom = np.zeros(2)
    for s, label, color in zip(statuses, labels, COLORS):
        values = np.array(
            [100 * e2[m]["counts"][s] / e2[m]["denominator"] if e2[m]["denominator"] else 0 for m in methods]
        )
        ax.bar(np.arange(2), values, bottom=bottom, color=color, label=label, width=0.55)
        bottom += values
    ax.set_xticks([0, 1], [f"{m.title()}\nn={e2[m]['denominator']}" for m in methods])
    ax.set(ylabel="Fraction of fixed representative image features (%)", ylim=(0, 100))
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    finish(fig, figs / "figure2_actual_matching", smoke)

    e3 = json.loads((root / "experiment3" / "summary.json").read_text())
    rows = e3["outcomes"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    conditions = ["original", "random_removal", "cooccurrence_removal"]
    ticks = ["Original", "Random\nremoval", "Conditional\nremoval"]
    if rows:
        unique = list({(r["anchor"], r["remove"], r["repeat"], r["condition"]): r for r in rows}.values())
        for metric, name, color in [
            ("same_concept_score", "Same-concept partner", COLORS[0]),
            ("wrong_score", "Original wrong partner", COLORS[1]),
        ]:
            means = []
            for condition in conditions:
                values = [r[metric] for r in unique if r["condition"] == condition and r[metric] is not None]
                means.append(np.mean(values) if values else np.nan)
            axes[0].plot(range(3), means, "o-", color=color, label=name)
        for method, color in zip(methods, COLORS):
            recovered, damaged = [], []
            for condition in conditions:
                error_rows = [
                    r
                    for r in rows
                    if r["method"] == method
                    and r["condition"] == condition
                    and r["original_status"] == "different"
                ]
                recovered.append(
                    100 * np.mean([r["recovered"] for r in error_rows]) if error_rows else np.nan
                )
                correct_rows = [
                    r
                    for r in rows
                    if r["method"] == method and r["condition"] == condition and r["baseline_correct_rows"]
                ]
                damaged.append(
                    100
                    * np.mean(
                        [
                            (r["newly_wrong_rows"] + r["newly_unassessable_rows"])
                            / r["baseline_correct_rows"]
                            for r in correct_rows
                        ]
                    )
                    if correct_rows
                    else np.nan
                )
            axes[1].plot(range(3), recovered, "o-", label=method.title(), color=color)
            axes[2].plot(range(3), damaged, "o-", label=method.title(), color=color)
        off = [
            np.mean([r["mean_image_features_turned_off"] for r in unique if r["condition"] == condition])
            for condition in conditions
        ]
        axes[3].plot(range(3), off, "o-", color=COLORS[0])
        for ax in axes[:3]:
            ax.legend(frameon=False, fontsize=8)
    else:
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "No assessable intervention cases\nSee exclusions.csv",
                transform=ax.transAxes,
                ha="center",
                fontsize=10,
            )
    for ax in axes:
        ax.set_xticks(range(3), ticks)
    axes[0].set_ylabel("Mean correlation across case-specific interventions")
    axes[1].set(ylabel="Recovery of original errors (%)", ylim=(-2, 102))
    axes[2].set(ylabel="Loss of originally correct matches (%)", ylim=(-2, 102))
    axes[3].set_ylabel("Mean features turned off per edited image")
    finish(fig, figs / "figure3_interventions", smoke)
    atomic_json(
        root / "report.json",
        {
            "smoke_test_only": smoke,
            "figures": 3,
            "original_error_pairs": e3["original_error_concept_pairs"],
            "eligible_error_pairs": len(e3["eligible_pairs"]),
            "experiment1_pairs": e1["pairs"],
            "experiment2": e2,
            "cautions": [
                "Feature representatives may be weak or shared; inspect representatives.csv and held-out AUROC.",
                "Automatic caption span labels are not human-verified semantic ground truth.",
                "Figure 1 excludes self concepts and does not measure semantic discrimination accuracy.",
                "Each intervention defines its own correlation matrix; recovery is not global held-out matching accuracy.",
            ],
        },
    )
