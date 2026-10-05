"""Plot category distributions, retaining the same selected top-one/top-five features.

This only reads completed runs. The additional top-one transfer AUROC uses its
previously fitted presence probe and cached validation activations, without fitting.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy import sparse

from mm_sae.analysis.data import save_json
from mm_sae.analysis.evaluation import auroc, distribution
from mm_sae.analysis.linear import LinearScore
from mm_sae.io import sha256

CONDITIONS = ["embedding", "reconstruction", "all_sae", "selected_one", "selected_five"]
LABELS = ["원래 이미지\n임베딩", "SAE 복원\n임베딩", "SAE\n전체 특징", "선택한\n특징 1개", "선택한\n특징 5개"]
PALETTE = ["#FFADAD", "#FFD6A5", "#FDFFB6", "#CAFFBF", "#9BF6FF"]
TASKS = [
    (
        "presence",
        "presence",
        "범주가 있는 이미지와 없는 이미지의 구별",
        "분류기 학습 자료: 가리지 않은 원본 이미지",
    ),
    (
        "removal",
        "removal",
        "같은 범주의 원본과 가림본의 구별",
        "점수 구성 자료: 해당 범주의 원본·가림 이미지",
    ),
    (
        "presence",
        "removal",
        "원본으로 학습한 범주 점수의 가림 반응",
        "원본에서 학습한 분류기를 가림본에 그대로 적용",
    ),
]


def read_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_rows(path, records):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def collect(diagnosis, feature_run, source):
    paths = [diagnosis / "report/category_metrics.csv", feature_run / "report/concepts.csv"]
    records = []
    for r in read_rows(paths[0]):
        if r["representation"] in CONDITIONS[:-2] + ["selected_five", "single_feature"]:
            if (r["trained_on"], r["evaluated_on"]) not in [(a, b) for a, b, _, _ in TASKS]:
                continue
            records.append(
                dict(
                    category_id=int(r["category_id"]),
                    name=r["name"],
                    kind=r["kind"],
                    representation="selected_one"
                    if r["representation"] == "single_feature"
                    else r["representation"],
                    trained_on=r["trained_on"],
                    evaluated_on=r["evaluated_on"],
                    auroc=float(r["auroc"]),
                )
            )
    previous = read_rows(paths[1])
    for r in previous:
        if (
            r["side"] == "image"
            and r["task"] == "concept_presence"
            and r["condition"] == "top_n"
            and r["n"] == "1"
        ):
            records.append(
                dict(
                    category_id=int(r["category_id"]),
                    name=r["name"],
                    kind=r["kind"],
                    representation="selected_one",
                    trained_on="presence",
                    evaluated_on="presence",
                    auroc=float(r["auroc"]),
                )
            )
    ids = json.loads((source / "index/val2017/concept_ids.json").read_text())
    original_path = source / "activations/val2017/image.npz"
    original = sparse.load_npz(original_path).tocsr()
    paths.append(original_path)
    for cid in ids:
        probe_path = feature_run / "concepts" / f"image_{cid}.json"
        record = json.loads(probe_path.read_text())
        model = LinearScore.from_record(record["probes"]["top_n_1"])
        root = source / "counterfactual/val2017" / str(cid)
        rows = np.load(root / "image_rows.npy")
        changed = sparse.load_npz(root / "image_activations.npz").tocsr()
        before, after = model.predict(original[rows]), model.predict(changed)
        score, y = np.r_[before, after], np.r_[np.ones(len(rows)), np.zeros(len(rows))]
        meta = record["records"][0]
        records.append(
            dict(
                category_id=cid,
                name=meta["name"],
                kind=meta["kind"],
                representation="selected_one",
                trained_on="presence",
                evaluated_on="removal",
                auroc=auroc(y, score),
            )
        )
        paths += [probe_path, root / "image_rows.npy", root / "image_activations.npz"]
    # Enforce identical, complete populations in every plotted distribution.
    for rep in CONDITIONS:
        for trained, evaluated, _, _ in TASKS:
            subset = [
                r
                for r in records
                if (r["representation"], r["trained_on"], r["evaluated_on"]) == (rep, trained, evaluated)
            ]
            assert sorted(r["category_id"] for r in subset) == sorted(ids)
            assert np.isfinite([r["auroc"] for r in subset]).all()
    # The existing presence probes on the same selected five must agree across runs.
    old = {
        int(r["category_id"]): float(r["auroc"])
        for r in previous
        if r["side"] == "image"
        and r["task"] == "concept_presence"
        and r["condition"] == "top_n"
        and r["n"] == "5"
    }
    for r in records:
        if (r["representation"], r["trained_on"], r["evaluated_on"]) == (
            "selected_five",
            "presence",
            "presence",
        ):
            np.testing.assert_allclose(r["auroc"], old[r["category_id"]], atol=1e-10, rtol=0)
    return records, paths


def get_values(records, rep, trained, evaluated, kind="all"):
    return np.array(
        [
            r["auroc"]
            for r in records
            if (r["representation"], r["trained_on"], r["evaluated_on"]) == (rep, trained, evaluated)
            and (kind == "all" or r["kind"] == kind)
        ]
    )


def draw_box(ax, arrays, positions, colors, width=0.55):
    result = ax.boxplot(
        arrays,
        positions=positions,
        widths=width,
        patch_artist=True,
        whis=(0, 100),
        showfliers=False,
        showmeans=True,
        manage_ticks=False,
        meanprops=dict(marker="D", markersize=4.5, markerfacecolor="#252525", markeredgecolor="#252525"),
        medianprops=dict(color="#252525", linewidth=1.7),
        whiskerprops=dict(color="#6b7280", linewidth=1),
        capprops=dict(color="#6b7280", linewidth=1),
        boxprops=dict(edgecolor="#4b5563", linewidth=1.1),
    )
    for box, color in zip(result["boxes"], colors):
        box.set_facecolor(color)
    ax.axhline(0.5, color="#8b8b8b", linestyle="--", linewidth=0.9, zorder=0)
    ax.set_ylim(0, 1.035)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.grid(axis="y", alpha=0.12)


def export(fig, output, stem):
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(output / f"{stem}.{extension}", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figures(records, output):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.3), sharey=True)
    for ax, (trained, evaluated, title, subtitle) in zip(axes, TASKS):
        arrays = [get_values(records, rep, trained, evaluated) for rep in CONDITIONS]
        draw_box(ax, arrays, np.arange(1, 6), PALETTE)
        ax.set_xticks(np.arange(1, 6), LABELS, fontsize=11)
        ax.set_title(title + "\n" + subtitle, fontsize=12, linespacing=1.7, pad=14)
        ax.set_xlim(0.45, 5.55)
    axes[0].set_ylabel("범주별 AUROC", fontsize=13)
    fig.suptitle("같은 171개 범주에서 비교한 이미지 표현의 성능 분포", fontsize=17, y=1.02)
    fig.text(
        0.5,
        -0.025,
        "상자: 25–75% 분위수    가로선: 중앙값    ◆: 평균    수염 끝: 최솟값·최댓값",
        ha="center",
        fontsize=12,
    )
    fig.subplots_adjust(wspace=0.13, bottom=0.19)
    export(fig, output, "all_representations_boxplot")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.9), sharey=True)
    for ax, (trained, evaluated, title, _) in zip(axes, TASKS[:2]):
        arrays, positions, colors, ticks = [], [], [], []
        for j, kind in enumerate(["all", "object", "background"]):
            ticks.append(1 + j * 1.5)
            for k, rep in enumerate(["selected_one", "selected_five"]):
                arrays.append(get_values(records, rep, trained, evaluated, kind))
                positions.append(ticks[-1] + (k - 0.5) * 0.45)
                colors.append(PALETTE[3 + k])
        draw_box(ax, arrays, positions, colors, width=0.37)
        ax.set_xticks(ticks, ["전체\n171개 범주", "물체\n80개 범주", "배경\n91개 범주"])
        ax.set_title(title, fontsize=14, pad=16)
    axes[0].set_ylabel("범주별 AUROC")
    fig.legend(
        handles=[
            Patch(facecolor=PALETTE[3], edgecolor="#4b5563", label="선택한 특징 1개"),
            Patch(facecolor=PALETTE[4], edgecolor="#4b5563", label="선택한 특징 5개"),
            Line2D([], [], marker="D", color="#252525", linestyle="none", label="평균"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.subplots_adjust(bottom=0.22, wspace=0.15)
    export(fig, output, "one_vs_five_boxplot")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnosis-run", type=Path, required=True)
    parser.add_argument("--feature-run", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    font = args.font or next(
        (f for f in ["NanumGothic", "Apple SD Gothic Neo", "Noto Sans CJK KR"] if f in fonts), None
    )
    if font is None:
        raise RuntimeError("A Korean font is required; supply --font")
    plt.rcParams.update(
        {
            "font.family": font,
            "font.size": 12,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "path",
        }
    )
    records, paths = collect(args.diagnosis_run, args.feature_run, args.source_run)
    write_rows(args.output / "category_aurocs.csv", records)
    stats = []
    for trained, evaluated, _, _ in TASKS:
        for kind in ["all", "object", "background"]:
            for rep in CONDITIONS:
                stats.append(
                    dict(
                        trained_on=trained,
                        evaluated_on=evaluated,
                        kind=kind,
                        representation=rep,
                        **distribution(get_values(records, rep, trained, evaluated, kind)),
                    )
                )
    write_rows(args.output / "distribution_statistics.csv", stats)
    save_json(
        args.output / "sources.json",
        dict(
            inputs={str(p.resolve()): sha256(p) for p in paths},
            definition="Top one and top five from the same fit-removal AUROC ranking; no fitting in this report.",
            single_removal="Original top-one activation, no fitted weight.",
            five_removal="Previously fitted signed linear score on the same top five.",
            boxplot="Category distribution; whiskers min/max, box Q1/Q3, line median, diamond mean.",
        ),
    )
    figures(records, args.output)
    for r in stats:
        if r["kind"] == "all" and r["representation"] in ["selected_one", "selected_five"]:
            print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
