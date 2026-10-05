"""Export all category results and compact, common-cohort scientific figures."""

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mm_sae.analysis.data import save_json

LABELS = {
    "embedding": "원래 CLIP 임베딩",
    "reconstruction": "SAE 복원 임베딩",
    "all_sae": "SAE 전체 특징",
    "selected_five": "선택한 5개 (부호 허용)",
    "selected_five_positive": "선택한 5개 (음수 금지)",
    "single_feature": "기존 특징 1개",
    "reconstruction_fixed_probe": "복원 임베딩, 원래 분류기 고정",
}
COLORS = {
    "embedding": "#5A87B5",
    "reconstruction": "#8F83B5",
    "all_sae": "#64A58A",
    "selected_five": "#DAA25D",
    "selected_five_positive": "#CF8585",
    "single_feature": "#8D939B",
}
MAIN = ["embedding", "reconstruction", "all_sae", "selected_five"]
PANELS = [
    ("presence", "presence", "원본 이미지에서 범주의 존재 판별"),
    ("removal", "removal", "가림을 학습한 뒤 원본·가림 구별"),
    ("presence", "removal", "원본에서 학습한 존재 점수로 원본·가림 구별"),
]


def write_rows(path, records):
    if not records:
        return
    keys = list(dict.fromkeys(k for r in records for k in r))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def save_figure(fig, root, name):
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(root / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_report(out):
    rows = []
    for folder in ["jobs", "baselines"]:
        for path in sorted((out / folder).glob("*.json")):
            rows.extend(json.loads(path.read_text())["records"])
    root = out / "report"
    root.mkdir(exist_ok=True)
    write_rows(root / "category_metrics.csv", rows)
    lookup = {(r["category_id"], r["representation"], r["trained_on"], r["evaluated_on"]): r for r in rows}
    ids = sorted({r["category_id"] for r in rows})
    summary = []
    common = {}
    for trained, metric, _ in PANELS:
        cohort = [
            cid
            for cid in ids
            if all(lookup.get((cid, rep, trained, metric), {}).get("auroc") is not None for rep in MAIN)
        ]
        common[trained, metric] = cohort
        for kind in ["all", "object", "background"]:
            selected = [
                cid for cid in cohort if kind == "all" or ("object" if cid < 91 else "background") == kind
            ]
            for rep in MAIN + (["selected_five_positive", "single_feature"] if trained == "removal" else []):
                values = [
                    lookup[cid, rep, trained, metric]["auroc"]
                    for cid in selected
                    if (cid, rep, trained, metric) in lookup
                ]
                if not values:
                    continue
                summary.append(
                    dict(
                        trained_on=trained,
                        evaluated_on=metric,
                        representation=rep,
                        kind=kind,
                        categories=len(values),
                        mean=float(np.mean(values)),
                        median=float(np.median(values)),
                        std=float(np.std(values)),
                        min=float(np.min(values)),
                        max=float(np.max(values)),
                        at_least_0_7=int(np.sum(np.asarray(values) >= 0.7)),
                    )
                )
    write_rows(root / "summary.csv", summary)
    save_json(
        root / "summary.json",
        dict(summary=summary, common_categories={f"{a}_{b}": v for (a, b), v in common.items()}),
    )
    differences = []
    for trained, metric, _ in PANELS:
        pairs = [("embedding", "all_sae"), ("all_sae", "selected_five"), ("embedding", "reconstruction")]
        if trained == "removal":
            pairs.append(("selected_five", "selected_five_positive"))
        for cid in common[trained, metric]:
            for a, b in pairs:
                if (cid, b, trained, metric) not in lookup:
                    continue
                aa, bb = lookup[cid, a, trained, metric], lookup[cid, b, trained, metric]
                boot = []
                for rep in [a, b]:
                    with np.load(out / "scores" / f"{cid}_{rep}_{trained}.npz") as data:
                        boot.append(np.array(data[f"{metric}_bootstrap"]))
                low, high = (
                    np.quantile(boot[0] - boot[1], [0.025, 0.975])
                    if len(boot[0]) and len(boot[0]) == len(boot[1])
                    else [np.nan, np.nan]
                )
                differences.append(
                    dict(
                        category_id=cid,
                        name=aa["name"],
                        kind=aa["kind"],
                        trained_on=trained,
                        evaluated_on=metric,
                        first=a,
                        second=b,
                        auroc_difference=aa["auroc"] - bb["auroc"],
                        ci_low=low,
                        ci_high=high,
                        images=aa["images"],
                    )
                )
    write_rows(root / "paired_auroc_differences.csv", differences)
    plt.rcParams.update(
        {
            "font.family": "NanumGothic",
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    for ax, (trained, metric, title) in zip(axes, PANELS):
        for k, kind in enumerate(["all", "object", "background"]):
            for j, rep in enumerate(MAIN):
                item = next(
                    (
                        r
                        for r in summary
                        if r["trained_on"] == trained
                        and r["evaluated_on"] == metric
                        and r["representation"] == rep
                        and r["kind"] == kind
                    ),
                    None,
                )
                if item:
                    x = k + (j - 1.5) * 0.19
                    ax.bar(
                        x, item["mean"], width=0.18, color=COLORS[rep], label=LABELS[rep] if k == 0 else None
                    )
                    ax.text(
                        x,
                        item["mean"] + 0.015,
                        f"{item['mean']:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=90,
                    )
        ax.set_xticks([0, 1, 2], ["전체 범주", "물체", "배경"])
        ax.set_ylim(0, 1.08)
        ax.axhline(0.5, color="#666666", lw=0.8, ls="--")
        ax.set_title(title + f"\n동일한 {len(common[trained, metric])}개 범주 비교", fontsize=12)
    axes[0].set_ylabel("범주별 AUROC의 평균")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.19, wspace=0.10)
    save_figure(fig, root, "01_representation_comparison")
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for col, (trained, metric, title) in enumerate(PANELS):
        for row, (a, b) in enumerate([("embedding", "all_sae"), ("all_sae", "selected_five")]):
            ax = axes[row, col]
            for kind, color in [("object", "#5A87B5"), ("background", "#DAA25D")]:
                selected = [
                    cid for cid in common[trained, metric] if lookup[cid, a, trained, metric]["kind"] == kind
                ]
                ax.scatter(
                    [lookup[cid, a, trained, metric]["auroc"] for cid in selected],
                    [lookup[cid, b, trained, metric]["auroc"] for cid in selected],
                    s=17,
                    alpha=0.65,
                    label="물체" if kind == "object" else "배경",
                    color=color,
                )
            ax.plot([0, 1], [0, 1], color="#777777", lw=0.8, ls="--")
            ax.set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel=LABELS[a] + " AUROC", ylabel=LABELS[b] + " AUROC")
            if row == 0:
                ax.set_title(title, fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save_figure(fig, root, "02_category_scatter")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, kind in zip(axes, ["object", "background"]):
        reps = ["single_feature", "selected_five_positive", "selected_five", "all_sae"]
        data = [
            [
                r["auroc"]
                for r in rows
                if r["representation"] == rep
                and r["kind"] == kind
                and r["trained_on"] == "removal"
                and r["evaluated_on"] == "removal"
                and r["auroc"] is not None
                and r["category_id"] in common["removal", "removal"]
            ]
            for rep in reps
        ]
        ax.boxplot(
            data, tick_labels=[LABELS[r].replace(" ", "\n", 1) for r in reps], whis=(0, 100), showfliers=False
        )
        ax.axhline(0.5, color="#666666", lw=0.8, ls="--")
        ax.set(title="물체" if kind == "object" else "배경", ylabel="원본·가림 구별 AUROC", ylim=(0, 1.04))
    fig.tight_layout()
    save_figure(fig, root, "03_selection_and_sign_constraint")
    lines = [
        "# 이미지 표현 비교 결과",
        "",
        "동일한 분할과 범주를 사용한 선형 분류기의 AUROC를 비교했다.",
        "",
        "| 학습과 평가 | 표현 | 범주 수 | 평균 AUROC | 중앙값 AUROC |",
        "|---|---|---:|---:|---:|",
    ]
    task_labels = {
        ("presence", "presence"): "원본으로 학습하고 범주 존재를 평가했다.",
        ("removal", "removal"): "가림으로 학습하고 원본·가림을 구별했다.",
        ("presence", "removal"): "원본으로 학습한 존재 점수로 원본·가림을 구별했다.",
    }
    for r in summary:
        if r["kind"] == "all":
            lines.append(
                f"| {task_labels[r['trained_on'], r['evaluated_on']]} | {LABELS[r['representation']]} | "
                f"{r['categories']} | {r['mean']:.4f} | {r['median']:.4f} |"
            )
    lines += [
        "",
        "평균은 범주별 동일 비중의 기술 통계다. 범주별 신뢰구간과 표현 간 차이의 신뢰구간은 CSV에 저장했다.",
        "가림을 직접 학습한 분류기는 흰색 편집 흔적에도 반응할 수 있다. 원본으로 학습한 존재 점수도 주변 정보와 편집의 영향을 완전히 분리하지는 못한다.",
        "모든 비교는 고정한 표현에서 선형적으로 읽을 수 있는 정보를 측정하며, 낮은 AUROC가 정보의 완전한 부재를 뜻하지 않는다.",
        "이전 분석에서도 사용한 val 자료에서 수행한 탐색적 진단이다.",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
