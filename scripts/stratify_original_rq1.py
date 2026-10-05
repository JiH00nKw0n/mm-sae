"""Stratify the original three RQ1 analyses without changing their scores or representatives."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mm_sae.io import atomic_json, sha256, write_csv
from scripts.analyze_bidirectional_rq1 import build_directions, in_bin, ranking_summary, read_csv

GROUPS = ["all", "below_0.5", "0.5_0.6", "0.6_0.7", "0.7_0.8", "0.8_0.9", "0.9_1.0", "missing"]
LABELS = ["전체", "0.5 미만", "0.5–0.6", "0.6–0.7", "0.7–0.8", "0.8–0.9", "0.9–1.0", "계산 불가"]
EDGES = np.round(np.linspace(0, 1, 6), 10)


def quality_group(value):
    if value is None:
        return "missing"
    if value < .5:
        return "below_0.5"
    for lower, upper in zip([.5, .6, .7, .8, .9], [.6, .7, .8, .9, 1.01], strict=True):
        if lower <= value < upper:
            return f"{lower:.1f}_{min(upper, 1):.1f}"
    raise ValueError(value)


def above(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def distribution(values):
    keys = ["mean", "std", "min", "q1", "median", "q3", "max"]
    if not len(values):
        return dict.fromkeys(keys)
    return dict(zip(keys, map(float, [values.mean(), values.std(), values.min(),
                                     *np.quantile(values, [.25, .5, .75]), values.max()]), strict=True))


def save(fig, out, name):
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(out / f"{name}.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def draw_distributions(stats, direction, out, controlled):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), sharey=True)
    for ax, group, label in zip(axes.flat, GROUPS, LABELS, strict=True):
        for index, lower in enumerate(EDGES[:-1]):
            selected = [r for r in stats if r["direction"] == direction and r["quality_group"] == group
                        and r["annotation_left"] == lower]
            for condition in (["original", "controlled"] if controlled else ["original"]):
                r = next(r for r in selected if r["condition"] == condition)
                if not r["pairs"]:
                    continue
                offset = (-.17 if condition == "original" else .17) if controlled else 0
                color = "#FFADAD" if condition == "original" else "#9BF6FF"
                ax.bxp([dict(q1=r["q1"], med=r["median"], q3=r["q3"],
                             whislo=r["min"], whishi=r["max"], fliers=[])],
                       positions=[index+offset], widths=.28 if controlled else .5,
                       manage_ticks=False, showfliers=False, patch_artist=True,
                       boxprops={"facecolor": color}, medianprops={"color": "#263540"})
                ax.scatter(index+offset, r["mean"], color="#263540", marker="D", s=12, zorder=5)
        originals = [r for r in stats if r["direction"] == direction and r["quality_group"] == group
                     and r["condition"] == "original"]
        ax.set(title=f"AUROC {label}", xticks=range(5),
               xticklabels=[f"{r['annotation_left']:.1f}–{r['annotation_right']:.1f}\nn={r['pairs']}" for r in originals],
               ylim=(-.1, 1), xlim=(-.6, 4.6))
        ax.tick_params(axis="x", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    title = "동시 등장 통제 전후의 기존 상관점수" if controlled else "동시 등장 정도에 따른 기존 상관점수"
    side = "이미지" if direction == "image" else "텍스트"
    fig.suptitle(f"{title} · {side} 범주 기준", fontsize=16)
    fig.supxlabel("원본 이미지 주석 상관 구간 · 상자는 사분위수, 수염은 최솟값과 최댓값, 마름모는 평균", y=.065, fontsize=11)
    fig.supylabel("서로 다른 범주의 coactivation correlation", fontsize=11)
    note = "분홍은 원본, 파랑은 양의 동시 등장 관계 통제 후입니다. " if controlled else ""
    fig.text(.06, .025, note + "기준 범주의 이미지·텍스트 검증 AUROC 중 작은 값으로 구분했습니다. 비교 상대는 170개로 유지했습니다.", fontsize=10)
    fig.tight_layout(rect=(.02, .12, 1, .95))
    save(fig, out, f"{'03_controlled' if controlled else '01_original'}_{direction}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--font", default="NanumGothic")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": args.font, "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "svg.fonttype": "path", "font.size": 10})
    directions = build_directions(read_csv(args.source))
    reps_path = args.run / "representatives.csv"
    auc = {(int(r["concept_id"]), r["side"]): float(r["auroc"]) if r["auroc"] else None
           for r in read_csv(reps_path) if r["split"] == "val2017"}
    ids = sorted({r.anchor_id for r in directions["image"]})
    quality = {}
    for cid in ids:
        image_auc, text_auc = auc[cid, "image"], auc[cid, "text"]
        quality[cid] = min(image_auc, text_auc) if image_auc is not None and text_auc is not None else None
    groups = {cid: quality_group(value) for cid, value in quality.items()}
    categories = [{"category_id": cid, "min_val_auroc": quality[cid], "quality_group": groups[cid],
                   "image_val_auroc": auc[cid, "image"], "text_val_auroc": auc[cid, "text"]} for cid in ids]
    ranking, stats, effects = [], [], []
    for direction, rows in directions.items():
        assert len(rows) == len(ids)*(len(ids)-1)
        baseline = ranking_summary(rows)
        assert baseline["exceeding_pairs"] == {"image": 2399, "text": 2089}[direction]
        for group in GROUPS + ["ge_0.6", "ge_0.7"]:
            selected = rows if group == "all" else [r for r in rows if (
                above(quality[r.anchor_id], float(group[3:]))
                if group.startswith("ge_") else groups[r.anchor_id] == group)]
            record = {"direction": direction, "quality_group": group, **ranking_summary(selected)}
            record["object_categories"] = len({r.anchor_id for r in selected if r.anchor_type == "object"})
            record["background_categories"] = len({r.anchor_id for r in selected if r.anchor_type == "background"})
            ranking.append(record)
            assert len(selected) == record["eligible_categories"] * 170
            for lower, upper in zip(EDGES[:-1], EDGES[1:], strict=True):
                cell = [r for r in selected if in_bin(r, lower, upper)]
                before = np.array([r.original_other for r in cell])
                after = np.array([r.controlled_other for r in cell])
                base = {"direction": direction, "quality_group": group,
                        "annotation_left": float(lower), "annotation_right": float(upper), "pairs": len(cell)}
                for condition, values in [("original", before), ("controlled", after)]:
                    stats.append({**base, "condition": condition, **distribution(values)})
                effects.append({**base, "mean_change": float((after-before).mean()) if len(cell) else None,
                                "decreased_pairs": int(np.sum(after < before)),
                                "increased_pairs": int(np.sum(after > before))})
    for direction in directions:
        pieces = [r for r in ranking if r["direction"] == direction and r["quality_group"] in GROUPS[1:]]
        assert sum(r["eligible_categories"] for r in pieces) == 171
        assert sum(r["pairs"] for r in pieces) == 29070
        assert sum(r["exceeding_pairs"] for r in pieces) == {"image": 2399, "text": 2089}[direction]
    write_csv(args.out / "categories.csv", categories)
    write_csv(args.out / "ranking.csv", ranking)
    write_csv(args.out / "distributions.csv", stats)
    write_csv(args.out / "paired_changes.csv", effects)
    for direction in directions:
        for controlled in [False, True]:
            draw_distributions(stats, direction, args.out, controlled)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, direction in zip(axes, directions, strict=True):
        rows = [r for r in ranking if r["direction"] == direction and r["quality_group"] in GROUPS]
        bars = ax.bar(range(len(rows)), [r["category_percent"] for r in rows],
                      color=["#FFD6A5"]+["#FFADAD"]*2+["#CAFFBF"]*4+["#DDDDDD"])
        for bar, r in zip(bars, rows, strict=True):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                    f"{r['categories_with_higher_alternative']}/{r['eligible_categories']}", ha="center", fontsize=9)
        ax.set(xticks=range(len(rows)), xticklabels=LABELS, ylim=(0, 105),
               title="이미지 기준" if direction == "image" else "텍스트 기준",
               ylabel="다른 범주가 하나라도 더 높은 기준 범주 (%)")
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("기존 coactivation correlation · 기준 범주의 AUROC 구간별 비교")
    fig.tight_layout(rect=(0, .03, 1, .95))
    save(fig, args.out, "02_original_ranking")
    atomic_json(args.out / "manifest.json", {"hostname": socket.gethostname(), "regression_used": False,
        "quality": "Minimum of fixed anchor category image/text AUROC on val2017; candidates remain all170",
        "control": "Previously computed pair-specific reweighting zeroes positive image-label correlation only",
        "source_sha256": sha256(args.source), "representatives_sha256": sha256(reps_path),
        "script_sha256": sha256(Path(__file__)), "source": args.source, "run": args.run,
        "verified_original_pair_counts": {"image": 2399, "text": 2089}})
    print(json.dumps(ranking, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
