"""Compare completed feature-set results on identical evaluable categories and pairs.

No fitting or inference is performed. This supplements the all-available main report.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mm_sae.analysis.data import save_json
from mm_sae.analysis.evaluation import distribution
from mm_sae.io import write_csv


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(row, key):
    value = row.get(key)
    return float(value) if value not in (None, "", "None") else float("nan")


def paired_concepts(rows, counts):
    summaries, comparisons = [], []
    tasks = ["removal", "removal_score_presence", "concept_presence"]
    for side in ["image", "text"]:
        for task in tasks:
            conditions = ["top_n"] if task == "concept_presence" else ["learned", "equal"]
            for condition in conditions:
                entries = [r for r in rows if r["side"] == side and r["task"] == task
                           and r["condition"] == condition and np.isfinite(number(r, "auroc"))]
                by_n = {n: {int(r["category_id"]): r for r in entries if int(r["n"]) == n}
                        for n in counts}
                common = sorted(set.intersection(*(set(v) for v in by_n.values())))
                for kind in ["all", "object", "background"]:
                    ids = [c for c in common if kind == "all" or by_n[1][c]["kind"] == kind]
                    baseline = np.array([number(by_n[1][c], "auroc") for c in ids])
                    for n in counts:
                        values = np.array([number(by_n[n][c], "auroc") for c in ids])
                        changes = values-baseline
                        summaries.append(dict(side=side, task=task, condition=condition, kind=kind,
                            feature_count=n, categories=len(ids), category_ids=ids,
                            **{f"auroc_{k}": v for k, v in distribution(values).items() if k != "n"},
                            mean_change_from_one=float(np.mean(changes)) if len(changes) else None,
                            improved=int(np.sum(changes > 0)), declined=int(np.sum(changes < 0)),
                            unchanged=int(np.sum(changes == 0)), above_0_7=int(np.sum(values >= .7)),
                            paired_ci_above_zero=sum(number(by_n[n][c], "auroc_change_ci_low") > 0
                                                     for c in ids)))
                if task == "concept_presence":
                    best = {int(r["category_id"]): r for r in rows if r["side"] == side
                            and r["task"] == task and r["condition"] == "best_single"
                            and np.isfinite(number(r, "auroc"))}
                    for n in counts:
                        ids = sorted(set(common) & set(best))
                        diff = [number(by_n[n][c], "auroc")-number(best[c], "auroc") for c in ids]
                        comparisons.append(dict(side=side, feature_count=n, categories=len(ids),
                            mean_multi_auroc=float(np.mean([number(by_n[n][c], "auroc") for c in ids])),
                            mean_best_single_auroc=float(np.mean([number(best[c], "auroc") for c in ids])),
                            mean_difference=float(np.mean(diff)), multi_higher=sum(d > 0 for d in diff)))
    return summaries, comparisons


def paired_shared(rows, counts):
    result = []
    for side in ["image", "text"]:
        by_condition = defaultdict(dict)
        for row in rows:
            if row["side"] == side and np.isfinite(number(row, "auroc")):
                pair = (int(row["category_a"]), int(row["category_b"]))
                by_condition[row["condition"], int(row["n"])][pair] = row
        requested = [("union", n) for n in counts]+[("shared_one", 1), ("best_all", 1)]
        common = sorted(set.intersection(*(set(by_condition[key]) for key in requested)))
        for condition, n in requested:
            selected = [by_condition[condition, n][pair] for pair in common]
            values = [number(r, "auroc") for r in selected]
            diff = [number(r, "auroc")-number(by_condition["best_all", 1][pair], "auroc")
                    for r, pair in zip(selected, common, strict=True)]
            result.append(dict(side=side, condition=condition, feature_count=n, pairs=len(common),
                pair_ids=common, **{f"auroc_{k}": v for k, v in distribution(values).items() if k != "n"},
                mean_difference_from_best_single=float(np.mean(diff)) if diff else None,
                higher_than_best_single=sum(d > 0 for d in diff),
                paired_ci_above_best_single=sum(number(r, "auroc_change_ci_low") > 0 for r in selected)))
    return result


def paired_scores(run, counts):
    distributions, effects, rankings = [], [], []
    for readout in ["learned", "equal"]:
        by_n = {}
        for n in counts:
            entries = read_csv(run / "correspondence" / f"{readout}_{n}_pairs.csv")
            by_n[n] = {(r["direction"], int(r["anchor_id"]), int(r["rival_id"])): r for r in entries}
        common_natural = set.intersection(*(set(k for k, r in rows.items()
            if np.isfinite(number(r, "original_same")) and np.isfinite(number(r, "original_other")))
            for rows in by_n.values()))
        # Score distributions do not require an evaluable diagonal; ranking comparisons do.
        common_control = set.intersection(*(set(k for k, r in rows.items()
            if np.isfinite(number(r, "original_other")) and np.isfinite(number(r, "controlled_other")))
            for rows in by_n.values()))
        for direction in ["image_to_text", "text_to_image"]:
            # Restrict this sensitivity analysis to one identical rival set at every n.
            natural = sorted(k for k in common_natural if k[0] == direction)
            for n in counts:
                rows = [by_n[n][k] for k in natural]
                high = [r for r in rows if number(r, "original_other") > number(r, "original_same")]
                anchors = {int(r["anchor_id"]) for r in rows}
                high_anchors = {int(r["anchor_id"]) for r in high}
                rankings.append(dict(readout=readout, feature_count=n, direction=direction,
                    comparison="same_evaluable_rivals_at_every_n", pairs=len(rows), higher_pairs=len(high),
                    pair_percent=100*len(high)/len(rows) if rows else None, anchors=len(anchors),
                    anchors_with_higher=len(high_anchors),
                    anchor_percent=100*len(high_anchors)/len(anchors) if anchors else None))
            keys = sorted(k for k in common_control if k[0] == direction)
            for lower, upper in zip([0., .2, .4, .6, .8], [.2, .4, .6, .8, 1.00001], strict=True):
                selected = [k for k in keys if lower <= number(by_n[1][k], "annotation_correlation") < upper]
                for n in counts:
                    rows = [by_n[n][k] for k in selected]
                    before = np.array([number(r, "original_other") for r in rows])
                    after = np.array([number(r, "controlled_other") for r in rows])
                    meta = dict(readout=readout, direction=direction, feature_count=n,
                                lower=lower, upper=min(upper, 1.), pairs=len(rows))
                    for condition, values in [("original", before), ("controlled", after)]:
                        distributions.append(dict(**meta, condition=condition, **distribution(values)))
                    effects.append(dict(**meta, mean_change=float(np.mean(after-before)) if len(rows) else None,
                        reduced=int(np.sum(after < before)), increased=int(np.sum(after > before)),
                        unchanged=int(np.sum(after == before))))
    return distributions, effects, rankings


def figures(out, concepts, shared, counts):
    colors = ["#D48148", "#2D9EA7", "#6870BA"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, side in enumerate(["image", "text"]):
        for row, (task, label) in enumerate([
            ("removal", "원본과 가림 구별"), ("removal_score_presence", "같은 점수의 원본 개념 검출")]):
            ax = axes[row, col]
            for color, kind, name in zip(colors, ["all", "object", "background"],
                                          ["전체", "물체", "배경"], strict=True):
                entries = [r for r in concepts if r["side"] == side and r["task"] == task
                           and r["condition"] == "learned" and r["kind"] == kind]
                ax.plot(counts, [next(r.get("auroc_mean", np.nan) for r in entries if r["feature_count"] == n)
                                 for n in counts], "o-", color=color,
                        label=f"{name} ({entries[0]['categories']}개)")
            ax.axhline(.5, color="#AAAAAA", linestyle=":")
            ax.set(title=("이미지" if side == "image" else "텍스트")+" · "+label,
                   xlabel="사용할 특징 수", ylabel="동일 범주 집합의 평균 AUROC", xticks=counts, ylim=(0, 1))
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, fontsize=9)
    fig.suptitle("특징 개수별로 모두 평가 가능한 같은 범주의 비교")
    save_figure(fig, out, "common_category_auroc")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, side in zip(axes, ["image", "text"], strict=True):
        entries = [r for r in shared if r["side"] == side]
        for color, condition, name in zip(colors, ["union", "shared_one", "best_all"],
            ["두 범주의 특징 합집합", "공유된 단일 특징", "전체에서 고른 최선의 단일 특징"], strict=True):
            values = [next(r.get("auroc_mean", np.nan) for r in entries if r["condition"] == condition
                          and r["feature_count"] == (n if condition == "union" else 1)) for n in counts]
            ax.plot(counts, values, "o-", color=color, label=name)
        ax.set(title=("이미지" if side == "image" else "텍스트")+f" · 동일 {entries[0]['pairs']}쌍",
               xlabel="범주별 특징 수 · 합집합은 최대 두 배", ylabel="평균 AUROC", xticks=counts, ylim=(0, 1))
        ax.axhline(.5, color="#AAAAAA", linestyle=":")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("같은 1위 특징을 공유하는 범주를 직접 구별하는 성능")
    save_figure(fig, out, "common_pair_discrimination")


def save_figure(fig, out, name):
    fig.tight_layout()
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(out / f"{name}.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def ranking_figure(out, rankings, counts):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, direction in enumerate(["image_to_text", "text_to_image"]):
        rows = [r for r in rankings if r["direction"] == direction and r["readout"] == "learned"]
        for row, metric, ylabel, denominator in [
            (0, "pair_percent", "같은 범주의 점수를 넘는 다른 범주 쌍 (%)", "pairs"),
            (1, "anchor_percent", "더 높은 다른 범주가 하나라도 있는 기준 범주 (%)", "anchors")]:
            ax = axes[row, col]
            ax.plot(counts, [next(r[metric] for r in rows if r["feature_count"] == n) for n in counts],
                    "o-", color="#2D9EA7")
            ax.set(title=("이미지 기준" if col == 0 else "텍스트 기준")+
                   f" · 같은 {rows[0][denominator]:,}개 비교 대상",
                   xlabel="사용할 특징 수", ylabel=ylabel, xticks=counts, ylim=(0, 100 if row else 10))
            if row == 0:
                maximum = max(r[metric] for r in rows if r[metric] is not None)
                ax.set_ylim(0, max(10, maximum*1.15))
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("RQ1 · 모든 특징 개수에서 같은 평가 가능한 경쟁 상대의 비교")
    save_figure(fig, out, "common_rq1_ranking")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--font", default="NanumGothic")
    args = parser.parse_args()
    run = args.run.resolve()
    out = run / "comparison"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": args.font, "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "svg.fonttype": "path", "font.size": 11})
    concept_rows = read_csv(run / "report" / "concepts.csv")
    counts = sorted({int(r["n"]) for r in concept_rows})
    concepts, best = paired_concepts(concept_rows, counts)
    shared = paired_shared(read_csv(run / "report" / "shared_pairs.csv"), counts)
    distributions, effects, rankings = paired_scores(run, counts)
    tables = dict(concepts=concepts, concepts_vs_best_single=best, shared_pairs=shared,
                  rq1_distributions=distributions, rq1_control_effects=effects, rq1_rankings=rankings)
    for name, rows in tables.items():
        write_csv(out / f"{name}.csv", rows)
    save_json(out / "summary.json", tables)
    figures(out, concepts, shared, counts)
    ranking_figure(out, rankings, counts)
    (out / "README.md").write_text(
        "# 같은 평가 대상을 유지한 추가 비교\n\n"
        "특징 개수에 따라 계산 가능한 범주나 범주 쌍이 달라지는 영향을 제외했다. "
        "모든 특징 개수에서 평가 가능한 같은 범주와 범주 쌍을 사용했다. 모델이나 점수를 다시 학습하지 않았다.\n\n"
        "RQ1의 순위 표는 모든 특징 개수에서 계산 가능한 같은 경쟁 상대 집합만 비교한다. "
        "이는 전체 171개 범주를 평가한 본 분석을 대체하지 않는 보조 비교다. "
        "상관점수 통제 전후 표도 모든 특징 개수에서 두 조건이 모두 계산 가능한 같은 쌍을 사용했다.\n\n"
        "곡선은 범주 또는 범주 쌍의 기술 통계다. 개별 비교의 신뢰구간은 원래 보고서 CSV에 있다. "
        "유의한 개수는 다중 비교를 보정하지 않은 개별 95% 구간의 개수다.\n\n"
        f"원본 결과의 경로는 {run}다.\n"
    )
    print(json.dumps(dict(output=str(out), feature_counts=counts), ensure_ascii=False))


if __name__ == "__main__":
    main()
