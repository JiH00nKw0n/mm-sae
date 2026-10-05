"""Scientific plots and machine-readable tables for the three configured studies."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mm_sae.analysis.data import save_json
from mm_sae.io import write_csv


def records(folder):
    result = []
    for path in sorted(folder.glob("*.json")):
        value = json.loads(path.read_text())
        result.extend(value.get("records", []))
    return result


def finish(fig, out, name):
    fig.tight_layout()
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(out / f"{name}.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def curve(ax, rows, conditions, metric, counts):
    colors = ["#D48148", "#2D9EA7", "#6870BA", "#8A8A8A"]
    for (condition, label), color in zip(conditions, colors):
        values = []
        for n in counts:
            selected = [r[metric] for r in rows if r.get("condition") == condition and r.get("n") == n
                        and r.get(metric) is not None and np.isfinite(r[metric])]
            values.append(float(np.mean(selected)) if selected else np.nan)
        if condition in {"best_single", "best_all", "shared_one"}:
            vals = [r[metric] for r in rows if r.get("condition") == condition and r.get(metric) is not None]
            if vals:
                values = [float(np.mean(vals))]*len(counts)
        ax.plot(counts, values, "o-", color=color, label=label)
    ax.set(xticks=counts, xlabel="사용할 특징 수", ylabel=metric)
    if metric == "auroc":
        ax.set(ylabel="AUROC", ylim=(0, 1))
        ax.axhline(.5, color="#BBBBBB", linestyle=":", linewidth=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)


def boxes(ax, entries, kinds, colors):
    for index, lower in enumerate([0., .2, .4, .6, .8]):
        for j, (selector, label) in enumerate(kinds):
            selected = [r for r in entries if abs(r["left"]-lower) < 1e-7 and selector(r) and r.get("n", 0)]
            if not selected:
                continue
            r = selected[0]
            position = index+(j-(len(kinds)-1)/2)*.3
            ax.bxp([dict(q1=r["q1"], med=r["median"], q3=r["q3"], whislo=r["min"],
                         whishi=r["max"], fliers=[])], positions=[position], widths=.25,
                   manage_ticks=False, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor=colors[j]), medianprops=dict(color="#263540"))
            ax.scatter(position, r["mean"], s=15, marker="D", color="#263540", zorder=4)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=color, label=label) for (_, label), color in zip(kinds, colors)],
              frameon=False, fontsize=9)
    ax.set(xticks=range(5), xticklabels=["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1"],
           xlabel="원본 주석 상관 구간", ylabel="범주 점수의 Pearson 상관", ylim=(-.2, 1))
    ax.spines[["top", "right"]].set_visible(False)


def report(ctx):
    out = ctx.out / "report"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": ctx.o["report"]["font"], "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "svg.fonttype": "path", "font.size": 11})
    concepts = records(ctx.out / "concepts")
    raw = records(ctx.out / "raw_prediction")
    pairs = records(ctx.out / "shared_pairs")
    interventions = records(ctx.out / "intervention")
    summaries, distributions = [], []
    for path in (ctx.out / "correspondence").glob("*.json"):
        r = json.loads(path.read_text())
        summaries.extend([dict(readout=r["condition"], feature_count=r["n"], **v) for v in r["summary"]])
        distributions.extend([dict(readout=r["condition"], feature_count=r["n"], **v) for v in r["distributions"]])
    for name, rows in [("concepts", concepts), ("raw_prediction", raw), ("shared_pairs", pairs),
                       ("intervention_connections", interventions), ("rq1_summary", summaries),
                       ("rq1_distributions", distributions)]:
        write_csv(out / f"{name}.csv", rows)
    counts = ctx.counts
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, side in zip(axes, ["image", "text"], strict=True):
        curve(ax, [r for r in concepts if r["side"] == side and r["task"] == "concept_presence"],
              [("top_n", "가림 AUROC 상위 특징"), ("best_single", "별도로 고른 최선의 단일 특징")],
              "auroc", counts)
        ax.set_title("이미지의 개념 검출" if side == "image" else "텍스트의 개념 검출")
    fig.suptitle("원래 RQ2 · 특징 수에 따른 원본 개념 검출 AUROC")
    finish(fig, out, "01_original_concept_detection")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, direction in enumerate(["image_to_text", "text_to_image"]):
        subset = [r for r in raw if r["direction"] == direction]
        for row, metric in enumerate(["r2_mean", "decoder_contribution_relative_mse"]):
            curve(axes[row, col], subset, [("top_n", "같은 범주 특징"), ("cooccurring", "동시 등장 범주 특징")],
                  metric, counts)
            axes[row, col].set_title("이미지로 텍스트 예측" if col == 0 else "텍스트로 이미지 예측")
    fig.suptitle("원래 RQ2 · 고정 목표의 원본 활성값 예측")
    finish(fig, out, "02_original_cross_modal_prediction")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, side in zip(axes, ["image", "text"], strict=True):
        curve(ax, [r for r in pairs if r["side"] == side],
              [("union", "두 범주의 특징 합집합"), ("shared_one", "공유된 단일 특징"),
               ("best_all", "전체에서 고른 단일 특징")], "auroc", counts)
        ax.set_title(("이미지" if side == "image" else "텍스트")+" · A만 있는 표본과 B만 있는 표본")
        ax.set_xlabel("범주마다 고른 특징 수 · 합집합은 최대 두 배")
    fig.suptitle("수정 RQ2 · 같은 1위 특징을 공유하는 범주의 직접 구별")
    finish(fig, out, "03_shared_feature_discrimination")
    names = ["one_to_one", "reusable_one"]+[f"reusable_{n}" for n in counts if n > 1]
    labels = ["1대1", "입력 하나 재사용"]+[f"입력 최대 {n}개" for n in counts if n > 1]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for col, direction in enumerate(["image_to_text", "text_to_image"]):
        for row, metric in enumerate(["r2_mean", "normalized_mse"]):
            ax = axes[row, col]
            for condition, color, label in [("aligned", "#D48148", "원래 짝"), ("shuffled", "#2D9EA7", "같은 범주에서 짝 섞음")]:
                vals = [next((r[metric] for r in interventions if r["direction"] == direction and
                              r["condition"] == condition and r["model"] == name), np.nan) for name in names]
                ax.plot(range(len(names)), vals, "o-", color=color, label=label)
            ax.set(xticks=range(len(names)), xticklabels=labels, ylabel=metric,
                   title="이미지로 텍스트 변화 예측" if col == 0 else "텍스트로 이미지 변화 예측")
            ax.tick_params(axis="x", rotation=20, labelsize=8)
            ax.legend(frameon=False, fontsize=9)
    fig.suptitle("수정 RQ2 · 연결 제약에 따른 가림 반응 예측")
    finish(fig, out, "04_intervention_connection_constraints")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    quality = []
    for col, side in enumerate(["image", "text"]):
        for row, task in enumerate(["removal", "removal_score_presence"]):
            curve(axes[row, col], [r for r in concepts if r["side"] == side and r["task"] == task],
                  [("learned", "학습한 가중합"), ("equal", "동일 비중 합산")], "auroc", counts)
            axes[row, col].set_title(("이미지" if side == "image" else "텍스트")+
                                     (" · 원본과 가림 구별" if row == 0 else " · 같은 점수의 원본 개념 구별"))
        for n in counts:
            for kind in ["object", "background"]:
                selected = [r for r in concepts if r["side"] == side and r["kind"] == kind and
                            r["task"] == "removal" and r["condition"] == "learned" and r["n"] == n]
                for cutoff in ctx.o["multi_feature_rq1"]["descriptive_auroc_thresholds"]:
                    quality.append(dict(side=side, kind=kind, n=n, cutoff=cutoff,
                        evaluated=sum(r.get("auroc") is not None for r in selected),
                        above=sum(r.get("auroc") is not None and r["auroc"] >= cutoff for r in selected)))
    write_csv(out / "removal_quality_counts.csv", quality)
    fig.suptitle("확장 RQ1 · 특징 수에 따른 가림 구별과 원본 개념 구별")
    finish(fig, out, "05_multi_feature_auroc")
    compare_n = 5 if 5 in counts else max(counts)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, direction in zip(axes, ["image_to_text", "text_to_image"], strict=True):
        rows = [r for r in distributions if r["readout"] == "learned" and
                r["direction"] == direction and r["condition"] == "original"]
        boxes(ax, rows, [(lambda r: r["feature_count"] == 1, "특징 1개"),
                         (lambda r: r["feature_count"] == compare_n, f"특징 {compare_n}개")],
              ["#FFADAD", "#9BF6FF"])
        ax.set_title("이미지 기준" if direction == "image_to_text" else "텍스트 기준")
    fig.suptitle("확장 RQ1 · 주석 상관에 따른 점수 분포 · 수염은 최솟값과 최댓값")
    finish(fig, out, "06_rq1_score_distribution")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, direction in zip(axes, ["image_to_text", "text_to_image"], strict=True):
        rows = [dict(r, n=r["feature_count"], condition=r["readout"]) for r in summaries
                if r["direction"] == direction and r["group"] == "all" and r["condition"] == "original"]
        curve(ax, rows, [("learned", "학습한 가중합"), ("equal", "동일 비중 합산")],
              "category_percent_lower_bound", counts)
        ax.set_title("이미지 기준" if direction == "image_to_text" else "텍스트 기준")
        ax.set_ylabel("다른 범주가 더 높은 범주의 비율 · 하한 (%)")
    fig.suptitle("확장 RQ1 · 대각을 넘는 다른 범주의 존재 · 계산 불가는 별도 표기")
    finish(fig, out, "07_rq1_above_diagonal")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, direction in enumerate(["image_to_text", "text_to_image"]):
        for row, n in enumerate([1, compare_n]):
            rows = [r for r in distributions if r["readout"] == "learned" and
                    r["direction"] == direction and r["feature_count"] == n]
            boxes(axes[row, col], rows, [(lambda r: r["condition"] == "original", "원본"),
                                        (lambda r: r["condition"] == "controlled", "양의 동시 등장 통제")],
                  ["#FFADAD", "#9BF6FF"])
            axes[row, col].set_title(("이미지 기준" if col == 0 else "텍스트 기준")+f" · 특징 {n}개")
    fig.suptitle("확장 RQ1 · 같은 특징 결합 점수의 동시 등장 통제 전후")
    finish(fig, out, "08_rq1_decorrelation")
    save_json(out / "summary.json", dict(quality_counts=quality, rankings=summaries,
                                         intervention=interventions))
    text = ["# 여러 특징을 사용하는 RQ2와 RQ1의 결과",
            "",
            "기존 SAE와 원본·가림 활성값을 고정하고 선형 분류기와 회귀 예측기만 학습했다. "
            "모든 그림의 범주별 평균은 기술 통계다. 개별 범주·범주 쌍 AUROC와 예측오차의 이미지 단위 "
            "재표집 구간은 CSV에 있다. 평균 곡선 자체의 신뢰구간으로 개별 구간을 평균내지 않았다.",
            "",
            f"입력 캐시 경로는 {ctx.data.source.resolve()}다. "
            f"실행 설정과 분할 목록, 특징 번호와 계수는 {ctx.out.resolve()}에 저장했다.",
            "",
            "이미지 ID로 train을 학습용·조정용으로 분할하고 val에서 평가했다. val은 이전에 관찰한 자료다. "
            "캡션 정답은 고정된 사전의 검출이며 사람 검수 정답이 아니다.",
            "",
            "| 모달리티 | 범주 유형 | 특징 수 | 평가 가능 범주 수 | 가림 AUROC 0.7 이상 범주 수 |",
            "|---|---|---:|---:|---:|"]
    for r in quality:
        if r["cutoff"] == .7:
            text.append(f"| {r['side']} | {r['kind']} | {r['n']} | {r['evaluated']} | {r['above']} |")
    text += ["", "대각 초과 비율에는 계산 불가능한 경쟁 상대가 있을 수 있다. "
             "그 경우 범주 비율은 확인된 오류의 하한이며, 미해결 범주 수를 rq1_summary.csv에 함께 저장했다. "
             "재가중 전후 분포는 같은 계산 가능 범주 쌍으로 비교했다.",
             "", "여러 특징의 상관은 개별 SAE 좌표의 상관이 아니라, 라벨을 사용해 학습한 범주 점수의 상관이다. "
             "원본·가림 AUROC 상승만으로 서로 다른 의미를 더 잘 구별한다고 해석하지 않는다.",
             "", "세 실험의 결과와 모든 대조 조건은 CSV에 포함했다. 원래 RQ1의 train 점수 재현 여부는 "
             "상위 실행 폴더의 legacy_reproduction.json에 기록했다."]
    (out / "README.md").write_text("\n".join(text)+"\n")
