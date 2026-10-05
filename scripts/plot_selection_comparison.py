"""Export standalone scientific figures from the common-cohort summary.

This consumes recorded distribution statistics only. It does not fit models,
change cohorts, clip distribution tails, or replace minimum/maximum whiskers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

METHODS = ["pooled_auroc", "paired_mean_drop", "single_logistic", "probe_attribution"]
LABELS = ["원본·가림\nAUROC", "평균 활성값\n감소량", "단일 좌표\n분류 손실", "개념 분류기\n기여도"]
COLORS = ["#FFADAD", "#FFD6A5", "#CAFFBF", "#9BF6FF"]
ONE, FIVE = COLORS[0], COLORS[-1]
DIRECTIONS = [("image_to_text", "이미지 범주를 기준으로 비교"),
              ("text_to_image", "텍스트 범주를 기준으로 비교")]


class Figures:
    def __init__(self, data, output):
        self.data, self.output = data, output
        self.files = []
        output.mkdir(parents=True, exist_ok=True)

    def one(self, name, **conditions):
        rows = [r for r in self.data[name] if all(r.get(k) == v for k, v in conditions.items())]
        if len(rows) != 1:
            raise ValueError((name, conditions, len(rows)))
        return rows[0]

    @staticmethod
    def boxes(ax, rows, positions, colors, width=.22):
        selected = [(r, p, c) for r, p, c in zip(rows, positions, colors, strict=True) if r["n"]]
        if not selected:
            ax.text(.5, .5, "공통으로 평가 가능한 자료 없음", ha="center", transform=ax.transAxes)
            return
        specs = [dict(q1=r["q1"], med=r["median"], q3=r["q3"],
                      whislo=r["min"], whishi=r["max"], mean=r["mean"], fliers=[])
                 for r, _, _ in selected]
        artists = ax.bxp(specs, positions=[p for _, p, _ in selected], widths=width,
                         patch_artist=True, manage_ticks=False, showmeans=True, showfliers=False,
                         meanprops=dict(marker="D", markersize=3, markerfacecolor="#252525", markeredgecolor="#252525"),
                         medianprops=dict(color="#252525", linewidth=1.4),
                         whiskerprops=dict(color="#67717A"), capprops=dict(color="#67717A"))
        for box, (_, _, color) in zip(artists["boxes"], selected, strict=True):
            box.set(facecolor=color, edgecolor="#67717A")

    @staticmethod
    def style(ax, labels, ylabel="AUROC", limits=(0, 1)):
        ax.set_xticks(np.arange(len(labels)), labels)
        ax.set_ylabel(ylabel)
        if limits is not None:
            ax.set_ylim(*limits)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)

    def save(self, fig, name, title, legend=(), boxes=True):
        fig.suptitle(title, fontsize=15)
        if legend:
            fig.legend(handles=[Patch(facecolor=c, edgecolor="#67717A", label=t) for t, c in legend],
                       loc="lower center", bbox_to_anchor=(.5, .06), ncol=len(legend), frameon=False)
        if boxes:
            fig.text(.5, .018, "상자 25–75% · 가로선 중앙값 · 수염 최솟값–최댓값 · 마름모 평균",
                     ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .15, 1, .94))
        for extension in ["png", "svg", "pdf"]:
            path = self.output / f"{name}.{extension}"
            fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
            self.files.append(path.name)
        plt.close(fig)

    def concept_figures(self):
        for task, name, title in [("concept_presence", "concept_presence", "원본의 개념 존재 구별"),
                                  ("removal", "original_masked", "선택한 특징의 원본·가림 구별")]:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
            for ax, side, label in zip(axes, ["image", "text"], ["이미지", "텍스트"], strict=True):
                for n, offset, color in [(1, -.15, ONE), (5, .15, FIVE)]:
                    rows = [self.one("concepts", method=m, side=side, task=task, features=n, kind="all") for m in METHODS]
                    self.boxes(ax, rows, np.arange(4)+offset, [color]*4)
                self.style(ax, LABELS)
                ax.set_title(f"{label} · 동일 {rows[0]['n']}개 범주")
            self.save(fig, name, title, [("특징 1개", ONE), ("특징 5개", FIVE)])
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for ax, kind, label in zip(axes, ["object", "background"], ["물체", "배경"], strict=True):
            for n, offset, color in [(1, -.15, ONE), (5, .15, FIVE)]:
                rows = [self.one("concepts", method=m, side="image", task="concept_presence", features=n, kind=kind) for m in METHODS]
                self.boxes(ax, rows, np.arange(4)+offset, [color]*4)
            self.style(ax, LABELS)
            ax.set_title(f"이미지 {label} · 동일 {rows[0]['n']}개 범주")
        self.save(fig, "image_object_background", "물체와 배경의 개념 구별 변화", [("특징 1개", ONE), ("특징 5개", FIVE)])

    def rq1_figures(self):
        bins = [0., .2, .4, .6, .8]
        for split, n, label in [("train", 1, "학습 자료 전체 · 특징 1개"), ("val", 5, "평가 자료 · 특징 5개")]:
            common = dict(features=n, direction="image_to_text", types="all")
            fig, ax = plt.subplots(figsize=(12, 5.5))
            for i, m in enumerate(METHODS):
                rows = [self.one(split+"_distributions", method=m, cohort="original_available", condition="original", lower=b, **common) for b in bins]
                self.boxes(ax, rows, np.arange(5)+(i-1.5)*.19, [COLORS[i]]*5, .16)
            self.style(ax, [f"{b:.1f}–{b+.2:.1f}\n{r['n']:,}쌍" for b, r in zip(bins, rows, strict=True)], "활성값 상관계수", None)
            ax.set_xlabel("원래 이미지 주석의 상관계수 구간")
            self.save(fig, f"{split}_coactivation", label+"의 다른 범주 간 상관점수", list(zip([s.replace('\n', ' ') for s in LABELS], COLORS, strict=True)))

            fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharey=True)
            for ax, m, title in zip(axes.flat, METHODS, LABELS, strict=True):
                for condition, offset, color in [("original", -.15, ONE), ("controlled", .15, FIVE)]:
                    rows = [self.one(split+"_distributions", method=m, cohort="control_available", condition=condition, lower=b, **common) for b in bins]
                    self.boxes(ax, rows, np.arange(5)+offset, [color]*5)
                self.style(ax, [f"{b:.1f}–{b+.2:.1f}\n{r['n']:,}쌍" for b, r in zip(bins, rows, strict=True)], "활성값 상관계수", None)
                ax.set_title(title.replace("\n", " "))
            self.save(fig, f"{split}_decorrelation", label+"의 동시 등장 통제", [("원본", ONE), ("양의 주석 상관을 0으로 통제", FIVE)])

            for measure, stem, ylab in [("pair_percent", "higher_pairs", "같은 범주 점수를 넘은 쌍 (%)"),
                                         ("anchor_percent", "higher_anchors", "더 높은 상대가 있는 범주 (%)")]:
                fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
                ns = [1] if split == "train" else [1, 5]
                all_values = []
                for ax, (direction, title) in zip(axes, DIRECTIONS, strict=True):
                    for k, color in zip(ns, [ONE, FIVE], strict=False):
                        rows = [self.one(split+"_rankings", method=m, direction=direction, features=k, types="all", lower=None) for m in METHODS]
                        values = [r[measure] if measure == "pair_percent" else 100*r["higher_anchors"]/r["anchors"] for r in rows]
                        all_values.extend(values)
                        pos = np.arange(4)+(0 if len(ns) == 1 else (-.16 if k == 1 else .16))
                        bars = ax.bar(pos, values, width=.28, color=COLORS if len(ns) == 1 else color, edgecolor="#67717A")
                        ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
                    self.style(ax, LABELS, ylab, None)
                    count = f"{rows[0]['pairs']:,}쌍" if measure == "pair_percent" else f"{rows[0]['anchors']}개 범주"
                    ax.set_title(title+"\n"+count)
                axes[0].set_ylim(0, 100 if measure == "anchor_percent" else max(5, max(all_values)*1.2))
                rank_label = label if split == "train" else "평가 자료 · 특징 1개와 5개"
                self.save(fig, f"{split}_{stem}", "동일 범주보다 높은 다른 범주의 점수 · "+rank_label,
                          [("특징 1개", ONE), ("특징 5개", FIVE)] if len(ns) == 2 else [], boxes=False)

    def rq2_figures(self):
        for cohort in ["within_method", "all_methods"]:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
            for ax, side, label in zip(axes, ["image", "text"], ["이미지", "텍스트"], strict=True):
                for i, condition in enumerate(["shared_one", "best_all", "union"]):
                    rows = [self.one("shared_pairs", method=m, side=side, cohort=cohort, condition=condition) for m in METHODS]
                    self.boxes(ax, rows, np.arange(4)+(i-1)*.23, [ONE, COLORS[1], FIVE][i:i+1]*4, .18)
                self.style(ax, [s+f"\n{r['n']}쌍" for s, r in zip(LABELS, rows, strict=True)])
                ax.set_title(label)
            self.save(fig, "shared_"+cohort, "공유된 대표의 범주 간 직접 구별 · "+("기준별 집단" if cohort == "within_method" else "네 기준의 공통 범주 쌍"),
                      [("공유된 1개", ONE), ("별도 최선 1개", COLORS[1]), ("범주별 5개의 합집합 · 5–9개", FIVE)])

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
        for ax, (direction, title) in zip(axes, DIRECTIONS, strict=True):
            for n, offset, color in [(1, -.15, ONE), (5, .15, FIVE)]:
                rows = [self.one("raw_prediction", method=m, direction=direction, features=n) for m in METHODS]
                self.boxes(ax, rows, np.arange(4)+offset, [color]*4)
            self.style(ax, LABELS, "R²", None)
            prediction = "이미지 특징으로 텍스트 특징 예측" if direction == "image_to_text" else "텍스트 특징으로 이미지 특징 예측"
            ax.set_title(prediction+f"\n동일 {rows[0]['n']}개 범주")
        self.save(fig, "raw_prediction", "상대 활성값 예측 · 기준별 목표 좌표의 차이", [("입력 특징 1개", ONE), ("입력 특징 5개", FIVE)])

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
        for ax, (direction, title) in zip(axes, DIRECTIONS, strict=True):
            for condition, offset, color in [("aligned", -.16, ONE), ("shuffled", .16, FIVE)]:
                rows = [self.one("connections", method=m, direction=direction, condition=condition, model="reusable_5") for m in METHODS]
                bars = ax.bar(np.arange(4)+offset, [r["relative_improvement_percent"] for r in rows], width=.28, color=color, edgecolor="#67717A")
                ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
            self.style(ax, LABELS, "일대일 대비 예측 오차 감소 (%)", None)
            ax.axhline(0, color="#67717A", lw=.7)
            ax.margins(y=.2)
            ax.set_title("이미지 변화로 텍스트 변화 예측" if direction == "image_to_text" else "텍스트 변화로 이미지 변화 예측")
        self.save(fig, "connection_constraints", "가림 전후 변화 예측의 입력 재사용과 입력 5개 허용", [("실제 짝", ONE), ("같은 범주에서 짝을 섞은 대조", FIVE)], boxes=False)

    def diagnosis_figures(self):
        for train, test, label in [("presence", "presence", "이미지의 범주 존재 정보"),
                                   ("presence", "removal", "원본에서 배운 범주 점수의 제거 반응"),
                                   ("removal", "removal", "원본·가림 상태 구별의 보조 진단")]:
            rows = [self.one("diagnosis", method="pooled_auroc", representation=rep, trained_on=train, evaluated_on=test, measure="auroc") for rep in ["embedding", "all_sae"]]
            rows += [self.one("diagnosis", method=m, representation="selected_five", trained_on=train, evaluated_on=test, measure="auroc") for m in METHODS]
            fig, ax = plt.subplots(figsize=(12, 5.5))
            self.boxes(ax, rows, np.arange(6), ["#DDE2E6"]*2+COLORS, .5)
            self.style(ax, ["원래 임베딩", "SAE 전체"]+LABELS)
            self.save(fig, f"diagnosis_{train}_{test}", label+" · 각 기준에서 선택한 특징 5개")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--font")
    args = parser.parse_args()
    names = {f.name for f in font_manager.fontManager.ttflist}
    font = args.font or next((f for f in ["NanumGothic", "Apple SD Gothic Neo", "AppleGothic", "Noto Sans CJK KR"] if f in names), None)
    if font is None:
        raise RuntimeError("A Korean font is required; supply --font")
    plt.rcParams.update({"font.family": font, "font.size": 11, "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "svg.fonttype": "path"})
    output = args.output or args.summary.parent/"figures"
    plots = Figures(json.loads(args.summary.read_text()), output)
    plots.concept_figures()
    plots.rq1_figures()
    plots.rq2_figures()
    plots.diagnosis_figures()
    (output/"manifest.json").write_text(json.dumps(dict(source=str(args.summary.resolve()),
        sha256=hashlib.sha256(args.summary.read_bytes()).hexdigest(), files=plots.files), indent=2))
    print(f"Saved {len(plots.files)} files in {output}")


if __name__ == "__main__":
    main()
