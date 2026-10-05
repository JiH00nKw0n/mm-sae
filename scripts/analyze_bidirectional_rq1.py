"""Compare both anchor modalities using saved natural-independence RQ1 scores."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from mm_sae.io import atomic_json, sha256, write_csv

COLORS = ["#FFADAD", "#FFD6A5", "#FDFFB6", "#CAFFBF", "#9BF6FF"]
INK = "#263540"
DIRECTIONS = {"image": "이미지 범주를 기준으로 비교", "text": "텍스트 범주를 기준으로 비교"}
GROUPS = ["all", "object_object", "object_background", "background_object", "background_background"]
EDGES = np.round(np.linspace(-1, 1, 11), 10)


@dataclass(frozen=True)
class Comparison:
    direction: str
    anchor_id: int
    candidate_id: int
    anchor_name: str
    candidate_name: str
    anchor_type: str
    candidate_type: str
    annotation: float
    original_other: float
    controlled_other: float
    original_same: float
    controlled_same: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def build_directions(rows: list[dict[str, str]]) -> dict[str, list[Comparison]]:
    """Transpose the full pair table; text anchors use the text category's diagonal."""
    indexed = {(int(r["image_category"]), int(r["other_category"])): r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate image/text category pair")
    ids = {a for a, _ in indexed}
    if set(indexed) != {(a, b) for a in ids for b in ids if a != b}:
        raise ValueError("Expected all off-diagonal pairs for the same category set")
    result: dict[str, list[Comparison]] = {"image": [], "text": []}
    for (a, b), row in indexed.items():
        reverse = indexed[b, a]
        if float(row["original_annotation_correlation"]) != float(reverse["original_annotation_correlation"]):
            raise ValueError("Image-label correlation must be symmetric")
        image_type, text_type = row["category_types"].split("_")
        for direction in DIRECTIONS:
            same = row if direction == "image" else reverse
            item = Comparison(
                direction=direction,
                anchor_id=a if direction == "image" else b,
                candidate_id=b if direction == "image" else a,
                anchor_name=row["image_name"] if direction == "image" else row["other_name"],
                candidate_name=row["other_name"] if direction == "image" else row["image_name"],
                anchor_type=image_type if direction == "image" else text_type,
                candidate_type=text_type if direction == "image" else image_type,
                annotation=float(row["original_annotation_correlation"]),
                original_other=float(row["original_other"]),
                controlled_other=float(row["independent_other"]),
                original_same=float(same["original_same"]),
                controlled_same=float(same["independent_same"]),
            )
            if not all(np.isfinite(x) for x in [item.original_other, item.controlled_other,
                                                item.original_same, item.controlled_same, item.annotation]):
                raise ValueError("Undefined comparison")
            result[direction].append(item)
    return result


def in_group(r: Comparison, group: str) -> bool:
    return group == "all" or f"{r.anchor_type}_{r.candidate_type}" == group


def in_bin(r: Comparison, lo: float, hi: float) -> bool:
    return lo <= r.annotation and (r.annotation < hi or hi == 1 and r.annotation <= hi)


def summarize_distribution(directions: dict[str, list[Comparison]]) -> list[dict]:
    output = []
    for direction, rows in directions.items():
        for group in GROUPS:
            for lo, hi in zip(EDGES[:-1], EDGES[1:], strict=True):
                selected = [r for r in rows if in_group(r, group) and in_bin(r, lo, hi)]
                before = np.asarray([r.original_other for r in selected])
                after = np.asarray([r.controlled_other for r in selected])
                for condition, values in [("original", before), ("controlled", after)]:
                    stats = dict.fromkeys(["mean", "std_population", "min", "q1", "median", "q3", "max"])
                    if len(values):
                        stats.update(zip(stats, [float(x) for x in [values.mean(), values.std(ddof=0),
                                         values.min(), *np.quantile(values, [.25, .5, .75]), values.max()]], strict=True))
                    output.append({"direction": direction, "anchor_candidate_types": group,
                                   "left": float(lo), "right": float(hi), "condition": condition,
                                   "pairs": len(selected), **stats,
                                   "mean_controlled_minus_original": float((after-before).mean()) if len(before) else None,
                                   "decreased_pairs": int((after < before).sum()),
                                   "increased_pairs": int((after > before).sum())})
    return output


def ranking_summary(rows: list[Comparison]) -> dict:
    anchors = {r.anchor_id for r in rows}
    exceeds = [r for r in rows if r.original_other > r.original_same]
    affected = {r.anchor_id for r in exceeds}
    return {"eligible_categories": len(anchors), "categories_with_higher_alternative": len(affected),
            "category_percent": 100 * len(affected) / len(anchors) if anchors else None,
            "pairs": len(rows), "exceeding_pairs": len(exceeds),
            "pair_percent": 100 * len(exceeds) / len(rows) if rows else None,
            "ties": sum(r.original_other == r.original_same for r in rows)}


def summarize_ranking(directions: dict[str, list[Comparison]]) -> tuple[list[dict], list[dict], list[dict]]:
    summaries, bins, per_anchor = [], [], []
    for direction, rows in directions.items():
        for anchor_type in ["all", "object", "background"]:
            selected = [r for r in rows if anchor_type == "all" or r.anchor_type == anchor_type]
            summaries.append({"direction": direction, "anchor_type": anchor_type,
                              "candidate_type": "all", **ranking_summary(selected)})
        for group in GROUPS[1:]:
            a, b = group.split("_")
            summaries.append({"direction": direction, "anchor_type": a, "candidate_type": b,
                              **ranking_summary([r for r in rows if in_group(r, group)])})
        for lo, hi in zip(EDGES[:-1], EDGES[1:], strict=True):
            bins.append({"direction": direction, "left": float(lo), "right": float(hi),
                         **ranking_summary([r for r in rows if in_bin(r, lo, hi)])})
        for anchor in sorted({r.anchor_id for r in rows}):
            selected = [r for r in rows if r.anchor_id == anchor]
            assert len({r.original_same for r in selected}) == 1
            per_anchor.append({"direction": direction, "anchor_id": anchor,
                               "anchor_name": selected[0].anchor_name, "anchor_type": selected[0].anchor_type,
                               "original_same": selected[0].original_same, **ranking_summary(selected)})
    return summaries, bins, per_anchor


def style(ax: Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E5E9ED", linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#B4BEC6")
    ax.tick_params(length=0, pad=7)


def save(fig: Figure, out: Path, name: str) -> None:
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(out / f"{name}.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)


def box(ax: Axes, row: dict, position: float, color: str, width: float) -> None:
    if not row["pairs"]:
        return
    ax.bxp([{"q1": row["q1"], "med": row["median"], "q3": row["q3"],
             "whislo": row["min"], "whishi": row["max"], "fliers": []}],
           positions=[position], widths=width, patch_artist=True, manage_ticks=False, showfliers=False,
           boxprops={"facecolor": color, "edgecolor": INK}, whiskerprops={"color": INK},
           capprops={"color": INK}, medianprops={"color": INK, "linewidth": 1.8})
    ax.scatter(position, row["mean"], color=INK, marker="D", s=23, zorder=4)


def plot_distributions(stats: list[dict], out: Path, controlled: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), sharey=True)
    fig.subplots_adjust(left=.065, right=.985, bottom=.26, top=.73, wspace=.16)
    title = "3. 양의 동시 등장 관계를 통제하면 상관점수가 낮아지는가?" if controlled else "1. 함께 등장하는 정도가 클수록 다른 범주의 상관점수가 높은가?"
    fig.text(.065, .94, title, fontsize=17)
    fig.text(.065, .885, "전체 범주 쌍을 포함하면 두 방향은 같은 점수 집합입니다. 두 그림의 분포가 같은 것은 집계 구조에 따른 결과입니다.", fontsize=10.5)
    for ax, direction in zip(axes, DIRECTIONS, strict=True):
        ax.set_title(DIRECTIONS[direction], fontsize=13, pad=14)
        original = [r for r in stats if r["direction"] == direction and r["anchor_candidate_types"] == "all"
                    and r["left"] >= 0 and r["condition"] == "original"]
        after = [r for r in stats if r["direction"] == direction and r["anchor_candidate_types"] == "all"
                 and r["left"] >= 0 and r["condition"] == "controlled"]
        for i, row in enumerate(original):
            if controlled:
                box(ax, row, i-.17, COLORS[0], .28)
                box(ax, after[i], i+.17, COLORS[4], .28)
            else:
                box(ax, row, i, COLORS[i], .52)
        labels = [f"[{r['left']:.1f}, {r['right']:.1f}{']' if i == 4 else ')'}\n{r['pairs']:,}쌍"
                  for i, r in enumerate(original)]
        ax.set(xticks=range(5), xticklabels=labels, xlim=(-.65, 4.65), ylim=(-.1, 1))
        ax.set_xlabel("원본 이미지 주석의 범주 간 상관계수", labelpad=10)
        ax.tick_params(axis="x", labelsize=9)
        style(ax)
    axes[0].set_ylabel("서로 다른 범주의 이미지·텍스트 특징 상관점수", labelpad=9)
    if controlled:
        fig.legend(handles=[Patch(facecolor=COLORS[0], edgecolor=INK, label="원본"),
                            Patch(facecolor=COLORS[4], edgecolor=INK, label="양의 동시 등장 관계 통제 후")],
                   loc="upper center", bbox_to_anchor=(.54, .854), ncol=2, frameon=False, fontsize=10)
    fig.text(.065, .105, "상자는 25–75% 분위수, 가운데 선은 중앙값, 양 끝은 최솟값·최댓값, 마름모는 평균입니다.", fontsize=10)
    fig.text(.065, .065, "구간과 비교 쌍은 원본 기준으로 고정했습니다. 주석 상관 0.6 이상은 두 구간에 각각 4쌍과 2쌍뿐입니다.", fontsize=10)
    fig.text(.065, .025, "텍스트를 기준으로 보아도 주석 상관의 출처와 통제 방식은 동일합니다. 텍스트 언급 주석의 상관으로 바꾸지 않았습니다.", fontsize=9.5)
    save(fig, out, "03_controlled_scores_both_directions" if controlled else "01_original_scores_both_directions")


def plot_ranking(summaries: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8))
    fig.subplots_adjust(left=.1, right=.97, top=.72, bottom=.22)
    fig.text(.1, .94, "2. 같은 범주보다 점수가 높은 다른 상대가 하나라도 있는가?", fontsize=17)
    count = next(r["eligible_categories"] for r in summaries if r["direction"] == "image" and r["anchor_type"] == "all")
    fig.text(.1, .885, f"각 이미지 범주는 다른 {count-1}개 텍스트 범주와, 각 텍스트 범주는 다른 {count-1}개 이미지 범주와 비교했습니다.", fontsize=10.5)
    for j, (direction, label) in enumerate(DIRECTIONS.items()):
        selected = [r for r in summaries if r["direction"] == direction and r["candidate_type"] == "all"]
        for i, row in enumerate(selected):
            x = i + [-.18, .18][j]
            ax.bar(x, row["category_percent"], width=.32, color=COLORS[0 if j == 0 else 4], edgecolor=INK, linewidth=.8)
            ax.text(x, row["category_percent"]+1.8,
                    f"{row['category_percent']:.2f}%\n{row['categories_with_higher_alternative']} / {row['eligible_categories']}개",
                    ha="center", fontsize=10.5)
    ax.set(xticks=range(3), xticklabels=["기준 범주 전체", "기준 범주가 객체", "기준 범주가 배경"], ylim=(0, 104))
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_ylabel("다른 상대가 하나라도 더 높은 기준 범주의 비율", labelpad=12)
    fig.legend(handles=[Patch(facecolor=COLORS[0], edgecolor=INK, label=DIRECTIONS["image"]),
                        Patch(facecolor=COLORS[4], edgecolor=INK, label=DIRECTIONS["text"])],
               loc="upper center", bbox_to_anchor=(.54, .84), ncol=2, frameon=False, fontsize=10)
    style(ax)
    fig.text(.1, .095, "각 기준 범주는 한 번만 셉니다. 상대에는 객체와 배경을 모두 포함하며, 동점은 초과로 세지 않습니다.", fontsize=10)
    fig.text(.1, .05, "이미지 기준은 같은 이미지 범주의 대각 점수와, 텍스트 기준은 같은 텍스트 범주의 대각 점수와 비교합니다.", fontsize=10)
    save(fig, out, "02_higher_alternatives_both_directions")


def verify_pooling(directions: dict[str, list[Comparison]]) -> None:
    for lo, hi in zip(EDGES[:-1], EDGES[1:], strict=True):
        for group in GROUPS:
            reverse_group = "_".join(group.split("_")[::-1])
            for field in ["original_other", "controlled_other"]:
                image = sorted(getattr(r, field) for r in directions["image"] if in_group(r, group) and in_bin(r, lo, hi))
                text = sorted(getattr(r, field) for r in directions["text"] if in_group(r, reverse_group) and in_bin(r, lo, hi))
                np.testing.assert_array_equal(image, text)


def write_report(out: Path, stats: list[dict], summaries: list[dict], bins: list[dict]) -> None:
    labels = {"all": "전체 범주", "object": "객체 범주", "background": "배경 범주"}
    text = """# RQ1의 세 질문을 이미지 기준과 텍스트 기준으로 비교한 결과

이미지를 기준으로 다른 텍스트 범주와 비교할 때와 텍스트를 기준으로 다른 이미지 범주와 비교할 때를 모두 계산했다. 다른 상대가 하나라도 같은 범주의 점수를 넘는 비율은 두 방향에서 달랐다. 모든 범주 쌍의 상관점수 분포 자체는 두 방향에서 정확히 같았다. 이 동일성은 서로 독립인 두 실험에서 같은 결과를 얻었다는 뜻이 아니라, 같은 점수들을 기준 범주에 따라 다시 배열한 결과다.

## 비교 방향을 정의하는 방법

이미지 범주 A의 대표 특징과 텍스트 범주 B의 대표 특징 사이의 상관점수를 C(A,B)로 쓴다. 이미지 A를 기준으로 다른 텍스트 B를 평가하면 C(A,B)를 같은 범주의 점수 C(A,A)와 비교한다. 텍스트 A를 기준으로 다른 이미지 B를 평가하면 C(B,A)를 C(A,A)와 비교한다. C(A,B)와 C(B,A)는 일반적으로 서로 다르다. 모든 A와 B를 포함하면 두 방향의 전체 점수 집합이 같아진다.

원본 COCO2017 train의 고정된 대표 특징과 저장된 원본·통제 점수를 사용했다. 모델을 다시 학습하거나 활성값을 새로 계산하지 않았다. 동일한 특징 번호를 여러 범주가 공유해도 제외하지 않았다. 다른 범주의 점수가 엄격히 클 때만 초과로 세며, 동점은 초과로 세지 않았다.

두 방향 모두 모델 입력 영역의 이미지 주석을 사용한다. 주석 상관은 각 이미지·캡션 관측에 같은 원본 비중을 주어 계산한 값이다. 텍스트 기준 비교에서도 주석 상관을 텍스트 언급 주석으로 바꾸지 않았다. 각 범주 쌍의 원본 주석 상관으로 0.2 간격의 구간을 나누고, 통제 후에도 같은 구간을 유지했다.

## 1. 함께 등장하는 정도가 클수록 다른 범주의 상관점수가 높은가?

두 방향 모두 주석 상관이 높은 구간일수록 평균과 중앙값이 높았다. 각 구간의 평균, 표준편차, 사분위수, 최솟값, 최댓값은 두 방향에서 정확히 같다. 이미지·객체와 텍스트·배경처럼 비교하는 종류를 제한할 때는 상대 역할도 함께 바꾸어야 동일한 점수 집합이 된다. 예를 들어 이미지 객체·텍스트 배경은 텍스트 배경·이미지 객체와 같은 집합이다.

"""
    text += f"![원본 점수 분포를 두 방향에서 비교했다.]({out}/01_original_scores_both_directions.png)\n\n"
    text += """## 2. 다른 범주의 점수가 같은 범주의 점수를 넘는가?

각 기준 범주는 반대쪽의 나머지 모든 범주와 비교했다. 다음 표는 같은 범주보다 높은 상대가 하나라도 있는 기준 범주를 세며, 여러 높은 상대가 있어도 기준 범주는 한 번만 센다. 객체 행과 배경 행에서도 비교 상대는 다른 객체와 배경을 모두 포함한다.

| 기준 범주의 종류 | 이미지 범주를 기준으로 비교 | 텍스트 범주를 기준으로 비교 |
| --- | ---: | ---: |
"""
    for group, label in labels.items():
        cells = []
        for direction in DIRECTIONS:
            row = next(r for r in summaries if r["direction"] == direction and r["anchor_type"] == group and r["candidate_type"] == "all")
            cells.append(f"{row['categories_with_higher_alternative']} / {row['eligible_categories']}개 = {row['category_percent']:.2f}%")
        text += f"| {label} | {' | '.join(cells)} |\n"
    text += f"\n![같은 범주보다 높은 상대가 하나라도 있는 기준 범주의 비율]({out}/02_higher_alternatives_both_directions.png)\n\n"
    text += "전체 범주를 합쳤을 때 개별 방향별 쌍을 세는 결과는 다음과 같다. 이는 기준 범주를 한 번만 세는 위 표와 분모가 다르다.\n\n| 비교 방향 | 같은 범주보다 점수가 높은 개별 쌍 | 동점인 개별 쌍 |\n| --- | ---: | ---: |\n"
    for direction, label in DIRECTIONS.items():
        r = next(r for r in summaries if r["direction"] == direction and r["anchor_type"] == r["candidate_type"] == "all")
        text += f"| {label} | {r['exceeding_pairs']:,} / {r['pairs']:,}쌍 = {r['pair_percent']:.2f}% | {r['ties']:,}쌍 |\n"
    text += "\n구간별 비율에서는 해당 구간에 비교 상대가 있는 기준 범주만 분모에 포함한다. 같은 기준 범주가 여러 구간에 들어갈 수 있으므로 구간별 범주 수를 더하지 않는다.\n\n| 원본 주석 상관 구간 | 이미지 기준의 범주 비율 | 텍스트 기준의 범주 비율 | 이미지 기준의 쌍 비율 | 텍스트 기준의 쌍 비율 |\n| --- | ---: | ---: | ---: | ---: |\n"
    for r in bins:
        if r["direction"] != "image" or not r["pairs"]:
            continue
        t = next(t for t in bins if t["direction"] == "text" and t["left"] == r["left"])
        interval = f"{r['left']:.1f} 이상 {r['right']:.1f} " + ("이하" if r["right"] == 1 else "미만")
        cells = [f"{x['categories_with_higher_alternative']} / {x['eligible_categories']}개 = {x['category_percent']:.2f}%" for x in [r, t]]
        cells += [f"{x['exceeding_pairs']:,} / {x['pairs']:,}쌍 = {x['pair_percent']:.2f}%" for x in [r, t]]
        text += f"| {interval} | {' | '.join(cells)} |\n"
    text += """
## 3. 양의 동시 등장 관계를 통제하면 다른 범주의 상관점수가 낮아지는가?

세 번째 질문은 같은 범주 점수를 넘는 사례가 줄어드는지가 아니라, 다른 범주의 상관점수 자체가 낮아지는지다. 두 방향 모두 다섯 구간에서 평균과 중앙값이 낮아졌다. 주석 상관이 0.6 미만인 세 구간에서는 평균이 약 14.6–17.9% 낮아졌다. 모든 개별 점수가 낮아진 것은 아니며, 통제 후에도 높은 점수와 구간 간 차이가 남았다.

"""
    text += f"![통제 전후 점수 분포를 두 방향에서 비교했다.]({out}/03_controlled_scores_both_directions.png)\n\n"
    text += "다음 통계는 이미지 기준과 텍스트 기준에서 모두 동일하다. 상자는 25–75% 분위수, 가운데 선은 중앙값, 양 끝은 실제 최솟값·최댓값, 마름모는 평균이다. 표준편차는 구간의 전체 쌍을 대상으로 계산했고, 사분위수는 선형 보간했다.\n\n| 원본 주석 상관 구간 | 방향별 쌍 수 | 조건 | 평균 | 표준편차 | 최솟값 | 25% 분위수 | 중앙값 | 75% 분위수 | 최댓값 |\n| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    for r in stats:
        if r["direction"] != "image" or r["anchor_candidate_types"] != "all" or r["left"] < 0:
            continue
        interval = f"{r['left']:.1f} 이상 {r['right']:.1f} " + ("이하" if r["right"] == 1 else "미만")
        condition = "원본" if r["condition"] == "original" else "동시 등장 관계 통제 후"
        cells = [interval, f"{r['pairs']:,}", condition]
        cells += [f"{r[k]:.4f}" for k in ["mean", "std_population", "min", "q1", "median", "q3", "max"]]
        text += "| " + " | ".join(cells) + " |\n"
    text += """
주석 상관이 0.6 이상인 두 구간은 방향별 쌍이 4개와 2개뿐이므로 결과를 일반화하지 않는다. 반대 방향의 비교는 같은 이미지 주석 관계를 공유하며 독립적인 추가 사례가 아니다. 배정한 대표 특징이 실제 의미를 정확히 나타내는지에 대한 불확실성도 그대로 남아 있다.

각 이미지 범주 A와 B의 원래 등장 비율을 p와 q로 두고, 모두 없음·B만 있음·A만 있음·모두 있음의 집단 비중을 각각 (1-p)(1-q), (1-p)q, p(1-q), pq로 조절했다. 양의 주석 상관이 있는 쌍만 조절했고 나머지는 원본 점수를 유지했다. 각 집단 안에서는 관측의 상대 비중을 유지했다. 각 쌍마다 평가 분포가 다르며, 캡션에서 범주를 언급하는 비율과 다른 장면 요소는 별도로 고정하지 않았다. 따라서 독립적인 인과 효과나 학습 당시의 동시 등장 효과까지 판정하는 실험은 아니다.

## 검증과 저장 파일

전치한 비교표에서 모든 기준 범주에 다른 범주 전체가 포함되는지 검사했다. 텍스트 기준에서 같은 범주 점수를 잘못된 이미지 범주의 값으로 가져오지 않도록 서로 다른 대각 점수를 가진 비대칭 예제로 검증했다. 모든 원본 주석 구간에서 두 방향의 전체 점수 집합이 정확히 같음을 검사했고, 객체·배경으로 나눈 집합은 기준과 상대 종류를 교환하면 정확히 같음을 검사했다.

"""
    for name, description in [
        ("comparisons_both_directions.csv", "모든 기준 범주·상대 범주의 원본 및 통제 점수를 저장했다."),
        ("ranking_summary.csv", "전체·객체·배경 및 기준·상대 종류별 비율을 저장했다."),
        ("ranking_by_annotation_bin.csv", "주석 상관 구간마다 두 방향의 기준 범주 비율과 개별 쌍 비율을 저장했다."),
        ("ranking_per_category.csv", "각 기준 범주의 같은 범주 점수와 높은 상대 수를 저장했다."),
        ("score_distributions.csv", "음의 구간을 포함해 두 방향·기준과 상대 종류·통제 조건별 전체 분포 통계를 저장했다."),
        ("manifest.json", "입력 파일과 코드의 SHA256 해시 및 집계 기준을 저장했다."),
    ]:
        text += f"\n`{out / name}`에 {description}\n"
    (out / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--font", default="AppleGothic", help="Use an installed font supporting Korean")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    source = (args.comparison_dir / "all_category_comparisons.csv").resolve()
    directions = build_directions(read_csv(source))
    verify_pooling(directions)
    stats = summarize_distribution(directions)
    summaries, bins, anchors = summarize_ranking(directions)
    write_csv(args.out / "comparisons_both_directions.csv", [asdict(r) for rows in directions.values() for r in rows])
    write_csv(args.out / "score_distributions.csv", stats)
    write_csv(args.out / "ranking_summary.csv", summaries)
    write_csv(args.out / "ranking_by_annotation_bin.csv", bins)
    write_csv(args.out / "ranking_per_category.csv", anchors)
    plt.rcParams.update({"font.family": args.font, "font.size": 11, "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "svg.fonttype": "path", "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
    plot_distributions(stats, args.out, controlled=False)
    plot_ranking(summaries, args.out)
    plot_distributions(stats, args.out, controlled=True)
    write_report(args.out, stats, summaries, bins)
    atomic_json(args.out / "manifest.json", {
        "source": str(source), "source_sha256": sha256(source), "script_sha256": sha256(Path(__file__)),
        "categories_per_direction": len(anchors)//2, "comparisons_per_direction": len(directions["image"]),
        "annotation_source": "Original caption-weighted model-visible image labels in both directions",
        "control": "Zero only positive image-label dependence, retaining original image-label prevalences",
        "pooling_verified": "Exact equality in every original annotation bin; cross-type groups swap anchor/candidate roles",
        "same_category_baseline": "Diagonal of the anchor category; controlled diagonal also comes from that anchor",
        "quartiles": "linear interpolation", "standard_deviation": "population, ddof=0",
        "whiskers": "actual minimum and maximum", "new_model_inference": False,
    })
    for row in summaries:
        if row["candidate_type"] == "all":
            print(row)
    print(f"Saved three bidirectional figures and complete CSVs to {args.out}")


if __name__ == "__main__":
    main()
