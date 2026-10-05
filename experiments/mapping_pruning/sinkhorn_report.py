"""Render the fixed Sinkhorn epsilon and pruning grid from saved evaluations."""

from __future__ import annotations

import base64
import csv
import html
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


Record = dict[str, Any]
DIRECTIONS = ("image_to_text", "text_to_image")
RANKS = (1, 5, 10)
EPSILONS = (0.003, 0.01, 0.03, 0.05, 0.1, 0.3)
BUDGETS = (None, 1, 2, 4, 8, 16, 32, 64)
SPACES = {
    "text_projected_to_image": "문장을 이미지 특징 공간으로 변환했다",
    "image_projected_to_text": "이미지를 문장 특징 공간으로 변환했다",
}
COLORS = ("#163b59", "#287ba2", "#43a5a0", "#aa8530", "#b95c53", "#843f79")
STRUCTURE_FIELDS = (
    "edge_count", "density", "image_coverage", "text_coverage",
    "image_effective_degree_mean", "text_effective_degree_mean",
    "image_mass95_degree_mean", "text_mass95_degree_mean", "retained_mass",
)


def _recall(record: Record, direction: str, rank: int) -> float:
    return 100 * float(record["retrieval"][direction]["recall"][str(rank)])


def _condition(record: Record) -> str:
    k = record["metadata"]["k"]
    return "대응 가중치를 모두 유지했다" if k is None else f"각 특징에 연결된 상대 특징을 최대 {k}개 유지했다"


def _load(out: Path, population: Record) -> list[Record]:
    paths = sorted((out / "sinkhorn/results").glob("sinkhorn_*.json"))
    if not paths:
        return []
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    observed = set()
    for record in records:
        metadata = record["metadata"]
        identity = (float(metadata["epsilon"]), metadata["k"], record["space"])
        if identity in observed:
            raise ValueError(f"Duplicate Sinkhorn evaluation: {identity}")
        observed.add(identity)
        for field in STRUCTURE_FIELDS:
            if not math.isfinite(float(metadata["structure"][field])):
                raise ValueError(f"Non-finite Sinkhorn structure value: {identity}/{field}")
        for direction in DIRECTIONS:
            metric = record["retrieval"][direction]
            expected = (population["test_images"], population["test_captions"])
            if direction == "text_to_image":
                expected = expected[::-1]
            if (metric["query_count"], metric["candidate_count"]) != expected:
                raise ValueError(f"Sinkhorn retrieval population differs: {identity}/{direction}")
            for rank in RANKS:
                if not 0 <= _recall(record, direction, rank) <= 100:
                    raise ValueError(f"Invalid Sinkhorn recall: {identity}/{direction}/{rank}")
    expected_grid = {(epsilon, k, space) for epsilon in EPSILONS for k in BUDGETS for space in SPACES}
    if observed != expected_grid:
        raise ValueError(f"Sinkhorn grid is incomplete or unexpected: {len(observed)} of {len(expected_grid)}")
    lookup = {(r["metadata"]["epsilon"], r["metadata"]["k"], r["space"]): r for r in records}
    for epsilon in EPSILONS:
        for k in BUDGETS:
            left, right = [lookup[epsilon, k, space]["metadata"] for space in SPACES]
            if left["structure"] != right["structure"] or left["solver"] != right["solver"]:
                raise ValueError("Both projection spaces must use the same Sinkhorn mapping")
    return [lookup[epsilon, k, space] for epsilon in EPSILONS for k in BUDGETS for space in SPACES]


def _write_csv(out: Path, records: list[Record]) -> None:
    rows = []
    for record in records:
        metadata = record["metadata"]
        row = {
            "method": "Sinkhorn",
            "epsilon": metadata["epsilon"],
            "k": "full" if metadata["k"] is None else metadata["k"],
            "condition_definition": _condition(record),
            "projection_space": record["space"],
            "projection_definition": SPACES[record["space"]],
            **{field: metadata["structure"][field] for field in STRUCTURE_FIELDS},
            "numeric_zero_count": metadata["structure"]["numeric_zero_count"],
            **{f"solver_{field}": metadata["solver"][field]
               for field in ("converged", "iterations", "residual", "tol", "max_iter")},
        }
        for direction in DIRECTIONS:
            metric = record["retrieval"][direction]
            row[f"{direction}_query_count"] = metric["query_count"]
            row[f"{direction}_candidate_count"] = metric["candidate_count"]
            row[f"{direction}_zero_norm_queries"] = metric["zero_norm_query_count"]
            for rank in RANKS:
                row[f"{direction}_recall_at_{rank}_percent"] = _recall(record, direction, rank)
        rows.append(row)
    with (out / "sinkhorn-summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_settings() -> dict[str, Any]:
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ("Apple SD Gothic Neo", "Nanum Gothic", "Noto Sans CJK KR")
                 if name in fonts), "DejaVu Sans")
    return {"font.family": font, "font.size": 11, "axes.unicode_minus": False,
            "axes.spines.top": False, "axes.spines.right": False}


def _axis_style(axis: Any) -> None:
    axis.grid(axis="y", color="#dce2e7", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.set_ylim(bottom=0)


def _figures(out: Path, records: list[Record]) -> None:
    full = [r for r in records if r["metadata"]["k"] is None]
    with plt.rc_context(_plot_settings()):
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), sharey=True)
        for axis, (space, name) in zip(axes, SPACES.items(), strict=True):
            subset = [r for r in full if r["space"] == space]
            for direction, label, color, marker in zip(
                DIRECTIONS, ("이미지로 문장 검색", "문장으로 이미지 검색"),
                (COLORS[0], COLORS[-1]), ("o", "s"), strict=True,
            ):
                axis.plot(EPSILONS, [_recall(r, direction, 10) for r in subset], color=color,
                          marker=marker, linewidth=2, markersize=5, label=label)
            axis.set_xscale("log")
            axis.set_xticks(EPSILONS, labels=[f"{epsilon:g}" for epsilon in EPSILONS])
            axis.set_xlabel("가중치 분산을 조절하는 값 ε")
            axis.set_title(name, fontsize=13, loc="left", pad=14)
            _axis_style(axis)
            axis.legend(frameon=False, loc="lower left")
        axes[0].set_ylabel("상위 10개 정답 포함 비율 (%)")
        fig.suptitle("Sinkhorn에서 대응 가중치를 모두 유지한 검색 결과", fontsize=16, y=0.96)
        fig.subplots_adjust(top=0.82, bottom=0.17, left=0.07, right=0.98, wspace=0.12)
        fig.savefig(out / "sinkhorn-epsilon-curves.png", dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.18)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
        subset = [r for r in full if r["space"] == next(iter(SPACES))]
        for modality, label, color, marker in zip(
            ("image", "text"), ("이미지 특징 전체의 평균", "문장 특징 전체의 평균"),
            (COLORS[0], COLORS[-1]), ("o", "s"), strict=True,
        ):
            for axis, field in zip(axes, ("effective_degree_mean", "mass95_degree_mean"), strict=True):
                axis.plot(EPSILONS, [r["metadata"]["structure"][f"{modality}_{field}"] for r in subset],
                          color=color, marker=marker, linewidth=2, markersize=5, label=label)
        for axis, title in zip(axes, ("가중치 집중을 반영한 유효 연결 수", "가중치 합의 95%를 차지하는 연결 수"), strict=True):
            axis.set_xscale("log")
            axis.set_xticks(EPSILONS, labels=[f"{epsilon:g}" for epsilon in EPSILONS])
            axis.set_xlabel("가중치 분산을 조절하는 값 ε")
            axis.set_ylabel("특징 하나당 상대 특징 수의 평균 (개)")
            axis.set_title(title, fontsize=13, loc="left", pad=14)
            _axis_style(axis)
            axis.legend(frameon=False, loc="upper left")
        fig.suptitle("Sinkhorn에서 대응 가중치를 모두 유지한 연결 구조", fontsize=16, y=0.96)
        fig.subplots_adjust(top=0.82, bottom=0.17, left=0.065, right=0.985, wspace=0.23)
        fig.savefig(out / "sinkhorn-connectivity.png", dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.18)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), sharex=True, sharey=True)
        for row, (space, space_name) in enumerate(SPACES.items()):
            for col, (direction, direction_name) in enumerate(zip(
                DIRECTIONS, ("이미지로 문장 검색", "문장으로 이미지 검색"), strict=True,
            )):
                axis = axes[row, col]
                for epsilon, color in zip(EPSILONS, COLORS, strict=True):
                    subset = [r for r in records if r["space"] == space and r["metadata"]["epsilon"] == epsilon]
                    reference = next(r for r in subset if r["metadata"]["k"] is None)
                    pruned = [r for r in subset if r["metadata"]["k"] is not None]
                    axis.plot([r["metadata"]["k"] for r in pruned], [_recall(r, direction, 10) for r in pruned],
                              color=color, marker="o", markersize=3.5, linewidth=1.8, label=f"ε = {epsilon:g}")
                    axis.axhline(_recall(reference, direction, 10), color=color, linestyle=(0, (4, 4)),
                                 linewidth=0.9, alpha=0.6)
                axis.set_xscale("log", base=2)
                axis.set_xticks(BUDGETS[1:], labels=[str(k) for k in BUDGETS[1:]])
                axis.set_title(f"{space_name}\n{direction_name}", fontsize=13, loc="left", pad=13)
                _axis_style(axis)
                if row == 1:
                    axis.set_xlabel("각 특징에 연결할 상대 특징의 최대 개수")
                if col == 0:
                    axis.set_ylabel("상위 10개 정답 포함 비율 (%)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.01))
        fig.suptitle("Sinkhorn의 작은 대응 가중치를 0으로 바꾼 검색 결과", fontsize=16, y=0.96)
        fig.subplots_adjust(top=0.88, bottom=0.13, left=0.07, right=0.985, wspace=0.13, hspace=0.35)
        fig.savefig(out / "sinkhorn-pruning-curves.png", dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.18)
        plt.close(fig)


def _retrieval_table(records: list[Record], *, epsilon_only: bool) -> str:
    header = "가중치 분산을 조절하는 값 ε" if epsilon_only else "각 특징의 연결 유지 조건"
    parts = [f'<div class="table-scroll"><table><thead><tr><th rowspan="2" scope="col">{header}</th>',
             '<th rowspan="2" scope="col">0보다 큰 특징 쌍의 수</th>',
             '<th rowspan="2" scope="col">전체 특징 쌍 중 연결 비율</th>',
             '<th colspan="3" scope="colgroup">이미지로 문장 검색</th>',
             '<th colspan="3" scope="colgroup">문장으로 이미지 검색</th></tr><tr>']
    parts.extend(f'<th scope="col">Recall@{rank}</th>' for _ in DIRECTIONS for rank in RANKS)
    parts.append("</tr></thead><tbody>")
    for record in records:
        metadata = record["metadata"]
        structure = metadata["structure"]
        label = f'ε = {metadata["epsilon"]:g}' if epsilon_only else _condition(record)
        css = ' class="reference"' if metadata["k"] is None else ""
        parts.append(f'<tr{css}><th scope="row">{label}</th><td>{structure["edge_count"]:,}개</td>'
                     f'<td>{100 * structure["density"]:.2f}%</td>')
        parts.extend(f'<td>{_recall(record, direction, rank):.2f}%</td>' for direction in DIRECTIONS for rank in RANKS)
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _structure_table(records: list[Record], *, epsilon_only: bool) -> str:
    heading = "가중치 분산을 조절하는 값 ε" if epsilon_only else "각 특징의 연결 유지 조건"
    columns = (
        heading, "0보다 큰 특징 쌍의 수", "연결이 남은 이미지 특징 비율", "연결이 남은 문장 특징 비율",
        "이미지 특징당 유효 연결 수의 평균", "문장 특징당 유효 연결 수의 평균",
        "이미지 특징당 가중치 95%의 연결 수 평균", "문장 특징당 가중치 95%의 연결 수 평균", "제거 전 가중치 합 중 유지 비율",
    )
    parts = ['<div class="table-scroll"><table class="structure-table"><thead><tr>']
    parts.extend(f'<th scope="col">{heading}</th>' for heading in columns)
    parts.append("</tr></thead><tbody>")
    for record in records:
        metadata = record["metadata"]
        structure = metadata["structure"]
        label = f'ε = {metadata["epsilon"]:g}' if epsilon_only else _condition(record)
        parts.append(f'<tr><th scope="row">{label}</th><td>{structure["edge_count"]:,}개</td>')
        parts.extend(f'<td>{100 * structure[field]:.2f}%</td>' for field in ("image_coverage", "text_coverage"))
        parts.extend(f'<td>{structure[field]:.2f}개</td>' for field in (
            "image_effective_degree_mean", "text_effective_degree_mean",
            "image_mass95_degree_mean", "text_mass95_degree_mean",
        ))
        parts.append(f'<td>{100 * structure["retained_mass"]:.2f}%</td></tr>')
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _figure_html(out: Path, filename: str, alt: str, caption: str) -> str:
    data = base64.b64encode((out / filename).read_bytes()).decode("ascii")
    return f'<figure><img src="data:image/png;base64,{data}" alt="{alt}"><figcaption>{caption}</figcaption></figure>'


def _lead(records: list[Record]) -> str:
    base_space = next(iter(SPACES))
    lookup = {(r["metadata"]["epsilon"], r["metadata"]["k"], r["space"]): r for r in records}
    first_record = lookup[EPSILONS[0], None, base_space]
    middle_record = lookup[0.05, None, base_space]
    last_record = lookup[EPSILONS[-1], None, base_space]
    first = first_record["metadata"]["structure"]
    last = lookup[EPSILONS[-1], None, base_space]["metadata"]["structure"]
    return (f'<p><strong>Sinkhorn에서 가중치를 제거하지 않고 ε를 {EPSILONS[-1]:g}에서 {EPSILONS[0]:g}으로 '
            f'줄였을 때, 이미지 특징당 유효 연결 수의 평균은 {last["image_effective_degree_mean"]:.2f}개에서 '
            f'{first["image_effective_degree_mean"]:.2f}개로 바뀌었다.</strong> '
            '유효 연결 수는 한 특징에 연결된 가중치 합의 제곱을 가중치 제곱의 합으로 나눈 값이다. '
            '가중치가 여러 연결에 얼마나 고르게 나뉘었는지를 나타내므로 실제로 남은 연결 수와 다르다. '
            f'같은 조건에서 문장 특징당 유효 연결 수의 평균은 {last["text_effective_degree_mean"]:.2f}개에서 '
            f'{first["text_effective_degree_mean"]:.2f}개로 바뀌었다. ε를 바꾸는 실험과 작은 가중치를 '
            '0으로 바꾸는 실험을 구분해 아래에 제시했다.</p>'
            '<p><strong>ε를 줄여도 검색 성능이 계속 높아지지는 않았다.</strong> '
            '문장을 이미지 특징 공간으로 변환해 이미지로 문장을 검색했을 때, '
            f'ε = {EPSILONS[-1]:g}·0.05·{EPSILONS[0]:g}에서 상위 10개 검색 결과에 정답이 포함된 비율(Recall@10)은 각각 '
            f'{_recall(last_record, "image_to_text", 10):.2f}%·{_recall(middle_record, "image_to_text", 10):.2f}%·'
            f'{_recall(first_record, "image_to_text", 10):.2f}%였다. '
            f'가중치를 제거하기 전에는 모든 ε에서 {first["edge_count"]:,}개 특징 쌍의 가중치가 0보다 컸다. '
            '이번 실험에서 정확히 0인 가중치는 작은 연결을 명시적으로 제거할 때 생겼다.</p>')


def render_sinkhorn(out: Path, population: Record) -> dict[str, str]:
    """Return report fragments and save the complete CSV and standalone figures."""
    records = _load(out, population)
    if not records:
        return {"lead": "", "body": "", "sources": "", "scope": ""}
    _write_csv(out, records)
    _figures(out, records)
    base_space = next(iter(SPACES))
    full = [r for r in records if r["metadata"]["k"] is None]
    unique_full = [r for r in full if r["space"] == base_space]
    epsilon_text = "·".join(f"{epsilon:g}" for epsilon in EPSILONS)
    budget_text = "·".join(str(k) for k in BUDGETS if k is not None)
    ni, nt = population["image_features"], population["text_features"]
    pieces = [f'''<section id="sinkhorn"><h2>Sinkhorn에서는 가중치 집중과 연결 제거를 따로 확인했다</h2>
    <p>Sinkhorn은 이미지 특징과 문장 특징의 각 쌍에 0 이상의 대응 가중치를 부여하는 방법이다.
    각 이미지 특징에 연결된 가중치의 합을 1/{ni:,}, 각 문장 특징에 연결된 가중치의 합을 1/{nt:,}로 고정했다.
    학습 자료에서 계산한 특징 간 상관계수에 높은 가중치를 주면서, 가중치가 여러 특징 쌍에 분산되는 정도도 고려했다.</p>
    <p>엔트로피는 가중치가 여러 특징 쌍에 분산된 정도를 나타내는 양이다.
    ε는 이 분산을 얼마나 선호할지 조절하는 값이다. ε가 작으면 상관계수 차이의 영향이 커져 가중치가 집중되는 경향이 있다.
    ε가 양수이고 모든 특징의 가중치 합이 양수인 이번 조건에서는 이론적인 대응 가중치가 모든 특징 쌍에서 양수다.
    따라서 ε를 줄이는 것만으로 정확히 0인 연결을 선택하지는 않는다.
    계산 정밀도의 한계로 나타나는 0은 명시적으로 연결을 제거한 결과와 구분해야 한다.</p>
    <p>ε는 {epsilon_text}으로 고정해 비교했다. 각 ε에서 대응 가중치를 모두 유지한 결과를 먼저 측정했다.
    이어서 각 이미지 특징과 각 문장 특징에서 큰 가중치를 각각 선택하고, 양쪽에서 모두 선택된 특징 쌍만 남겼다.
    각 특징에 연결할 상대 특징의 최대 개수는 {budget_text}으로 고정했다.
    선택되지 않은 가중치는 0으로 바꿨고, 남긴 가중치의 원래 값은 유지했다.</p>
    <p>연결을 제거한 뒤 Sinkhorn을 다시 풀지는 않았다. 따라서 제거 후의 가중치 합은 원래 제약을 충족하지 않을 수 있다.
    문장을 이미지 특징 공간으로 변환할 때는 연결을 제거한 뒤 각 이미지 특징에 남은 가중치의 합으로 나누어 문장 특징의 가중평균을 구했다.
    이미지를 문장 특징 공간으로 변환할 때는 연결을 제거한 뒤 각 문장 특징에 남은 가중치의 합으로 나누어 이미지 특징의 가중평균을 구했다.
    연결이 모두 사라진 특징의 변환값은 0으로 두었다. 두 경우 모두 변환된 표현과 원래 상대 표현의 코사인 유사도를 계산했다.</p>
    <p class="note">ε와 연결 개수는 평가 성능으로 다시 선택하지 않았다. 아래 결과는 미리 정한 모든 조합의 탐색적 비교이며,
    특정 ε나 연결 개수를 새로운 최종 모델로 선택한 결과가 아니다.</p></section>''']
    pieces.append('<section><h2>가중치를 모두 유지한 ε별 검색 결과</h2>')
    for space, name in SPACES.items():
        subset = [r for r in full if r["space"] == space]
        first, last = subset[0], subset[-1]
        pieces.append(f'<h3>{name}</h3><p>ε = {EPSILONS[0]:g}과 ε = {EPSILONS[-1]:g}에서 '
                      f'이미지로 문장을 검색한 Recall@10은 각각 {_recall(first, "image_to_text", 10):.2f}%와 '
                      f'{_recall(last, "image_to_text", 10):.2f}%였다. 같은 두 조건에서 '
                      f'문장으로 이미지를 검색한 Recall@10은 각각 {_recall(first, "text_to_image", 10):.2f}%와 '
                      f'{_recall(last, "text_to_image", 10):.2f}%였다.</p>')
        pieces.append(_retrieval_table(subset, epsilon_only=True))
    pieces.append(_figure_html(out, "sinkhorn-epsilon-curves.png", "두 변환 공간에서 ε별 Sinkhorn 검색 성능을 비교한 그래프",
                              "각 그래프는 변환 공간을 고정한 상태에서 ε만 바꾼 결과다. 모든 대응 가중치를 유지했다. "
                              "두 선은 이미지로 문장을 검색한 결과와 문장으로 이미지를 검색한 결과를 나타낸다."))
    pieces.append('</section><section><h2>작은 ε의 가중치 집중을 연결 구조로 확인했다</h2>')
    pieces.append('''<p>유효 연결 수는 한 특징에 연결된 가중치 합의 제곱을 가중치 제곱의 합으로 나눈 값이다.
    동일한 가중치가 8개 있으면 유효 연결 수는 8개이고, 연결은 여러 개여도 한 가중치가 대부분을 차지하면 1개에 가까워진다.
    가중치 95%의 연결 수는 가중치가 큰 순서대로 더해 합의 95% 이상이 되는 데 필요한 최소 연결 수다.
    두 수치는 이미지 특징 전체와 문장 특징 전체에서 각각 평균을 구했으며, 연결이 없는 특징은 0개로 계산했다.</p>
    <p>0보다 큰 특징 쌍의 수는 계산 결과에서 가중치가 정확히 0보다 큰 쌍만 센 값이다.
    연결이 남은 특징 비율은 전체 이미지 또는 문장 특징 중 가중치가 0보다 큰 연결을 하나 이상 가진 특징의 비율이다.
    제거 전 가중치 합 중 유지 비율은 해당 ε에서 가중치를 모두 유지한 결과와 비교했다.</p>''')
    positive_counts = [r["metadata"]["structure"]["edge_count"] for r in unique_full]
    numeric_zeros = [r["metadata"]["structure"]["numeric_zero_count"] for r in unique_full]
    if all(count == ni * nt for count in positive_counts):
        pieces.append(f'<p><strong>가중치를 제거하기 전에는 {len(EPSILONS)}개 ε 모두에서 '
                      f'{ni * nt:,}개 특징 쌍의 가중치가 0보다 컸다.</strong> '
                      '이번 계산에서는 ε를 줄여도 정확히 0인 가중치가 생기지 않았다.</p>')
    else:
        pieces.append(f'<p>가중치를 제거하기 전 각 ε에서 계산 결과가 정확히 0인 특징 쌍은 '
                      f'최소 {min(numeric_zeros):,}개, 최대 {max(numeric_zeros):,}개였다. '
                      '이론적인 해는 모든 특징 쌍에서 양수이므로 이 0을 학습된 연결 제거로 해석하지 않았다.</p>')
    pieces.append(_structure_table(unique_full, epsilon_only=True))
    pieces.append(_figure_html(out, "sinkhorn-connectivity.png", "ε별 유효 연결 수와 가중치 95퍼센트의 연결 수",
                              "두 그래프는 대응 가중치를 모두 유지한 결과다. 유효 연결 수와 가중치 95%의 연결 수가 "
                              "작아져도 나머지 연결의 가중치가 정확히 0이라는 뜻은 아니다."))
    pieces.append('</section><section><h2>각 ε에서 작은 가중치를 제거한 검색 결과</h2>')
    focus_epsilon = 0.05
    focus_k = 8
    focus = next(r for r in records if r["space"] == base_space
                 and r["metadata"]["epsilon"] == focus_epsilon and r["metadata"]["k"] == focus_k)
    structure = focus["metadata"]["structure"]
    pieces.append(f'<p>예를 들어 ε = {focus_epsilon:g}에서 각 특징의 상대 특징을 최대 {focus_k}개로 제한하면 '
                  f'{structure["edge_count"]:,}개 특징 쌍이 남았다. 전체 {ni * nt:,}개 특징 쌍의 '
                  f'{100 * structure["density"]:.2f}%이며, 제거 전 가중치 합의 '
                  f'{100 * structure["retained_mass"]:.2f}%를 유지했다. '
                  '이 조건은 연결 개수와 가중치 합의 차이를 설명하기 위한 예시다.</p>')
    for space, space_name in SPACES.items():
        chosen = next(r for r in records if r["space"] == space
                      and r["metadata"]["epsilon"] == focus_epsilon and r["metadata"]["k"] == focus_k)
        reference = next(r for r in records if r["space"] == space
                         and r["metadata"]["epsilon"] == focus_epsilon and r["metadata"]["k"] is None)
        pieces.append(f'<p>ε = {focus_epsilon:g}에서 {space_name.replace("변환했다", "변환했")}을 때, '
                      f'각 특징의 상대 특징을 최대 {focus_k}개로 제한한 Recall@10은 '
                      f'이미지로 문장을 검색할 때 {_recall(chosen, "image_to_text", 10):.2f}%, '
                      f'문장으로 이미지를 검색할 때 {_recall(chosen, "text_to_image", 10):.2f}%였다. '
                      f'같은 ε에서 모든 가중치를 유지한 값은 각각 {_recall(reference, "image_to_text", 10):.2f}%와 '
                      f'{_recall(reference, "text_to_image", 10):.2f}%였다.</p>')
    pieces.append(_figure_html(out, "sinkhorn-pruning-curves.png", "ε와 각 특징의 최대 연결 수에 따른 네 가지 Sinkhorn 검색 결과",
                              "실선과 점은 연결을 제거한 결과다. 같은 색의 수평 점선은 같은 ε에서 모든 가중치를 유지한 기준이다. "
                              "행은 변환 공간을, 열은 검색 방향을 구분한다. 각 ε의 전체 검색 지표와 연결 구조는 아래 표에서 확인할 수 있다."))
    for epsilon in EPSILONS:
        pieces.append(f'<details class="sinkhorn-details"{" open" if epsilon == focus_epsilon else ""}><summary>ε = {epsilon:g}의 전체 검색 지표와 연결 구조를 확인한다</summary>')
        subset = [r for r in records if r["metadata"]["epsilon"] == epsilon]
        for space, name in SPACES.items():
            pieces.append(f'<h3>{name}</h3>')
            pieces.append(_retrieval_table([r for r in subset if r["space"] == space], epsilon_only=False))
        pieces.append('<h3>두 변환 공간에서 함께 사용하는 대응 가중치의 연결 구조</h3>')
        pieces.append(_structure_table([r for r in subset if r["space"] == base_space], epsilon_only=False))
        pieces.append('</details>')
    pieces.append('</section><section><h2>Sinkhorn 계산의 수렴 여부를 확인했다</h2>')
    converged_count = sum(bool(r["metadata"]["solver"]["converged"]) for r in unique_full)
    pieces.append(f'<p>수렴은 각 이미지·문장 특징의 가중치 합이 정해진 값에 충분히 가까워진 상태다. '
                  f'{len(EPSILONS)}개 ε 중 {converged_count}개가 설정한 오차 기준을 충족했다. '
                  '표의 최대 오차는 이미지 특징과 문장 특징의 가중치 합에서 발생한 절대 오차 중 최댓값이다. '
                  '이 수렴 검사는 연결을 제거하기 전의 대응 가중치에 적용했다.</p>')
    pieces.append('<div class="table-scroll"><table><thead><tr><th scope="col">가중치 분산을 조절하는 값 ε</th>'
                  '<th scope="col">수렴 여부</th><th scope="col">반복 횟수</th><th scope="col">가중치 합의 최대 오차</th>'
                  '<th scope="col">허용 오차</th></tr></thead><tbody>')
    for record in unique_full:
        solver = record["metadata"]["solver"]
        status = "오차 기준을 충족했다" if solver["converged"] else "오차 기준을 충족하지 못했다"
        pieces.append(f'<tr><th scope="row">ε = {record["metadata"]["epsilon"]:g}</th><td>{status}</td>'
                      f'<td>{solver["iterations"]:,}회</td><td>{solver["residual"]:.2e}</td>'
                      f'<td>{solver["tol"]:.2e}</td></tr>')
    pieces.append('</tbody></table></div>')
    verification_path = out / "sinkhorn/verification.json"
    if verification_path.exists():
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        expected_recalls = len(records) * len(DIRECTIONS) * len(RANKS)
        if verification["verified_recalls"] != expected_recalls or not verification["test_queries_not_filtered"]:
            raise ValueError("Sinkhorn rank verification does not cover the complete fixed grid")
        pieces.append(f'<p>{len(records):,}개 ε·연결 개수·변환 공간 조합에 대해 저장된 질문별 정답 순위로 '
                      f'{expected_recalls:,}개 검색 지표를 다시 계산해 확인했다. '
                      '검색 질문이나 후보는 연결 제거 여부에 따라 제외하지 않았다.</p>')
    pieces.append('</section>')
    sources = (f'<p>Sinkhorn의 모든 ε·연결 개수·변환 공간에 대한 숫자 표는 '
               f'<code>{html.escape(str(out / "sinkhorn-summary.csv"))}</code>에 저장했다. '
               f'조건별 원본 검색 결과는 <code>{html.escape(str(out / "sinkhorn/results"))}</code>에 있다.</p>'
               f'<p>Sinkhorn 그림은 <code>{html.escape(str(out / "sinkhorn-epsilon-curves.png"))}</code>, '
               f'<code>{html.escape(str(out / "sinkhorn-connectivity.png"))}</code>, '
               f'<code>{html.escape(str(out / "sinkhorn-pruning-curves.png"))}</code>에 저장했다.</p>')
    return {
        "lead": _lead(records), "body": "".join(pieces), "sources": sources,
        "scope": ('<p>Sinkhorn의 계수 하나는 이미지 특징과 문장 특징의 비음수 대응 가중치다. '
                  'CCA의 계수는 입력 특징이 공통 좌표에 기여하는 양이고, Procrustes의 계수는 부호가 있는 변환 계수다. '
                  'Sinkhorn은 가중치 합으로 나누어 가중평균을 계산하므로 계수를 남기는 개수가 같아도 세 방법의 변환은 같지 않다. '
                  '검색 성능과 연결 구조를 함께 비교하되, 계수 수만으로 대응 관계의 품질을 판단할 수 없다.</p>'),
    }
