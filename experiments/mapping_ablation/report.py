"""Report fixed-mapping retrieval ablations without selecting new configurations."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RECALL_KS = (1, 5, 10)
METHODS = (
    "hungarian", "greedy", "topk", "partial_one", "partial_many",
    "same_budget_many", "global_top_edges", "sinkhorn", "sparse_transport",
    "sparse_factorization",
)
METHOD_LABELS = {
    "hungarian": "Hungarian, one-to-one",
    "greedy": "One highest score per feature",
    "topk": "Several highest scores per feature",
    "partial_one": "Optional one-to-one",
    "partial_many": "Bounded many-to-many",
    "same_budget_many": "Many-to-many, fixed edge budget",
    "global_top_edges": "Highest scores across all pairs",
    "sinkhorn": "Sinkhorn weighted correspondence",
    "sparse_transport": "Sparse weighted correspondence",
    "sparse_factorization": "Small feature groups",
    "cross_svd": "Cross-moment projection without whitening",
    "cca": "Canonical correlation with whitening",
    "procrustes": "Full orthogonal transformation",
}
METHOD_KO = {
    "hungarian": "Hungarian · 전체 점수 합을 최대화하는 일대일 대응",
    "greedy": "Greedy · 각 특징에서 최고점 연결 하나를 선택한 대응",
    "topk": "Top-k · 각 특징에서 상위 점수 연결 여러 개를 선택한 대응",
    "partial_one": "연결 생략을 허용한 일대일 대응",
    "partial_many": "특징마다 연결 수를 제한한 다대다 대응",
    "same_budget_many": "전체 연결 수도 일대일 대응과 같게 제한한 다대다 대응",
    "global_top_edges": "전체 특징 쌍에서 최고점 연결을 선택한 대응",
    "sinkhorn": "Sinkhorn · 행과 열의 총 가중치를 고정한 가중 대응",
    "sparse_transport": "일부 가중치가 0이 되도록 구한 희소한 가중 대응",
    "sparse_factorization": "작은 특징 집합을 통해 구성한 대응",
    "cross_svd": "SVD · 추가 상관 보정 없이 공동변동 방향에 투영한 조건",
    "cca": "CCA · 분산과 특징 간 상관을 추가로 보정한 정준상관분석",
    "procrustes": "Procrustes · 전체 차원을 유지하는 직교변환",
}
SPACES = {
    "text_projected_to_image": "텍스트를 이미지 특징 공간으로 옮겨 비교",
    "image_projected_to_text": "이미지를 텍스트 특징 공간으로 옮겨 비교",
    "common": "양쪽을 같은 차원의 공통 공간으로 옮겨 비교",
}
SPACE_EN = {
    "text_projected_to_image": "Text projected into image feature space",
    "image_projected_to_text": "Image projected into text feature space",
    "common": "Paired common coordinates",
}
DIRECTIONS = {
    "image_to_text": "이미지로 캡션 검색",
    "text_to_image": "캡션으로 이미지 검색",
}
DIRECTION_EN = {"image_to_text": "Image queries, caption candidates",
                "text_to_image": "Caption queries, image candidates"}
PREPROCESSING = {
    "raw": ("원래 활성값", "Original activations", "#8c97a6"),
    "centered": ("학습 평균을 뺀 활성값", "Training mean subtracted", "#e2a44c"),
    "standardized": ("학습 평균을 빼고 학습 표준편차로 나눈 활성값",
                     "Training mean and standard deviation", "#318d9b"),
}
GROUP_CONDITIONS = {
    "raw_factors": ("분해 가중치를 그대로 사용한 집합 좌표", "Direct groups, original factor weights"),
    "weighted_average": ("집합 안의 가중치 합으로 나눈 집합 좌표", "Direct groups, weighted averages"),
    "unit_variance": ("학습 자료의 집합 분산까지 보정한 집합 좌표", "Direct groups, training variance normalized"),
    "unnormalized_mapping": ("행과 열의 합으로 나누지 않은 특징 간 대응",
                             "Feature correspondence without weight normalization"),
}
CSV_FIELDS = (
    "key", "experiment", "family", "method", "condition", "condition_description",
    "space", "space_description", "dimensions", "whitening", "ridge", "direction",
    "query_count", "candidate_count", "recall_at_1", "recall_at_5", "recall_at_10",
)


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _dimensions(record: dict) -> int | None:
    value = record.get("metadata", {}).get("dimensions")
    return int(value) if value is not None else None


def _condition_label(record: dict) -> str:
    condition = record["condition"]
    if record["experiment"] == "preprocessing":
        return PREPROCESSING.get(condition, (condition,))[0]
    if record["experiment"] == "feature_groups":
        return GROUP_CONDITIONS.get(condition, (condition,))[0]
    dimension = _dimensions(record)
    if record["family"] == "procrustes":
        return f"이미지의 전체 {dimension:,}차원을 유지" if dimension is not None else "전체 차원을 유지"
    metadata = record.get("metadata", {})
    whitening = metadata.get("whitening", metadata.get("whiten", record["family"] == "cca"))
    adjustment = "분산과 특징 간 상관을 함께 보정" if whitening else "분산과 특징 간 상관을 추가로 보정하지 않음"
    return f"공통 {dimension:,}차원, {adjustment}" if dimension is not None else adjustment


def _score(record: dict | None, direction: str, k: int = 10) -> float:
    if record is None:
        return float("nan")
    recalls = record["retrieval"][direction]["recall"]
    return float(recalls[str(k)] if str(k) in recalls else recalls[k])


def _load_records(output: Path) -> list[dict]:
    records, seen = [], set()
    for path in sorted((output / "results").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        for field in ("key", "family", "experiment", "condition", "space", "retrieval"):
            if field not in record:
                raise ValueError(f"{path} is missing {field}")
        if record["key"] in seen:
            raise ValueError(f"Duplicate result key {record['key']!r} in {path}")
        for direction in DIRECTIONS:
            for k in RECALL_KS:
                score = _score(record, direction, k)
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f"Invalid Recall@{k} for {direction} in {path}")
        seen.add(record["key"])
        records.append(record)
    if not records:
        raise ValueError(f"No result JSON files found in {output / 'results'}")
    return records


def _csv_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        metadata = record.get("metadata", {})
        for direction in DIRECTIONS:
            metric = record["retrieval"][direction]
            rows.append({
                "key": record["key"], "experiment": record["experiment"],
                "family": record["family"],
                "method": METHOD_KO.get(record["family"], record["family"]),
                "condition": record["condition"], "condition_description": _condition_label(record),
                "space": record["space"], "space_description": SPACES.get(record["space"], record["space"]),
                "dimensions": _dimensions(record),
                "whitening": metadata.get("whitening", metadata.get("whiten")),
                "ridge": metadata.get("ridge"), "direction": direction,
                "query_count": metric.get("query_count"), "candidate_count": metric.get("candidate_count"),
                **{f"recall_at_{k}": _score(record, direction, k) for k in RECALL_KS},
            })
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(fig, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _axis_finish(ax, title: str) -> None:
    ax.set_title(title, fontsize=11, pad=12)
    ax.set_xlabel("Recall@10 (%)")
    ax.set_xlim(left=0)
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)


def _preprocessing_plot(records: list[dict], output: Path) -> None:
    index = {(r["family"], r["condition"], r["space"]): r for r in records}
    methods = [m for m in METHODS if any(r["family"] == m for r in records)]
    fig, axes = plt.subplots(2, 2, figsize=(19, max(8, .7 * len(methods) + 7)))
    spaces = tuple(s for s in SPACES if s != "common")
    for row, space in enumerate(spaces):
        for col, direction in enumerate(DIRECTIONS):
            ax = axes[row, col]
            for offset, (condition, (_, label, color)) in zip((-.25, 0, .25), PREPROCESSING.items(), strict=True):
                values = [_score(index.get((m, condition, space)), direction) * 100 for m in methods]
                ax.barh(np.arange(len(methods)) + offset, values, height=.22, label=label, color=color)
            ax.set_yticks(np.arange(len(methods)), [METHOD_LABELS[m] for m in methods], fontsize=9)
            ax.invert_yaxis()
            _axis_finish(ax, SPACE_EN[space] + "\n" + DIRECTION_EN[direction])
            if not methods:
                ax.text(.5, .5, "No results available", transform=ax.transAxes, ha="center")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .045, 1, 1))
    _save_figure(fig, output, "preprocessing_r10")


def _cca_plot(records: list[dict], output: Path) -> None:
    dimensions = sorted({dimension for r in records if (dimension := _dimensions(r)) is not None})
    positions = {dim: i for i, dim in enumerate(dimensions)}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7), sharey=True)
    styles = (("cross_svd", "Without whitening", "#447ca6", "o"),
              ("cca", "With whitening (canonical correlation)", "#228578", "s"),
              ("procrustes", "Full orthogonal transformation", "#8a8296", "D"))
    values = [100 * _score(r, direction) for r in records for direction in DIRECTIONS]
    upper_limit = min(105, max(values, default=0) * 1.08 + 1)
    for ax, direction in zip(axes, DIRECTIONS, strict=True):
        for family, label, color, marker in styles:
            selected = sorted((r for r in records if r["family"] == family and _dimensions(r) is not None),
                              key=lambda r: _dimensions(r) or 0)
            if selected:
                ax.plot([positions[int(r["metadata"]["dimensions"])] for r in selected],
                        [100 * _score(r, direction) for r in selected],
                        label=label, color=color, marker=marker, markersize=7,
                        linestyle="none" if family == "procrustes" else "-")
        ax.set_xticks(list(positions.values()), [f"{d:,}" for d in dimensions])
        ax.set_xlabel("Number of output coordinates")
        ax.set_ylabel("Recall@10 (%)")
        ax.set_title(DIRECTION_EN[direction])
        ax.set_ylim(0, upper_limit)
        ax.grid(alpha=.2)
        if not records:
            ax.text(.5, .5, "No results available", transform=ax.transAxes, ha="center")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1, frameon=False)
    fig.tight_layout(rect=(0, .17, 1, 1))
    _save_figure(fig, output, "cca_components_r10")


def _group_comparisons(records: list[dict]) -> list[tuple[str, str, str, dict]]:
    comparisons = []
    for space, suffix in (("text_projected_to_image", "image space"),
                          ("image_projected_to_text", "text space")):
        selected = [r for r in records if r["experiment"] == "preprocessing"
                    and r["family"] == "sparse_factorization" and r["condition"] == "standardized"
                    and r["space"] == space]
        comparisons.extend((f"Normalized feature correspondence, {suffix}",
                            "기존 정규화", "#cf9551", r) for r in selected)
    for space, suffix in (("text_projected_to_image", "image space"),
                          ("image_projected_to_text", "text space")):
        comparisons.extend((f"Unnormalized feature correspondence, {suffix}", "가중치 합으로 나누지 않음", "#e4bd8a", r)
                           for r in records if r["experiment"] == "feature_groups"
                           and r["condition"] == "unnormalized_mapping" and r["space"] == space)
    for condition in ("raw_factors", "weighted_average", "unit_variance"):
        comparisons.extend((GROUP_CONDITIONS[condition][1], "집합 좌표", "#9c86b5", r)
                           for r in records if r["experiment"] == "feature_groups"
                           and r["condition"] == condition)
    for family in ("cross_svd", "cca"):
        comparisons.extend((METHOD_LABELS[family] + ", 256 coordinates", "공통 공간 참고값", "#5f9ca4", r)
                           for r in records if r["experiment"] == "cca_components"
                           and r["family"] == family and _dimensions(r) == 256)
    return comparisons


def _group_plot(comparisons: list[tuple[str, str, str, dict]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, max(5.5, .58 * len(comparisons) + 2)), sharey=True)
    for ax, direction in zip(axes, DIRECTIONS, strict=True):
        values = [100 * _score(r, direction) for _, _, _, r in comparisons]
        ax.barh(np.arange(len(values)), values, color=[c for _, _, c, _ in comparisons], height=.65)
        ax.set_yticks(np.arange(len(values)), [label for label, _, _, _ in comparisons], fontsize=9)
        for i, value in enumerate(values):
            ax.text(value + .12, i, f"{value:.2f}", va="center", fontsize=9)
        _axis_finish(ax, DIRECTION_EN[direction])
        if values:
            ax.set_xlim(0, min(110, max(values) * 1.14 + 1))
        else:
            ax.text(.5, .5, "No results available", transform=ax.transAxes, ha="center")
    axes[0].invert_yaxis()
    fig.tight_layout()
    _save_figure(fig, output, "feature_groups_r10")


def _table(records: list[dict], *, group_reference: bool = False) -> str:
    if not records:
        return "<p class='pending'>이 비교의 결과 파일이 아직 없습니다.</p>"
    header = "<thead><tr><th rowspan='2'>방법과 비교 조건</th><th rowspan='2'>비교 공간</th>"
    header += "".join(f"<th colspan='3'>{label}</th>" for label in DIRECTIONS.values())
    header += "</tr><tr>" + "".join(f"<th>상위 {k}개</th>" for _ in DIRECTIONS for k in RECALL_KS)
    header += "</tr></thead>"
    body = []
    for record in records:
        label = METHOD_KO.get(record["family"], record["family"])
        condition = _condition_label(record)
        if group_reference and record["experiment"] == "preprocessing":
            condition = "행 또는 열의 가중치 합으로 나눈 기존 대응에 표준화한 활성값을 사용"
        metadata = record.get("metadata", {})
        dimension = _dimensions(record)
        details = [condition]
        if dimension is not None and record["experiment"] != "cca_components":
            dimension_name = "분해 집합" if record["condition"] == "unnormalized_mapping" else "결과 좌표"
            details.append(f"{dimension_name} {dimension:,}개")
        if record["family"] == "cca" and metadata.get("ridge") is not None:
            details.append(f"공분산 행렬 대각에 더한 값 {metadata['ridge']:g}")
        cells = "".join(f"<td>{100 * _score(record, direction, k):.2f}%</td>"
                        for direction in DIRECTIONS for k in RECALL_KS)
        body.append(f"<tr><th scope='row'>{_escape(label)}<span class='detail'>{_escape(' · '.join(details))}"
                    f"</span></th><td class='space'>{_escape(SPACES.get(record['space'], record['space']))}"
                    f"</td>{cells}</tr>")
    return "<div class='table-scroll'><table>" + header + "<tbody>" + "".join(body) + "</tbody></table></div>"


def _population(output: Path, records: list[dict]) -> tuple[int | None, int | None]:
    path = output / "population.json"
    population = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    first = records[0]["retrieval"]
    return (population.get("test_images", first["image_to_text"].get("query_count")),
            population.get("test_captions", first["text_to_image"].get("query_count")))


def report(output: Path) -> Path:
    """Read ``results/*.json`` and return the generated HTML report's path."""
    output = Path(output)
    records = _load_records(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    preprocessing = [r for r in records if r["experiment"] == "preprocessing"]
    components = sorted((r for r in records if r["experiment"] == "cca_components"),
                        key=lambda r: (r["family"] == "procrustes", _dimensions(r) or 0, r["family"]))
    comparisons = _group_comparisons(records)
    rows = _csv_rows(records)
    _write_csv(output / "retrieval_summary.csv", rows)
    for name in ("preprocessing", "cca_components", "feature_groups"):
        _write_csv(output / f"{name}.csv", [r for r in rows if r["experiment"] == name])
    _write_csv(output / "feature_group_comparison.csv", _csv_rows([r for _, _, _, r in comparisons]))
    with plt.rc_context({"font.size": 10, "font.family": "DejaVu Sans",
                         "axes.spines.top": False, "axes.spines.right": False}):
        _preprocessing_plot(preprocessing, figures)
        _cca_plot(components, figures)
        _group_plot(comparisons, figures)

    images, captions = _population(output, records)
    population_path = output / "population.json"
    population = json.loads(population_path.read_text(encoding="utf-8")) if population_path.exists() else {}
    image_width, text_width = population.get("image_features"), population.get("text_features")
    if image_width is not None and text_width is not None:
        common_width = min(image_width, text_width)
        dimension_note = (
            f"전체 공통 차원 {common_width:,}개를 남기는 보정 없는 투영은 "
            f"{image_width:,}차원 이미지 공간에서 나머지 {image_width - common_width:,}개 방향을 제거합니다. "
            f"{image_width:,}차원 직교변환은 벡터의 길이와 각도를 보존하는 회전 또는 반사 변환으로 "
            f"전체 이미지 차원을 유지합니다. 따라서 {common_width:,}차원 공통 투영과 "
            f"{image_width:,}차원 직교변환은 출력 공간도 서로 다른 참고 조건입니다."
        )
    else:
        dimension_note = (
            "전체 공통 차원을 남기는 보정 없는 투영은 더 큰 쪽 특징 공간의 나머지 방향을 제거합니다. "
            "전체 직교변환은 벡터의 길이와 각도를 보존하는 회전 또는 반사 변환으로 전체 차원을 유지합니다. "
            "따라서 두 조건은 출력 공간도 서로 다른 참고 조건입니다."
        )
    ridge_values = sorted({float(r["metadata"]["ridge"]) for r in components
                           if r["family"] == "cca" and r.get("metadata", {}).get("ridge") is not None})
    ridge_note = ("정준상관분석에서 공분산 행렬의 대각에 더한 값은 "
                  + ", ".join(f"{value:g}" for value in ridge_values) + "입니다. "
                  if ridge_values else "공분산 행렬 대각에 더하는 값은 각 결과에 저장된 설정을 사용했습니다. ")
    interpretation_path = output / "interpretation.json"
    interpretation = (json.loads(interpretation_path.read_text(encoding="utf-8"))
                      if interpretation_path.exists() else {})
    sections = interpretation.get("sections", {})

    def section_result(name: str) -> str:
        value = sections.get(name)
        return f"<p class='result'>{_escape(value)}</p>" if value else ""

    lead = interpretation.get("lead") or (
        "입력 활성값의 처리, 공통 공간의 차원과 추가 상관 보정, 특징 집합을 비교하는 방식을 각각 바꿨습니다. "
        "기존에 선택한 대응을 유지하면서 각 변경이 검색 재현율에 미치는 영향을 비교합니다."
    )
    group_reference_note = (
        "같은 256차원의 보정 없는 투영과 정준상관분석도 참고값으로 표시했습니다. "
        if any(r["experiment"] == "cca_components" for _, _, _, r in comparisons) else ""
    )
    population_text = (f"이미지 {images:,}장과 캡션 {captions:,}개를 같은 질의와 후보로 사용했습니다. "
                       if images is not None and captions is not None else
                       "기존 실험과 같은 이미지와 캡션을 질의와 후보로 사용했습니다. ")
    body = ["<h1>특징 대응 검색의 세 가지 조건 비교</h1>",
            f"<p class='lead'>{_escape(lead)}</p>",
            "<nav aria-label='목차'><a href='#preprocessing'>입력값 처리 비교. 같은 대응에 넣는 활성값만 바꿉니다.</a>"
            "<a href='#components'>공통 공간 비교. 좌표 수와 특징 간 상관 보정의 효과를 구분합니다.</a>"
            "<a href='#groups'>특징 집합 비교. 특징 간 대응과 집합 좌표의 비교 방식을 구분합니다.</a></nav>",
            "<p>희소 자동부호화기는 모델의 내부 벡터를 소수의 활성 특징으로 표현하는 모델입니다. "
            "이번 비교에서는 희소 자동부호화기와 원래 임베딩 모델을 고정했습니다. "
            "특징 활성값은 각 입력에서 그 특징이 얼마나 강하게 반응했는지를 나타내는 수치입니다. "
            "특징 대응과 집합을 결정한 기존 설정도 고정했습니다. 기존 검증 자료에서 양방향 예측 오차의 제곱 평균으로 "
            "선택한 설정을 사용했으며, 이번 검색 결과에 맞춰 다시 선택하지 않았습니다.</p>",
            f"<p>{population_text}평균, 표준편차와 공분산은 학습 자료에서 계산한 값만 사용했습니다. "
            "이미지와 캡션의 짝 및 평가 자료도 모든 조건에서 유지했습니다.</p>",
            "<p>표의 검색 재현율은 정답이 검색 상위 1개, 5개 또는 10개 안에 들어간 질의의 비율입니다. "
            "이미지로 캡션을 찾을 때는 정답 캡션 중 하나 이상이 들어가면 성공입니다. "
            "캡션으로 이미지를 찾을 때는 해당 이미지가 들어가면 성공입니다. "
            "그림의 Recall@10은 이 중 상위 10개 검색 재현율을 백분율로 표시한 값입니다. "
            "검색 재현율은 특징들이 같은 개념을 나타내는지를 직접 판정한 정확도와 다릅니다.</p>",
            f"<p>현재 결과 파일 {_escape(len(records))}개를 포함했습니다. "
            "<a href='retrieval_summary.csv'>전체 검색 결과 CSV</a>의 재현율은 0부터 1 사이 값으로 저장했습니다. "
            "표에서는 같은 값을 백분율로 표시했습니다.</p>",
            "<h2 id='preprocessing'>입력 활성값의 처리만 바꾼 비교</h2>",
            section_result("preprocessing"),
            "<p>각 대응의 연결과 가중치를 그대로 유지하고, 양쪽 활성값에 적용하는 처리만 바꿨습니다. "
            "원래 값을 쓰는 조건, 학습 평균을 빼는 조건, 학습 평균을 뺀 뒤 학습 표준편차로 나누는 조건을 비교합니다. "
            "평균 제거는 각 특징의 평소 반응을 빼는 연산입니다. 표준편차로 나누면 특징별 변동 크기를 같은 단위로 비교합니다. "
            "이 처리는 검색 시 사용하는 벡터에 적용하며, 특징 간 상관 점수나 대응 가중치를 다시 학습하지 않습니다.</p>",
            "<p>그림의 위쪽은 텍스트를 이미지 특징 공간으로 옮긴 결과이고, 아래쪽은 이미지를 텍스트 특징 공간으로 옮긴 결과입니다. "
            "왼쪽은 이미지로 캡션을 찾고, 오른쪽은 캡션으로 이미지를 찾습니다.</p>",
            "<figure><img src='figures/preprocessing_r10.svg' alt='열 가지 고정 대응에서 세 입력 처리 조건의 양방향 상위 10개 검색 재현율'>"
            "<figcaption>같은 방법의 세 막대를 비교하면 대응 구조를 고정한 상태에서 입력값 처리의 영향을 확인할 수 있습니다.</figcaption></figure>"]
    for space in ("text_projected_to_image", "image_projected_to_text"):
        subset = [r for r in preprocessing if r["space"] == space]
        subset.sort(key=lambda r: (METHODS.index(r["family"]) if r["family"] in METHODS else len(METHODS),
                                   tuple(PREPROCESSING).index(r["condition"]) if r["condition"] in PREPROCESSING else 99))
        body.extend([f"<h3>{SPACES[space]}</h3>", _table(subset)])
    body.extend([
        "<p><a href='preprocessing.csv'>입력값 처리 비교 CSV</a>에서 모든 조건의 상위 1개, 5개, 10개 검색 재현율을 확인할 수 있습니다.</p>",
        "<h2 id='components'>공통 공간의 차원과 상관 보정을 바꾼 비교</h2>",
        section_result("cca_components"),
        "<p>두 조건 모두 각 특징에서 학습 평균을 빼고 학습 표준편차로 나눈 활성값을 입력으로 사용합니다. "
        "이 상태에서 두 자료에서 함께 변하는 방향을 구하고, 같은 수의 방향을 남겨 비교합니다. "
        "추가 보정이 없는 조건은 이미지와 텍스트 사이 공동변동 행렬의 특이값분해로 방향을 구합니다. "
        "특이값분해는 이 행렬을 두 자료의 방향과 각 방향의 공동변동 크기로 나누는 계산입니다. "
        "정준상관분석은 각 자료 내부의 분산과 특징 간 상관을 추가로 보정한 뒤 두 자료에서 대응하는 방향을 구합니다. "
        "그림에서 without whitening은 추가 보정이 없는 조건, with whitening은 이 보정이 있는 조건입니다.</p>",
        "<p>같은 차원 수에서 두 조건을 비교하면 상관 보정의 영향을 확인할 수 있습니다. "
        "같은 보정 조건에서 차원 수를 비교하면 남긴 방향 수의 영향을 확인할 수 있습니다. "
        + ridge_note + "이 값은 행렬 계산을 안정화하는 데 사용합니다. "
        "공분산은 특징들이 함께 변하는 정도를 나타냅니다. 두 투영 모두 공동변동 크기를 좌표에 추가로 곱하지 않습니다.</p>",
        f"<p>{dimension_note}</p>",
        "<figure><img src='figures/cca_components_r10.svg' alt='공통 좌표 수와 특징 간 상관 보정에 따른 양방향 상위 10개 검색 재현율'>"
        "<figcaption>같은 가로축 값에서 두 선을 비교합니다. 전체 직교변환은 별도 점으로 표시했습니다.</figcaption></figure>",
        _table(components), "<p><a href='cca_components.csv'>공통 공간 비교 CSV</a>에는 차원 수와 상관 보정 조건을 함께 저장했습니다.</p>",
        "<h2 id='groups'>같은 특징 집합을 비교하는 방식을 바꾼 비교</h2>",
        section_result("feature_groups"),
        "<p>기존 비음수 분해에서 구한 이미지 쪽 가중치 U와 텍스트 쪽 가중치 V를 고정했습니다. "
        "비음수 분해는 특징 사이의 대응 행렬을 음수가 없는 두 행렬의 곱으로 표현하는 계산입니다. "
        "한 열에 가중치가 있는 특징들이 하나의 집합을 이룹니다. 이 집합이 하나의 의미 개념인지는 별도로 확인해야 합니다.</p>",
        "<p>먼저 특징 간 대응 행렬 P = UVᵀ를 사용해 한쪽을 다른 쪽 특징 공간으로 옮깁니다. "
        "이때 기존처럼 행 또는 열의 가중치 합으로 나누는 조건과 나누지 않는 조건을 비교합니다. "
        "다음으로 이미지 벡터 x와 텍스트 벡터 y에서 각각 xU와 yV를 계산해 집합 좌표끼리 직접 비교합니다. "
        "직접 비교에서는 분해 가중치를 그대로 사용하는 조건, 각 집합의 가중치 합으로 나누는 조건, "
        "학습 자료에서 구한 집합의 표준편차로도 나누는 조건을 비교합니다. 모든 조건의 입력 활성값에는 학습 평균과 표준편차를 적용했습니다.</p>",
        "<p>집합 좌표의 내적은 (xU)(yV)ᵀ = xPyᵀ입니다. 따라서 분해 가중치를 그대로 사용하는 집합 좌표는 "
        "새로운 개념을 추가로 학습한 결과가 아닙니다. 코사인 유사도는 내적을 두 벡터 길이의 곱으로 나눈 값이므로, "
        "직접 집합 비교와 특징 공간 비교는 같은 내적을 사용하더라도 길이를 나누는 방식에 따라 검색 순위가 달라질 수 있습니다. "
        "집합별 평균과 분산 보정도 집합 사이의 상대적인 크기를 바꿉니다.</p>",
        "<p>그림에는 두 특징 공간에서 평가한 기존 대응과 가중치 합으로 나누지 않은 대응을 함께 표시했습니다. "
        "세 가지 집합 좌표 조건은 별도 공통 공간을 사용합니다. "
        + group_reference_note + "가중 평균과 집합 분산 보정 조건에서는 학습 자료에서 양쪽 중 한쪽의 집합이 비어 있거나 "
        "분산이 0이면 해당 집합을 제외하며, 실제 좌표 수는 결과 표에 표시합니다.</p>",
        "<figure><img src='figures/feature_groups_r10.svg' alt='특징 간 대응과 직접 집합 좌표 및 256차원 공통 투영의 양방향 상위 10개 검색 재현율'>"
        "<figcaption>비교 공간과 가중치 정규화를 구분해 해석해야 합니다. 막대 하나가 새로운 특징 집합 학습을 뜻하지는 않습니다.</figcaption></figure>",
        _table([r for _, _, _, r in comparisons], group_reference=True),
        "<p><a href='feature_groups.csv'>특징 집합 조건 CSV</a>와 "
        "<a href='feature_group_comparison.csv'>그림에 포함한 전체 비교 CSV</a>를 제공합니다.</p>",
        "<p class='note'>이 비교에서는 선택한 설정을 유지한 채 검색 방식을 점검합니다. "
        "검색 재현율의 변화만으로 집합의 의미가 더 정확해졌거나, 우연한 동시 출현에 따른 상관이 제거됐다고 판단할 수는 없습니다.</p>",
    ])
    if (output / "contrasts.csv").exists():
        body.append(
            "<p><a href='contrasts.csv'>조건 간 차이와 95% 신뢰구간 CSV</a>도 제공합니다. "
            "같은 이미지에 속한 캡션을 함께 묶어 1,000회 재표집했습니다. "
            "이 신뢰구간은 평가 표본에 따른 변동을 나타내며, 재학습에 따른 변동은 포함하지 않습니다. "
            "여러 비교를 동시에 수행한 데 대한 보정은 적용하지 않았습니다.</p>"
        )
    css = """
    body{font:16px/1.75 system-ui,-apple-system,sans-serif;color:#283442;max-width:1500px;margin:44px auto;padding:0 28px}
    h1{font-size:30px;line-height:1.35}h2{margin-top:58px;padding-top:18px;border-top:2px solid #c8d5db}
    h3{margin-top:28px}p{max-width:1150px}.lead{font-size:18px}nav{display:grid;gap:9px;padding:20px;background:#f0f5f5;margin:25px 0}
    a{color:#176573;text-underline-offset:3px}figure{margin:28px 0}figure img{width:100%;height:auto}figcaption{color:#536575;font-size:14px}
    .table-scroll{overflow:auto;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
    th,td{border-bottom:1px solid #d7dfe4;padding:10px 12px;text-align:right;white-space:nowrap}thead{background:#edf3f5}
    thead th[colspan]{text-align:center}th[scope=row]{text-align:left;min-width:300px;white-space:normal;font-weight:600}
    .detail{display:block;font-weight:400;font-size:12px;color:#5b6b78}.space{white-space:normal;min-width:160px;text-align:left}
    .pending,.note{padding:15px 18px;background:#f6f5ee}.result{padding:15px 18px;background:#edf5f4;border-left:4px solid #318d9b}
    .note{margin-top:35px}tr:hover td,tr:hover th[scope=row]{background:#f5f8f8}
    @media(max-width:700px){body{padding:0 16px;margin-top:24px}h1{font-size:25px}h2{font-size:21px}}
    """
    path = output / "report.html"
    path.write_text("<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
                    "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                    "<title>특징 대응 검색의 세 가지 조건 비교</title><style>" + css + "</style></head><body>"
                    + "".join(body) + "</body></html>", encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Directory containing results/*.json")
    print(report(parser.parse_args().output))
