"""Render a self-contained Korean report from completed semantic evaluations."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image


Record = dict[str, Any]
DEFAULT_RUN = Path(__file__).resolve().parents[2] / "runs/mapping-semantics-2026-10-04"
DIRECTIONS = ("image_to_text", "text_to_image")
GROUPS = ("object", "background")
GROUP_NAMES = {"object": "사물", "background": "배경"}
DIRECTION_NAMES = {"image_to_text": "이미지에서 고른 좌표를 문장에 적용",
                   "text_to_image": "문장에서 고른 좌표를 이미지에 적용"}
RETRIEVAL_NAMES = {"image_to_text": "이미지로 문장 검색", "text_to_image": "문장으로 이미지 검색"}
SPACE_NAMES = {"image": "이미지 특징 공간", "text": "문장 특징 공간", "common": "공통 좌표 공간"}
FAMILY_NAMES = {"hungarian": "헝가리안 일대일 대응", "cca": "CCA의 계수를 학습 후 제거",
                "procrustes": "Procrustes의 연결을 학습 후 제거", "sinkhorn": "Sinkhorn의 연결을 학습 후 제거",
                "signed": "고정 연결에서 음수를 허용한 회귀", "nonnegative": "고정 연결에서 음수를 금지한 회귀",
                "cca_renormalized": "CCA 계수 제거 후 좌표 크기 재정규화", "sparse_cca": "특징 수를 제한하며 CCA 학습"}
COLORS = {"hungarian": "#718096", "cca": "#24669b", "procrustes": "#95633b", "sinkhorn": "#8b4b83",
          "signed": "#95633b", "nonnegative": "#cb8662", "cca_renormalized": "#5a99bc", "sparse_cca": "#1d8469"}
FIELDS = ("source_auc", "target_transfer_auc", "paired_minimum_auc")
FIELD_NAMES = {"source_auc": "좌표를 고른 쪽의 AUROC", "target_transfer_auc": "대상 쪽의 AUROC",
               "paired_minimum_auc": "범주별 양쪽 AUROC의 최솟값"}


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _label(record: Record) -> str:
    if record["family"] == "sparse_cca":
        return f'학습하며 좌표당 최대 {record["options"]["k"]}개 입력을 사용한 CCA'
    return record["label"].replace("양수만 허용", "음수 금지").replace("Sinkhorn ε=", "Sinkhorn 연결 분산 벌점 ")


def _number(value, digits: int = 3) -> str:
    return "계산 불가" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _rows(record: Record, direction: str, group: str) -> list[Record]:
    return [r for r in record["semantic"] if r["direction"] == direction and r["group"] == group]


def _summary(record: Record, direction: str, group: str, field: str) -> tuple[float | None, int, int]:
    rows = _rows(record, direction, group)
    values = [r[field] for r in rows if r[field] is not None and np.isfinite(r[field])]
    return (float(np.mean(values)) if values else None, len(values), len(rows))


def _recall(record: Record, direction: str, rank: int) -> float:
    return 100 * float(record["retrieval"][direction]["recall"][str(rank)])


def _structure(record: Record) -> Record:
    # An identity matrix is a fixed copy operation, not fitted correspondence coefficients.
    fitted_sides = ("image", "text") if record["space"] == "common" else (
        "text" if record["space"] == "image" else "image",)
    structure = record["structure"]
    count = sum(structure[s]["nonzero_coefficients"] for s in fitted_sides)
    negative = sum(structure[s]["negative_coefficients"] for s in fitted_sides)
    return {"transform_nonzero_coefficients": count, "transform_negative_coefficients": negative,
            "transform_negative_fraction": negative / count if count else 0,
            "image_max_inputs_per_coordinate": structure["image"]["max_per_column"],
            "text_max_inputs_per_coordinate": structure["text"]["max_per_column"]}


def _order(record: Record):
    family_order = {key: i for i, key in enumerate(FAMILY_NAMES)}
    k = (record.get("options") or {}).get("k")
    return (family_order[record["family"]], -1 if k is None else k,
            ("common", "image", "text").index(record["space"]))


def _validate(records: list[Record], protocol: Record) -> None:
    expected_ids = {r["id"] for r in protocol.get("concepts", [])}
    for record in records:
        for direction in DIRECTIONS:
            rows = [r for r in record["semantic"] if r["direction"] == direction]
            if len(rows) != len(expected_ids) or {r["id"] for r in rows} != expected_ids:
                raise ValueError(f"Concept denominator changed for {record['key']} {direction}")
            for row in rows:
                for field in FIELDS:
                    if row[field] is not None and not 0 <= row[field] <= 1:
                        raise ValueError("Semantic AUROC is outside [0, 1]")
                values = [row["source_auc"], row["target_transfer_auc"]]
                expected = min(values) if None not in values else None
                if row["paired_minimum_auc"] != expected:
                    raise ValueError("Paired minimum differs from the two frozen-polarity AUROCs")
            metric = record["retrieval"][direction]
            expected_count = (protocol["evaluation_images"], protocol["evaluation_captions"])
            if direction == "text_to_image":
                expected_count = expected_count[::-1]
            if (metric["query_count"], metric["candidate_count"]) != expected_count:
                raise ValueError("Retrieval population differs between saved conditions")
            for rank in (1, 5, 10):
                np.testing.assert_allclose(float(metric["recall"][str(rank)]),
                                           np.mean(np.asarray(metric["ranks"]) <= min(rank, metric["candidate_count"])))


def _csv(path: Path, rows: list[Record], empty_fields: tuple[str, ...] = ("status",)) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else list(empty_fields)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _export(out: Path, records: list[Record]) -> None:
    summaries, concepts = [], []
    for record in records:
        metadata = {"key": record["key"], "section": record["section"], "family": record["family"],
                    "condition": _label(record), "space": record["space"], **_structure(record)}
        for direction in DIRECTIONS:
            for group in GROUPS:
                row = {**metadata, "direction": direction, "group": group}
                for rank in (1, 5, 10):
                    row[f"recall_at_{rank}_percent"] = _recall(record, direction, rank)
                for field in FIELDS:
                    value, valid, total = _summary(record, direction, group, field)
                    row[field] = value
                    row[field + "_valid_concepts"] = valid
                    row[field + "_total_concepts"] = total
                summaries.append(row)
        concepts.extend({**metadata, **row} for row in record["semantic"])
    _csv(out / "summary.csv", summaries)
    _csv(out / "semantic_per_concept.csv", concepts)


def _paired_difference(left: Record, right: Record, direction: str, group: str, field: str) -> Record:
    """Difference is left minus right with category-level paired resampling."""
    lhs = {r["id"]: r[field] for r in _rows(left, direction, group)}
    rhs = {r["id"]: r[field] for r in _rows(right, direction, group)}
    ids = sorted(lhs.keys() | rhs.keys())
    valid = [c for c in ids if lhs.get(c) is not None and rhs.get(c) is not None]
    values = np.array([lhs[c] - rhs[c] for c in valid])
    mean, low, high = None, None, None
    if len(values):
        rng = np.random.default_rng(42)
        estimates = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
        mean = float(values.mean())
        low, high = [float(v) for v in np.quantile(estimates, [.025, .975])]
    return {"left_key": left["key"], "right_key": right["key"], "left_condition": _label(left),
            "right_condition": _label(right), "space": left["space"], "direction": direction,
            "group": group, "metric": field, "difference": mean, "ci_low": low, "ci_high": high,
            "valid_concepts": len(values), "total_concepts": len(ids), "bootstrap_unit": "category",
            "bootstrap_repeats": 2000, "bootstrap_seed": 42}


def _differences(records: list[Record]) -> list[Record]:
    lookup = {r["key"]: r for r in records}
    pairs = []
    for record in records:
        k = (record.get("options") or {}).get("k")
        if record["family"] == "nonnegative":
            key = f"procrustes_support_{k}_signed__{record['space']}"
            if key in lookup:
                pairs.append(("sign", record, lookup[key]))
        elif record["family"] == "sparse_cca":
            for key in (f"cca_{k}", f"cca_pruned_renormalized_{k}"):
                if key in lookup:
                    pairs.append(("sparse", record, lookup[key]))
    return [{"comparison": kind, **_paired_difference(left, right, direction, group, field)}
            for kind, left, right in pairs for direction in DIRECTIONS for group in GROUPS
            for field in ("target_transfer_auc", "paired_minimum_auc")]


def _metric_cell(record: Record, direction: str, field: str) -> str:
    lines = []
    for group in GROUPS:
        mean, n, total = _summary(record, direction, group, field)
        lines.append(f'{GROUP_NAMES[group]} {_number(mean)}<small>계산 가능한 범주 {n}/{total}개</small>')
    return "<br>".join(lines)


def _table(records: list[Record]) -> str:
    if not records:
        return '<p class="pending">이 비교의 계산 결과는 아직 저장되지 않았다.</p>'
    parts = ['<div class="table-scroll"><table><thead><tr><th>방법과 연결 제한</th><th>비교 공간</th>'
             '<th>좌표를 고른 방향</th><th>같은 방향의 상위 10개 검색 성공률</th><th>실제 입력 수와 계수</th>'
             '<th>좌표를 고른 쪽의 평균 AUROC</th><th>대상 쪽의 평균 AUROC</th>'
             '<th>범주별 양쪽 최솟값의 평균</th></tr></thead><tbody>']
    previous_family = None
    for record in sorted(records, key=_order):
        if record["family"] != previous_family:
            parts.append(f'<tr class="family"><th colspan="8">{FAMILY_NAMES[record["family"]]}</th></tr>')
            previous_family = record["family"]
        structural = _structure(record)
        detail = (f'좌표당 이미지 입력 최대 {structural["image_max_inputs_per_coordinate"]}개<br>'
                  f'좌표당 문장 입력 최대 {structural["text_max_inputs_per_coordinate"]}개<br>'
                  f'변환 계수 {structural["transform_nonzero_coefficients"]:,}개<br>'
                  f'그중 음수 {structural["transform_negative_coefficients"]:,}개')
        for index, direction in enumerate(DIRECTIONS):
            parts.append('<tr>')
            if index == 0:
                parts.append(f'<th rowspan="2">{html.escape(_label(record))}</th>'
                             f'<td rowspan="2">{SPACE_NAMES[record["space"]]}</td>')
            parts.append(f'<td>{DIRECTION_NAMES[direction]}</td><td>{_recall(record, direction, 10):.2f}%</td>')
            if index == 0:
                parts.append(f'<td rowspan="2">{detail}</td>')
            parts.extend(f'<td>{_metric_cell(record, direction, field)}</td>' for field in FIELDS)
            parts.append('</tr>')
    parts.append('</tbody></table></div>')
    return "".join(parts)


def _retrieval_details(records: list[Record]) -> str:
    parts = ['<details><summary>상위 1개·5개·10개의 검색 성공률을 모두 확인한다</summary>'
             '<div class="table-scroll"><table><thead><tr><th>방법과 연결 제한</th><th>비교 공간</th>'
             '<th>검색 방향</th><th>상위 1개 성공률</th><th>상위 5개 성공률</th><th>상위 10개 성공률</th>'
             '</tr></thead><tbody>']
    for record in sorted(records, key=_order):
        for direction in DIRECTIONS:
            parts.append(f'<tr><th>{html.escape(_label(record))}</th><td>{SPACE_NAMES[record["space"]]}</td>'
                         f'<td>{RETRIEVAL_NAMES[direction]}</td>')
            parts.extend(f'<td>{_recall(record, direction, k):.2f}%</td>' for k in (1, 5, 10))
            parts.append('</tr>')
    return "".join(parts) + '</tbody></table></div></details>'


def _font() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in ("AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR")
                 if f in available), "DejaVu Sans")


def _save_figure(out: Path, name: str, fig) -> None:
    fig.savefig(out / name, dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _figures(out: Path, records: list[Record], differences: list[Record]) -> list[str]:
    if not records:
        return []
    figures = []
    with plt.rc_context({"font.family": _font(), "font.size": 10, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1))
        plotted = [r for r in records if r["section"] in ("baseline", "sparse")]
        for record in plotted:
            for direction, marker in zip(DIRECTIONS, ("o", "^"), strict=True):
                for axis, field in zip(axes, ("target_transfer_auc", "paired_minimum_auc"), strict=True):
                    value, _, _ = _summary(record, direction, "object", field)
                    if value is not None:
                        axis.scatter(_recall(record, direction, 10), value, marker=marker,
                                     color=COLORS[record["family"]], alpha=.8, s=40, edgecolors="white", linewidths=.5)
        for axis, field in zip(axes, ("target_transfer_auc", "paired_minimum_auc"), strict=True):
            axis.set(xlabel="상위 10개 검색 성공률 (%)", ylabel=f"사물 범주 평균 {FIELD_NAMES[field]}")
            axis.axhline(.5, color="#abb5be", linestyle="--", linewidth=.8)
            axis.grid(alpha=.18)
        families = list(dict.fromkeys(r["family"] for r in plotted))
        handles = [Line2D([0], [0], marker="o", linestyle="none", color=COLORS[f], label=FAMILY_NAMES[f])
                   for f in families]
        handles.extend(Line2D([0], [0], marker=m, linestyle="none", color="#444", label=DIRECTION_NAMES[d])
                       for d, m in zip(DIRECTIONS, ("o", "^"), strict=True))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, -.18), ncol=3, frameon=False, fontsize=9)
        fig.suptitle("검색 성능과 고정 좌표의 개념 구분 성능은 서로 다른 측정값이다", fontsize=14)
        fig.tight_layout()
        _save_figure(out, "retrieval_semantics.png", fig)
        figures.append("retrieval_semantics.png")

        selected_keys = ["hungarian__image", "cca_full", "cca_8", "cca_pruned_renormalized_8", "sparse_cca_8",
                         "procrustes_8__image", "sinkhorn_e0.05_k8__image"]
        lookup = {r["key"]: r for r in records}
        selected = [lookup[k] for k in selected_keys if k in lookup]
        if selected:
            fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True, sharey=True)
            labels = [f'{_label(r)}\n{SPACE_NAMES[r["space"]]}' for r in selected]
            for row, direction in enumerate(DIRECTIONS):
                for column, group in enumerate(GROUPS):
                    axis = axes[row, column]
                    distributions = [[v["target_transfer_auc"] for v in _rows(record, direction, group)
                                      if v["target_transfer_auc"] is not None] for record in selected]
                    boxes = axis.boxplot([values if values else [np.nan] for values in distributions],
                                         vert=False, patch_artist=True, widths=.5, showfliers=False)
                    for box, record in zip(boxes["boxes"], selected, strict=True):
                        box.set(facecolor=COLORS[record["family"]], alpha=.45)
                    axis.set_yticks(np.arange(1, len(labels) + 1), labels=labels, fontsize=8)
                    axis.set_title(f'{GROUP_NAMES[group]} 범주 · {DIRECTION_NAMES[direction]}', fontsize=10)
                    axis.axvline(.5, color="#8c99a4", linestyle="--", linewidth=.8)
                    axis.set_xlim(0, 1)
                    axis.grid(axis="x", alpha=.2)
            axes[1, 0].set_xlabel("범주별 대상 쪽 AUROC")
            axes[1, 1].set_xlabel("범주별 대상 쪽 AUROC")
            fig.suptitle("대상 쪽 AUROC의 범주별 분포", fontsize=14)
            fig.tight_layout()
            _save_figure(out, "semantic_distributions.png", fig)
            figures.append("semantic_distributions.png")

        sign_rows = [r for r in differences if r["comparison"] == "sign"]
        if sign_rows:
            fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
            for axis, field in zip(axes, ("target_transfer_auc", "paired_minimum_auc"), strict=True):
                rows = [r for r in sign_rows if r["metric"] == field]
                labels = []
                for position, row in enumerate(rows):
                    k = lookup[row["left_key"]]["options"]["k"]
                    direction_name = "이미지 기준" if row["direction"] == "image_to_text" else "문장 기준"
                    labels.append(f'연결 상한 {k}개 · {SPACE_NAMES[row["space"]]}\n'
                                  f'{direction_name} · {GROUP_NAMES[row["group"]]} {row["valid_concepts"]}/{row["total_concepts"]}개')
                    if row["difference"] is not None:
                        value = row["difference"]
                        axis.errorbar(value, position,
                                      xerr=[[max(0, value - row["ci_low"])], [max(0, row["ci_high"] - value)]],
                                      fmt="o", markersize=4, capsize=2,
                                      color="#24669b" if row["group"] == "object" else "#95633b")
                axis.set_yticks(range(len(labels)), labels=labels, fontsize=8)
                axis.invert_yaxis()
                axis.axvline(0, color="#687580", linestyle="--", linewidth=1)
                axis.set_title(FIELD_NAMES[field])
                axis.set_xlabel("음수를 금지한 조건에서 음수를 허용한 조건을 뺀 차이")
                axis.grid(axis="x", alpha=.2)
            fig.suptitle("같은 범주를 짝지어 비교한 부호 제약의 차이와 95% 구간", fontsize=14)
            fig.tight_layout()
            _save_figure(out, "sign_differences.png", fig)
            figures.append("sign_differences.png")
    return figures


def _image(path: Path, alt: str) -> str:
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img class="chart" src="data:image/png;base64,{data}" alt="{html.escape(alt)}">'


def _thumbnail(path: Path) -> str:
    if not path.is_file():
        return '<div class="missing-image">원본 이미지가 로컬에 없어 표시하지 않았다.</div>'
    with Image.open(path) as original:
        picture = original.convert("RGB")
        picture.thumbnail((300, 210))
        stream = io.BytesIO()
        picture.save(stream, format="JPEG", quality=77)
    data = base64.b64encode(stream.getvalue()).decode("ascii")
    return f'<img class="example" src="data:image/jpeg;base64,{data}" alt="해당 좌표 점수가 높은 검증 이미지">'


def _cases(out: Path, records: list[Record]) -> str:
    lookup = {r["key"]: r for r in records}
    korean = {"dog": "개", "skis": "스키", "snow": "눈", "grass": "풀", "road": "도로", "sand": "모래"}
    cards = []
    for key in ("cca_8", "sparse_cca_8"):
        path = out / "cases" / (key + ".json")
        if key not in lookup or not path.exists():
            continue
        for case in _read(path):
            if case["source"] != "image" or case["name"] not in korean:
                continue
            metric = next(r for r in lookup[key]["semantic"]
                          if r["id"] == case["id"] and r["direction"] == "image_to_text")
            parts = [f'<article class="case"><h4>{korean[case["name"]]} · {html.escape(_label(lookup[key]))}</h4>',
                     f'<p>이미지 기준으로 고른 좌표 {case["coordinate"]}번에 부호 {case["sign"]:+d}을 적용했다. '
                     f'이미지 AUROC는 {_number(metric["source_auc"])}이고 문장 AUROC는 '
                     f'{_number(metric["target_transfer_auc"])}이다.</p>']
            parts.append('<div class="example-grid">')
            for example in case["image_examples"][:2]:
                parts.append('<figure>' + _thumbnail(Path(example["image_path"])) +
                             f'<figcaption>이미지 {example["image_id"]}번의 기존 존재 표지는 '
                             f'{"존재" if example["positive"] else "부재"}이다.</figcaption></figure>')
            parts.append('</div><ul>')
            for example in case["text_examples"][:2]:
                parts.append(f'<li><q>{html.escape(example["text"])}</q> '
                             f'사전으로 만든 정답값은 {"언급" if example["positive"] else "미언급"}이다.</li>')
            parts.append('</ul>')
            for side, name in (("image", "이미지"), ("text", "문장")):
                weights = case[side + "_weights"]
                text = ", ".join(f'특징 {w["feature"]}번의 계수 {w["weight"]:+.3f}' for w in weights)
                parts.append(f'<p class="weights">{name} 입력 특징은 총 {case[side + "_feature_count"]}개이다. '
                             f'{html.escape(text)}.</p>')
            cards.append("".join(parts) + '</article>')
    if not cards:
        return '<p>사례를 표시할 두 조건의 계산 결과가 아직 모두 저장되지 않았다.</p>'
    return ('<details><summary>개·스키·눈·풀·도로·모래의 높은 점수 사례를 확인한다</summary>'
            '<p>사례는 각 좌표의 검증 점수가 높은 순서로 고른 이미지와 문장이다. 이미지와 문장은 '
            '서로 정답 쌍일 필요가 없다. 표시한 정답값은 기존 자동 표지이며 사람이 다시 판정한 사실값이 아니다. '
            '오탐 여부와 다의성은 이 예시만으로 확정하지 않았다.</p><div class="case-grid">'
            + "".join(cards) + '</div></details>')


def _coordinate_reuse(records: list[Record]) -> str:
    lookup = {r["key"]: r for r in records}
    paragraphs = []
    sparse = lookup.get("sparse_cca_8")
    if sparse is not None:
        found = {r["name"]: r for r in sparse["semantic"]
                 if r["direction"] == "image_to_text" and r["name"] in ("skis", "snow")}
        if len(found) == 2:
            skis, snow = found["skis"], found["snow"]
            if skis["coordinate"] is not None and (skis["coordinate"], skis["sign"]) == (snow["coordinate"], snow["sign"]):
                paragraphs.append(f'<p class="note"><strong>스키와 눈의 높은 AUROC는 서로 다른 좌표를 뜻하지 않았다.</strong> '
                                  f'입력을 최대 8개로 제한하며 학습한 CCA에서 이미지 기준으로 스키와 눈에 고른 좌표는 '
                                  f'모두 {skis["coordinate"]}번이었고 부호도 모두 {skis["sign"]:+d}이었다. '
                                  f'스키의 이미지·문장 AUROC는 각각 {_number(skis["source_auc"])}와 '
                                  f'{_number(skis["target_transfer_auc"])}였고, 눈은 각각 {_number(snow["source_auc"])}와 '
                                  f'{_number(snow["target_transfer_auc"])}였다. 이 좌표가 눈이 있는 스키 장면처럼 '
                                  '함께 나타나는 내용을 구분했을 가능성이 있다. 두 개념의 의미가 각각 독립된 좌표에 '
                                  '대응했다고 결론 내릴 수는 없다.</p>')
    parts = ['<details><summary>여러 범주에서 같은 좌표와 부호를 고른 횟수를 확인한다</summary>'
             '<p>사물과 배경의 모든 범주를 합쳐 좌표 번호와 부호가 모두 같은 선택을 한 묶음으로 셌다. '
             '공유 범주는 다른 범주 하나 이상과 같은 묶음에 속한 범주를 뜻한다. 좌표를 고를 수 없는 범주는 '
             '선택된 범주 수에서 제외하고 전체 범주 수에는 남겼다. 이 수치만으로 의미가 중복되거나 '
             '독립되어 있다고 판정하지 않는다.</p><div class="table-scroll"><table><thead><tr>'
             '<th>비교 조건</th><th>좌표 선택 방향</th><th>선택된 범주와 전체 범주</th>'
             '<th>서로 다른 좌표·부호 묶음</th><th>다른 범주와 좌표·부호가 같은 범주</th></tr></thead><tbody>']
    for key in ("cca_full", "cca_8", "cca_pruned_renormalized_8", "sparse_cca_8", "sparse_cca_16"):
        if key not in lookup:
            continue
        record = lookup[key]
        for direction in DIRECTIONS:
            rows = [r for r in record["semantic"] if r["direction"] == direction]
            selected = [r for r in rows if r["coordinate"] is not None]
            counts: dict[tuple[int, int], int] = {}
            for row in selected:
                pair = (row["coordinate"], row["sign"])
                counts[pair] = counts.get(pair, 0) + 1
            shared = sum(n for n in counts.values() if n > 1)
            parts.append(f'<tr><th>{html.escape(_label(record))}<br>공통 좌표 공간</th>'
                         f'<td>{DIRECTION_NAMES[direction]}</td><td>{len(selected)}/{len(rows)}개 범주</td>'
                         f'<td>{len(counts)}개 묶음</td><td>{shared}/{len(selected)}개 범주</td></tr>')
    return "".join(paragraphs + parts) + '</tbody></table></div></details>'


def _baseline_lead(lookup: dict[str, Record]) -> str:
    if "cca_8" not in lookup or "cca_full" not in lookup:
        return "CCA 전체 계수와 계수 제거 조건의 결과가 함께 저장되면 같은 공간에서의 차이를 계산한다."
    full, pruned = lookup["cca_full"], lookup["cca_8"]
    parts = []
    for direction in DIRECTIONS:
        original = _summary(full, direction, "object", "target_transfer_auc")[0]
        current = _summary(pruned, direction, "object", "target_transfer_auc")[0]
        parts.append(f'{RETRIEVAL_NAMES[direction]}의 상위 10개 성공률은 {_recall(full, direction, 10):.2f}%에서 '
                     f'{_recall(pruned, direction, 10):.2f}%로 바뀌었고, 사물 범주의 대상 쪽 평균 AUROC는 '
                     f'{_number(original)}에서 {_number(current)}로 바뀌었다.')
    return '공통 좌표마다 이미지·문장 입력을 각각 최대 8개로 줄인 CCA를 전체 계수 조건과 비교했다. ' + " ".join(parts)


def _sign_lead(lookup: dict[str, Record], differences: list[Record]) -> str:
    key = "procrustes_support_8_nonnegative__image"
    reference = "procrustes_support_8_signed__image"
    if key not in lookup or reference not in lookup:
        return "같은 허용 연결과 회귀 목적함수를 쓴 두 부호 조건의 결과가 함께 저장되면 차이를 계산한다."
    record, original = lookup[key], lookup[reference]
    parts = ['연결 상한을 특징당 8개로 고정한 회귀를 이미지 특징 공간에서 비교했다.']
    for direction in DIRECTIONS:
        row = next(r for r in differences if r["left_key"] == key and r["direction"] == direction
                   and r["group"] == "object" and r["metric"] == "target_transfer_auc")
        parts.append(f'음수를 허용할 때와 금지할 때 {RETRIEVAL_NAMES[direction]}의 상위 10개 성공률은 각각 '
                     f'{_recall(original, direction, 10):.2f}%와 {_recall(record, direction, 10):.2f}%였다. '
                     f'사물 범주의 대상 쪽 평균 AUROC 차이는 {_number(row["difference"], 4)}였고, '
                     f'범주를 재추출한 95% 구간은 {_number(row["ci_low"], 4)}에서 {_number(row["ci_high"], 4)}였다.')
    return " ".join(parts)


def _sparse_lead(lookup: dict[str, Record]) -> str:
    keys = ("cca_8", "cca_pruned_renormalized_8", "sparse_cca_8")
    if not all(k in lookup for k in keys):
        return "특징 수를 제한하며 학습한 CCA와 두 사후 제거 기준의 결과가 함께 저장되면 차이를 계산한다."
    original, normalized, trained = [lookup[k] for k in keys]
    improved = True
    for direction in DIRECTIONS:
        learned_mean = _summary(trained, direction, "object", "target_transfer_auc")[0]
        original_mean = _summary(original, direction, "object", "target_transfer_auc")[0]
        improved &= (learned_mean is not None and original_mean is not None and learned_mean > original_mean
                     and _recall(trained, direction, 10) > _recall(original, direction, 10))
    parts = [('공통 좌표당 입력을 최대 8개로 제한하며 학습한 CCA는 사후 계수 제거보다 양방향 검색과 '
              '사물 범주의 대상 쪽 평균 AUROC가 모두 높았다.') if improved else
             '공통 좌표당 입력을 최대 8개로 제한하며 학습한 CCA와 사후 계수 제거의 차이는 측정값별로 달랐다.']
    parts.append('아래 수치는 계수 사후 제거·좌표 크기 재정규화·학습 중 제한의 순서로 제시했다.')
    for direction in DIRECTIONS:
        recalls = '·'.join(f'{_recall(r, direction, 10):.2f}%' for r in (original, normalized, trained))
        semantic = '·'.join(_number(_summary(r, direction, "object", "target_transfer_auc")[0])
                            for r in (original, normalized, trained))
        parts.append(f'{RETRIEVAL_NAMES[direction]}의 상위 10개 성공률은 세 조건의 순서대로 {recalls}였고, '
                     f'사물 범주의 대상 쪽 평균 AUROC는 {semantic}였다.')
    if "cca_full" in lookup:
        full = lookup["cca_full"]
        parts.append(f'전체 계수를 유지한 CCA의 상위 10개 검색 성공률은 이미지로 문장을 검색할 때 '
                     f'{_recall(full, "image_to_text", 10):.2f}%, 문장으로 이미지를 검색할 때 '
                     f'{_recall(full, "text_to_image", 10):.2f}%였다. 문장에서 좌표를 골라 이미지에 적용한 '
                     f'사물 평균 AUROC는 {_number(_summary(full, "text_to_image", "object", "target_transfer_auc")[0])}였다.')
    return " ".join(parts)


def _difference_details(differences: list[Record], comparison: str) -> str:
    records = [r for r in differences if r["comparison"] == comparison]
    if not records:
        return ""
    parts = ['<details><summary>범주를 짝지은 AUROC 차이와 95% 구간을 모두 확인한다</summary>'
             '<p>차이는 비교 조건에서 기준 조건을 뺀 값이다. 같은 범주를 두 조건에서 함께 재추출하는 계산을 '
             '2,000번 반복했다. 이 구간은 이미지나 문장을 재추출한 구간이 아니며, 주어진 범주 집합에서 '
             '차이가 얼마나 달라지는지를 나타낸다.</p><div class="table-scroll"><table><thead><tr>'
             '<th>비교 조건과 기준 조건</th><th>비교 공간</th><th>좌표 선택 방향과 범주</th><th>측정값</th>'
             '<th>평균 차이</th><th>범주 재추출 95% 구간</th><th>함께 계산 가능한 범주</th></tr></thead><tbody>']
    for row in records:
        parts.append(f'<tr><th>{html.escape(row["left_condition"])}에서<br>{html.escape(row["right_condition"])}을 뺀다</th>'
                     f'<td>{SPACE_NAMES[row["space"]]}</td><td>{DIRECTION_NAMES[row["direction"]]}<br>'
                     f'{GROUP_NAMES[row["group"]]} 범주</td><td>{FIELD_NAMES[row["metric"]]}</td>'
                     f'<td>{_number(row["difference"], 4)}</td><td>{_number(row["ci_low"], 4)}에서 '
                     f'{_number(row["ci_high"], 4)}</td><td>{row["valid_concepts"]}/{row["total_concepts"]}개</td></tr>')
    return "".join(parts) + '</tbody></table></div></details>'


def _limitations(records: list[Record], protocol: Record) -> str:
    if not records:
        return ""
    base = next((r for r in records if r["key"] == "hungarian__image"), records[0])
    parts = []
    for group in GROUPS:
        for direction in DIRECTIONS:
            _, valid, total = _summary(base, direction, group, "target_transfer_auc")
            parts.append(f'{DIRECTION_NAMES[direction]} 조건의 {GROUP_NAMES[group]} 범주는 '
                         f'전체 {total}개 중 {valid}개에서 대상 AUROC를 계산할 수 있었다.')
    return (" ".join(parts) + ' 사물과 배경의 평균에는 계산 가능한 범주만 같은 비중으로 반영했다. '
            '계산할 수 없는 범주는 개별 기록과 전체 범주 수에 남겼다. 문장 표지는 사람이 모든 문장을 읽고 '
            '만든 정답이 아니라 기존 사전으로 찾은 언급이다. 특히 배경 범주에서는 미언급과 시각적 부재가 '
            '같지 않다. 따라서 여기서 측정한 값은 고정 좌표가 자동 개념 표지를 얼마나 구분하는지이며, '
            '그 좌표의 의미가 하나뿐인지 또는 사람이 쉽게 해석할 수 있는지를 증명하지 않는다.')


def report(out: Path) -> Path:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out / "protocol.json"
    protocol = _read(protocol_path) if protocol_path.exists() else {}
    records = [_read(p) for p in sorted((out / "results").glob("*.json"))]
    if records:
        _validate(records, protocol)
    lookup = {r["key"]: r for r in records}
    _export(out, records)
    differences = _differences(records)
    _csv(out / "paired_differences.csv", differences)
    figures = _figures(out, records, differences)
    sections = {kind: [r for r in records if r["section"] == kind] for kind in ("baseline", "sign", "sparse")}
    # Both original post-pruned CCA and its rescaled control are explicit sparse-training references.
    sparse_k = {(r.get("options") or {}).get("k") for r in sections["sparse"]}
    sections["sparse"] += [r for r in sections["baseline"] if r["family"] == "cca"
                           and (r.get("options") or {}).get("k") in sparse_k]
    manifest_path = out / "manifest.json"
    cfg = _read(manifest_path).get("config", {}) if manifest_path.exists() else {}
    expected = (2 + 5 * (1 + len(cfg.get("baseline_k", [4, 8, 16])))
                + 4 * len(cfg.get("sign_k", [8, 16])) + 2 * len(cfg.get("sparse_k", [4, 8, 16])))
    status = (f'예정한 {expected}개 조건 중 {len(records)}개 결과를 반영했다. '
              + ('아직 계산 중인 조건이 있으므로 이 보고서는 중간 결과이다.' if len(records) < expected else
                 '예정한 조건의 계산 결과를 모두 반영했다.'))
    css = """
    *{box-sizing:border-box}body{margin:0;background:#f2f4f5;color:#203442;
    font:15px/1.75 -apple-system,BlinkMacSystemFont,'Apple SD Gothic Neo','Malgun Gothic',sans-serif}
    main{max-width:1510px;margin:auto;padding:36px 30px 70px}h1{font-size:32px;line-height:1.35;letter-spacing:-.8px}
    h2{font-size:24px;line-height:1.4;margin:0 0 18px}h3{font-size:18px}p{max-width:1150px;margin:12px 0}
    section{background:white;border:1px solid #d9e0e5;border-radius:8px;padding:26px;margin:24px 0}
    .status,.note{background:#eaf0f3;padding:14px 18px;border-radius:5px}.lead{font-size:17px;line-height:1.9}
    nav{display:grid;gap:8px;margin:24px 0}nav a{color:#245a79;text-decoration:none}small{display:block;font-size:11px;color:#59707f}
    .table-scroll{overflow-x:auto;margin-top:20px}table{border-collapse:collapse;width:100%;font-size:12px;min-width:1120px}
    th,td{border-bottom:1px solid #e2e6e9;padding:11px 10px;text-align:left;vertical-align:top;min-width:100px}
    thead th{background:#233e51;color:white;font-size:12px}tbody th{font-weight:600;min-width:180px}
    .family th{background:#e8eef2;color:#224c66;padding:8px 12px;font-size:13px}td{font-variant-numeric:tabular-nums}
    details{margin:20px 0;border-top:1px solid #d9e0e5;padding-top:14px}summary{cursor:pointer;font-weight:650;color:#245a79}
    .chart{display:block;width:100%;height:auto;margin:24px 0 6px}.caption{font-size:13px;color:#526c7c}
    .case-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.case{padding:18px;border:1px solid #d8e0e6;border-radius:6px}
    .case h4{margin:0;font-size:17px}.case p,.case li{font-size:13px}.example-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
    figure{margin:0}.example{width:100%;height:165px;object-fit:contain;background:#f1f3f5}figcaption{font-size:11px}
    .missing-image{padding:24px 12px;background:#f1f3f5;color:#667d8a;font-size:12px;min-height:100px}
    .weights{color:#59717e}.pending{color:#865b31}code{font-size:12px;overflow-wrap:anywhere}a{color:#24669b}
    @media(max-width:760px){main{padding:20px 12px}section{padding:18px}.case-grid{grid-template-columns:1fr}h1{font-size:26px}}
    """
    body = ['<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" '
            'content="width=device-width,initial-scale=1"><title>대응 방법의 검색 성능과 개념 구분 성능</title>'
            f'<style>{css}</style></head><body><main><h1>대응 방법의 검색 성능과 개념 구분 성능</h1>'
            f'<p class="status">{status}</p>',
            '<p>검색 성공률과 개념 구분 성능을 같은 변환 공간에서 함께 비교했다. 개념 구분 성능은 학습과 분리한 '
            '자료에서 고른 좌표 하나를 이미지와 문장에 그대로 적용해 측정했다.</p>',
            '<nav><a href="#baseline">1. 계수를 줄인 대응 방법을 비교한다. 검색 성공률과 개념 구분 성능을 함께 확인한다.</a>'
            '<a href="#sign">2. 같은 연결에서 음수 계수를 허용할지 비교한다. 부호의 허용 조건만 바꾼 결과를 확인한다.</a>'
            '<a href="#sparse">3. 특징 수를 제한하며 학습한 CCA를 비교한다. 사후 계수 제거와 크기 재정규화를 함께 확인한다.</a></nav>',
            '<section><h2>측정값을 읽는 방법</h2>',
            '<p>희소 자기부호화기는 원래 표현을 드문드문 활성화되는 특징들로 바꾸는 모델이다. 이 보고서의 입력 특징은 '
            '이미 학습한 희소 자기부호화기의 값을 사용했다. 좌표는 대응 방법이 만든 출력값 하나이며 여러 입력 특징의 '
            '가중합일 수 있다. 각 개념에서는 출발 쪽 자료만 사용해 좌표 하나와 부호 하나를 골랐다.</p>',
            '<p>AUROC는 개념이 있는 예시의 점수가 없는 예시보다 높을 확률이며, 같은 점수에는 절반의 점수를 준다. '
            '0.5는 구분하지 못하는 수준이고 1은 모든 양성 예시의 점수가 모든 음성 예시보다 높다는 뜻이다. '
            '표의 첫 AUROC는 좌표를 고른 쪽을, 두 번째는 같은 좌표와 같은 부호를 전달한 대상 쪽을 평가한다. '
            '마지막 값은 각 범주에서 두 AUROC 중 작은 값을 먼저 구한 뒤 범주별로 평균했다. 두 평균 중 작은 값을 '
            '계산한 결과와는 다르다.</p>',
            '<p>상위 10개 검색 성공률은 검색 결과 10개 안에 정답이 하나 이상 있는 질문의 비율이다. '
            'CCA는 이미지와 문장의 좌표가 함께 변하도록 두 선형 변환을 학습한다. Procrustes는 두 표현의 거리를 '
            '줄이는 직교 변환에서 대응 계수를 얻는다. Sinkhorn은 전체 대응량을 분배하는 양수 가중치를 계산한다. '
            'Sinkhorn의 연결 분산 벌점은 가중치가 여러 연결에 퍼지도록 하는 정도이며 여기서는 0.05로 고정했다. '
            '헝가리안 방법은 특징을 일대일로 연결한다. 공통 좌표 공간, 이미지 특징 공간, 문장 특징 공간을 구분해 표시했다.</p>',
            f'<p>대응 변환 학습에는 이미지 {protocol.get("fit_images", 0):,}개를 사용했고, 좌표와 부호를 고르는 데에는 '
            f'다른 이미지 {protocol.get("calibration_images", 0):,}개와 그 문장을 사용했다. 검증에는 이미지 '
            f'{protocol.get("evaluation_images", 0):,}개와 문장 {protocol.get("evaluation_captions", 0):,}개를 사용했다. '
            '검증 자료는 이전 탐색 분석에서도 사용한 COCO val2017이며, 이번에 처음 확인한 최종 시험 자료가 아니다. '
            '대상의 개념 표지는 좌표나 부호 선택에 사용하지 않았고 새로운 다중 좌표 분류기도 학습하지 않았다.</p>',
            '<p class="note">표의 평균 옆에는 계산 가능한 범주 수와 전체 범주 수를 표시했다. 상수 좌표는 0.5로 남겼고, '
            '양성이나 음성이 없는 범주는 계산 불가로 남겼다. 이미지 또는 문장 특징을 그대로 복사하는 계수는 학습한 '
            '변환 계수 수에 넣지 않았다. 입력 수와 음수 계수 수는 실제 저장된 계수에서 계산했다.</p></section>',
            '<section id="baseline"><h2>1. 계수를 줄였을 때 검색 성능과 개념 구분 성능은 어떻게 달라졌는가</h2>',
            f'<p class="lead">{_baseline_lead(lookup)}</p>', _table(sections["baseline"]),
            _retrieval_details(sections["baseline"])]
    if "retrieval_semantics.png" in figures:
        body.extend([_image(out / "retrieval_semantics.png", "검색 성공률과 대상 AUROC 및 양쪽 최솟값의 관계"),
                     '<p class="caption">점 하나는 방법·공간·좌표 선택 방향 하나를 나타낸다. 색은 대응 방법을, 모양은 '
                     '좌표를 고른 방향을 나타낸다. 서로 다른 공간의 점을 함께 표시했으므로 두 축의 관계를 단일 원인으로 '
                     '해석하지 않았다. 세부 조건의 수치는 표에 제시했다.</p>'])
    body.extend(['</section><section id="sign"><h2>2. 같은 연결에서 음수 계수를 금지하면 무엇이 달라졌는가</h2>',
                 f'<p class="lead">{_sign_lead(lookup, differences)}</p>',
                 '<p>회귀는 한쪽 특징의 값을 다른 쪽 특징의 가중합으로 예측하는 계산이다. 두 조건은 Procrustes에서 '
                 '남긴 같은 허용 연결에 대해, 예측오차의 제곱과 계수 크기 벌점의 합을 줄이도록 계수를 학습했다. '
                 '한 조건은 계수의 양수와 음수를 모두 허용했고 다른 조건은 음수를 금지했다. 따라서 이 비교는 '
                 '원래 Procrustes와 다른 회귀 모델 사이의 비교가 아니라 같은 회귀에서 부호 제약을 바꾼 비교이다. '
                 '허용한 연결 수가 같아도 음수를 금지한 조건에서 실제 계수가 0이 될 수 있다.</p>',
                 _table(sections["sign"]), _retrieval_details(sections["sign"])])
    if "sign_differences.png" in figures:
        body.extend([_image(out / "sign_differences.png", "같은 연결에서 음수 허용과 금지의 범주별 짝 비교"),
                     '<p class="caption">오른쪽 값은 음수를 금지한 조건이 높다는 뜻이다. 구간은 같은 범주를 두 조건에서 '
                     '함께 재추출한 95% 구간이며 이미지 단위의 불확실성을 나타내지 않는다. 세로축의 분수는 함께 계산 '
                     '가능한 범주 수와 전체 범주 수이다.</p>'])
    body.extend([_difference_details(differences, "sign"),
                 '</section><section id="sparse"><h2>3. 처음부터 특징 수를 제한해 학습하면 사후 제거와 달라지는가</h2>',
                 f'<p class="lead">{_sparse_lead(lookup)}</p>',
                 '<p>특징 수를 제한한 CCA는 각 좌표의 원래 입력 특징을 최대 4개·8개·16개로 제한하며 계수를 다시 '
                 '학습했다. 사후 제거 기준은 학습한 CCA에서 절댓값이 큰 계수만 남겼다. 별도의 재정규화 기준은 '
                 '남긴 좌표의 분산과 계수 제곱합의 0.01배를 더한 값이 1이 되도록 좌표 크기를 조정했다. '
                 '분산은 학습 자료의 좌표 점수가 평균 주변에 얼마나 퍼져 있는지를 나타낸다. '
                 '양수 배율 조정은 단일 좌표 AUROC를 바꾸지 않지만 전체 좌표를 사용하는 검색 결과에는 영향을 줄 수 있다.</p>',
                 '<p>특징 수를 제한한 학습은 256개 좌표를 순서대로 구한다. 이미지와 문장 특징들이 함께 변하는 '
                 '정도인 교차 공분산에서 이전 좌표가 설명한 부분을 뺀 뒤 다음 좌표를 구한다. '
                 '선택한 특징의 계수를 번갈아 최적화하고 일부 특징 교체를 확인한다. 전체 경우를 탐색한 최적해나 '
                 '서로 다른 좌표들이 함께 변하지 않는 성질은 보장하지 않는다. 이미지와 문장의 좌표 쌍이 '
                 '원래 학습 자료에서 음의 상관을 갖지 않도록 문장 계수 전체의 부호를 정했다. 이 과정은 개념 표지를 '
                 '사용하지 않았다. 이후 개념마다 좌표와 부호를 고를 때는 앞서 설명한 출발 쪽 자료만 사용했다. '
                 '사후 제거와 비교하면 선택한 특징·계수·앞선 좌표를 '
                 '제거하는 계산이 함께 달라지므로, 성능 차이를 특징 수 제한 하나의 효과로 해석할 수 없다.</p>',
                 _table(sections["sparse"]), _retrieval_details(sections["sparse"]),
                 _difference_details(differences, "sparse")])
    if "semantic_distributions.png" in figures:
        body.extend([_image(out / "semantic_distributions.png", "대표 조건의 대상 AUROC 범주별 분포"),
                     '<p class="caption">가운데 선은 중앙값이고 상자는 25백분위수와 75백분위수 사이를 나타낸다. '
                     '수염은 상자 길이의 1.5배 안에 있는 관측값까지 표시하며 바깥 점은 그림에서만 생략했다. '
                     '모든 범주의 값과 계산 불가 기록은 개별 CSV에 남겼다.</p>'])
    body.extend([_coordinate_reuse(records), _cases(out, records), '</section><section><h2>이 결과에서 확인한 범위</h2>',
                 f'<p>{_limitations(records, protocol)}</p>',
                 '<p>이 분석은 개념을 가리거나 이미지를 수정하는 실험을 하지 않았다. 저장된 이미지 존재 표지와 '
                 '문장 언급 표지를 사용한 관측적 평가이므로 특정 좌표가 그 개념에만 반응한다거나 그 개념이 '
                 '출력의 원인이라고 결론 내리지 않는다.</p>',
                 '<details><summary>계산 결과 파일과 저장 위치를 확인한다</summary>',
                 f'<p>전체 결과 폴더는 <code>{html.escape(str(out))}</code>이다. 이 HTML은 그래프와 불러올 수 있는 '
                 '예시 이미지를 내부에 포함하므로 단독으로 열 수 있다.</p>'])
    for name, description in (("summary.csv", "조건·공간·방향·범주군별 평균과 전체 분모를 저장했다"),
                              ("semantic_per_concept.csv", "모든 범주의 좌표·부호·AUROC·양성 및 음성 수를 저장했다"),
                              ("paired_differences.csv", "같은 범주를 짝지은 차이와 범주 재추출 95% 구간을 저장했다")):
        body.append(f'<p><a href="{html.escape(str(out / name))}">{html.escape(str(out / name))}</a>에 {description}.</p>')
    body.append('</details></section></main></body></html>')
    destination = out / "report.html"
    destination.write_text("".join(body), encoding="utf-8")
    print(json.dumps({"report": str(destination), "completed_conditions": len(records),
                      "expected_conditions": expected, "figures": figures}), flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    report(args.run)


if __name__ == "__main__":
    main()
