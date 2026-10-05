"""Render the annotation-guided concept-set retrieval experiment."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


Record = dict[str, Any]
DEFAULT_RUN = Path(__file__).resolve().parents[2] / "runs/oracle-sets-2026-10-04"
DIRECTIONS = ("image_to_text", "text_to_image")
DIRECTION_NAMES = {"image_to_text": "이미지로 문장 검색", "text_to_image": "문장으로 이미지 검색"}
TARGET_NAMES = {
    "propagated_presence": "이미지의 범주 주석을 이미지와 문장에 공통으로 사용",
    "caption_mentions": "이미지는 범주 주석, 문장은 해당 개념의 언급을 사용",
    "unsupervised": "개념 주석 없이 학습한 대응 방법",
}
SHORT_TARGET_NAMES = {
    "propagated_presence": "이미지 주석을 양쪽에 사용",
    "caption_mentions": "문장에는 사전 기반 언급을 사용",
}
COLORS = {"propagated_presence": "#285f9b", "caption_mentions": "#c57947"}
GROUP_NAMES = {"object": "물체", "background": "배경"}
BUDGETS = (1, 4, 8, 16, "full")
DIAGNOSTIC_NAMES = {
    "prediction_unit_variance": "기존 조건처럼 학습 예측값의 표준편차로 나눔",
    "centered_prediction": "학습 예측값에서 평균만 빼고 표준편차로 나누지 않음",
    "label_unit_variance": "학습 예측값을 학습 정답 표지의 표준편차로 나눔",
    "objects_prediction_unit_variance": "기존 크기 보정을 유지하고 물체 80개 좌표만 사용",
    "background_prediction_unit_variance": "기존 크기 보정을 유지하고 배경 91개 좌표만 사용",
    "true_image_labels_predicted_text": "이미지에는 평가 정답 주석, 문장에는 학습한 예측값을 사용",
    "predicted_image_true_text_labels": "이미지에는 학습한 예측값, 문장에는 연결된 이미지의 평가 정답 주석을 사용",
}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> bool:
    return value is not None and np.isfinite(value)


def _number(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}" if _finite(value) else "계산 불가"


def _budget(value: Any) -> str:
    return "전체 특징" if value == "full" else f"개념당 최대 {value}개"


def _recall(record: Record, direction: str, k: int) -> float:
    return 100.0 * float(record["retrieval"][direction]["recall"][str(k)])


def _order(record: Record) -> tuple[int, int, str]:
    target_order = {name: i for i, name in enumerate(TARGET_NAMES)}
    budget = record.get("k", "full")
    budget_index = BUDGETS.index(budget) if budget in BUDGETS else len(BUDGETS)
    return target_order.get(record["label_target"], 3), budget_index, record["key"]


def _mean_auc(record: Record, field: str, group: str | None = None) -> tuple[float | None, int, int]:
    rows = [r for r in record.get("semantic", []) if group is None or r["group"] == group]
    values = [float(r[field]) for r in rows if _finite(r.get(field))]
    return (float(np.mean(values)) if values else None), len(values), len(rows)


def _cca_control(records: list[Record]) -> Record | None:
    controls = [r for r in records if r["label_target"] == "unsupervised" and r["dimensions"] == 171
                and "sparse" not in r["key"]]
    return next((r for r in controls if r.get("output_scaling") == "fit_unit_variance"),
                controls[0] if controls else None)


def _validate(records: list[Record], protocol: Record) -> None:
    keys = [r["key"] for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Result keys must be unique")
    for record in records:
        for direction in DIRECTIONS:
            metric = record["retrieval"][direction]
            expected = (protocol["evaluation_images"], protocol["evaluation_captions"])
            if direction == "text_to_image":
                expected = expected[::-1]
            if (metric["query_count"], metric["candidate_count"]) != expected:
                raise ValueError(f"Retrieval denominator differs for {record['key']}")
            if "ranks" in metric:
                ranks = np.asarray(metric["ranks"])
                if len(ranks) != metric["query_count"]:
                    raise ValueError("Ranks and query count differ")
                for k in (1, 5, 10):
                    np.testing.assert_allclose(
                        metric["recall"][str(k)],
                        np.mean(ranks <= min(k, metric["candidate_count"])),
                    )
        if record["label_target"] == "unsupervised":
            continue
        rows = record.get("semantic", [])
        if len(rows) != record["dimensions"]:
            raise ValueError("Each annotated concept needs a semantic evaluation row")
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Repeated concept IDs")
        for row in rows:
            for field in ("image_auc", "text_auc", "text_presence_auc", "text_mention_auc"):
                if _finite(row.get(field)) and not 0 <= row[field] <= 1:
                    raise ValueError(f"AUROC outside zero to one for {record['key']}")


def _csv(path: Path, rows: list[Record]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _export(out: Path, records: list[Record]) -> None:
    summaries, semantics = [], []
    for record in records:
        metadata = {k: record.get(k) for k in
                    ("key", "label", "label_target", "k", "dimensions", "output_scaling")}
        summary = dict(metadata)
        for direction in DIRECTIONS:
            for rank in (1, 5, 10):
                summary[f"{direction}_recall_at_{rank}_percent"] = _recall(record, direction, rank)
        for field in ("image_auc", "text_auc", "text_presence_auc", "text_mention_auc"):
            for group in GROUP_NAMES:
                mean, count, total = _mean_auc(record, field, group)
                summary[f"{group}_{field}"] = mean
                summary[f"{group}_{field}_valid"] = count
                summary[f"{group}_{field}_total"] = total
        summaries.append(summary)
        semantics.extend({**metadata, **row} for row in record.get("semantic", []))
    _csv(out / "summary.csv", summaries)
    _csv(out / "semantic_per_concept.csv", semantics)


def _fonts() -> None:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in ("AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR"):
        if candidate in installed:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})


def _retrieval_plot(out: Path, records: list[Record]) -> str | None:
    supervised = [r for r in records if r["label_target"] != "unsupervised"]
    if not supervised:
        return None
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=True)
    for axis, direction in zip(axes, DIRECTIONS, strict=True):
        for target, color in COLORS.items():
            selected = {r["k"]: r for r in supervised if r["label_target"] == target}
            x = [i for i, budget in enumerate(BUDGETS) if budget in selected]
            y = [_recall(selected[BUDGETS[i]], direction, 10) for i in x]
            axis.plot(x, y, "o-", color=color, lw=2, label=SHORT_TARGET_NAMES[target])
        control = _cca_control(records)
        if control is not None:
            axis.axhline(_recall(control, direction, 10), color="#7b8490", ls="--", lw=1.5,
                         label="주석 없는 CCA 171좌표 · 학습 분산 보정")
        axis.set_title(DIRECTION_NAMES[direction], pad=14)
        axis.set_xticks(range(len(BUDGETS)), ["1개", "4개", "8개", "16개", "전체"])
        axis.set_xlabel("개념 하나를 예측하는 데 허용한 SAE 특징 수")
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=.16)
    axes[0].set_ylabel("상위 10개 검색 성공률 (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=3, frameon=False, fontsize=10)
    figure.suptitle("범주 주석으로 만든 개념 집합의 검색 성능", fontsize=16, y=.99)
    figure.tight_layout(rect=(0, .1, 1, .94))
    name = "retrieval_by_budget.png"
    figure.savefig(out / name, dpi=180)
    plt.close(figure)
    return name


def _semantic_plot(out: Path, records: list[Record]) -> str | None:
    supervised = sorted([r for r in records if r["label_target"] != "unsupervised"], key=_order)
    if not supervised:
        return None
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    fields = (("image_auc", "이미지에서 범주가 있는지 구별"),
              ("text_mention_auc", "문장에서 범주를 언급하는지 구별"))
    labels = [f'{"이미지 주석 공통" if r["label_target"] == "propagated_presence" else "문장 언급 사용"}\n'
              f'{"전체 특징" if r["k"] == "full" else str(r["k"]) + "개 특징"}' for r in supervised]
    for row_index, group in enumerate(GROUP_NAMES):
        for col_index, (field, name) in enumerate(fields):
            axis = axes[row_index, col_index]
            arrays = [[row[field] for row in record.get("semantic", [])
                       if row["group"] == group and _finite(row.get(field))] for record in supervised]
            boxes = axis.boxplot([values if values else [np.nan] for values in arrays],
                                 patch_artist=True, showfliers=False, widths=.62,
                                 medianprops={"color": "#233544", "linewidth": 1.5})
            for box, record in zip(boxes["boxes"], supervised, strict=True):
                box.set_facecolor(COLORS[record["label_target"]])
                box.set_alpha(.35)
            axis.axhline(.5, color="#8e99a2", lw=1, ls="--")
            axis.set_xticks(np.arange(1, len(labels) + 1), labels, rotation=40, ha="right", fontsize=8)
            axis.set_title(f"{GROUP_NAMES[group]} 범주 · {name}", pad=12)
            axis.set_ylim(0, 1.02)
            axis.grid(axis="y", alpha=.15)
            if col_index == 0:
                axis.set_ylabel("개념 있음·없음 AUROC")
    figure.suptitle("두 학습 조건을 동일한 개념 표지로 평가한 범주별 분포", fontsize=16, y=.99)
    figure.tight_layout(rect=(0, 0, 1, .96), h_pad=3)
    name = "concept_auc_distributions.png"
    figure.savefig(out / name, dpi=180)
    plt.close(figure)
    return name


def _image(out: Path, name: str, alt: str) -> str:
    payload = base64.b64encode((out / name).read_bytes()).decode("ascii")
    return f'<img class="chart" src="data:image/png;base64,{payload}" alt="{html.escape(alt)}">'


def _retrieval_table(records: list[Record]) -> str:
    if not records:
        return "<p>이 조건의 계산 결과는 아직 저장되지 않았다.</p>"
    maxima = {(direction, rank): max(_recall(r, direction, rank) for r in records)
              for direction in DIRECTIONS for rank in (1, 5, 10)}
    parts = ['<div class="table-scroll"><table><thead><tr><th rowspan="2">대응 방법과 학습 표지</th>'
             '<th rowspan="2">출력 좌표 수</th><th rowspan="2">좌표당 SAE 특징 수</th>'
             '<th colspan="3">이미지로 문장 검색</th><th colspan="3">문장으로 이미지 검색</th></tr><tr>']
    parts.extend(f"<th>상위 {k}개 성공률</th>" for _ in DIRECTIONS for k in (1, 5, 10))
    parts.append("</tr></thead><tbody>")
    previous_target = None
    for record in records:
        target = record["label_target"]
        if previous_target != target:
            parts.append(f'<tr class="family"><th colspan="9">{TARGET_NAMES[target]}</th></tr>')
            previous_target = target
        parts.append(f'<tr><th>{html.escape(record["label"])}</th><td>{record["dimensions"]}개</td>'
                     f'<td>{_budget(record.get("k", "full"))}</td>')
        for direction in DIRECTIONS:
            for rank in (1, 5, 10):
                value = _recall(record, direction, rank)
                text = f"{value:.2f}%"
                if value == maxima[direction, rank]:
                    text = f"<strong>{text}</strong>"
                parts.append(f"<td>{text}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _semantic_table(records: list[Record]) -> str:
    parts = ['<div class="table-scroll"><table><thead><tr><th>학습 조건과 특징 수</th><th>평가 범주</th>'
             '<th>이미지의 범주 존재 AUROC</th><th>문장의 범주 언급 AUROC</th>'
             '<th>문장으로 해당 이미지의 범주 존재를 예측한 AUROC</th></tr></thead><tbody>']
    for record in records:
        if record["label_target"] == "unsupervised":
            continue
        for group in GROUP_NAMES:
            parts.append(f'<tr><th>{SHORT_TARGET_NAMES[record["label_target"]]}<small>{_budget(record["k"])}</small>'
                         f'</th><td>{GROUP_NAMES[group]}</td>')
            for field in ("image_auc", "text_mention_auc", "text_presence_auc"):
                value, valid, total = _mean_auc(record, field, group)
                parts.append(f'<td>{_number(value)}<small>계산 가능한 범주 {valid}/{total}개</small></td>')
            parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _direction_values(record: Record) -> str:
    return (f'이미지로 문장 검색 {_recall(record, "image_to_text", 10):.2f}%, '
            f'문장으로 이미지 검색 {_recall(record, "text_to_image", 10):.2f}%')


def _leads(records: list[Record]) -> list[str]:
    supervised = [r for r in records if r["label_target"] != "unsupervised"]
    cca = _cca_control(records)
    statements = []
    prop16 = next((r for r in supervised if r["label_target"] == "propagated_presence" and r["k"] == 16), None)
    mention16 = next((r for r in supervised if r["label_target"] == "caption_mentions" and r["k"] == 16), None)
    sparse16 = next((r for r in records if r["label_target"] == "unsupervised" and r["dimensions"] == 171
                     and r["k"] == 16 and r.get("output_scaling") == "fit_unit_variance"), None)
    if prop16 is not None and sparse16 is not None:
        delta = [_recall(prop16, direction, 10) - _recall(sparse16, direction, 10) for direction in DIRECTIONS]
        conclusion = "두 방향 모두 높았다" if min(delta) > 0 else (
            "두 방향 모두 낮았다" if max(delta) < 0 else "검색 방향에 따라 우열이 달랐다")
        statements.append('<strong>같은 범주를 예측하도록 집합을 학습한 조건은 Sparse CCA보다 '
                          f'{conclusion}.</strong> 양쪽 모두 171좌표와 좌표당 특징 최대 16개를 사용하고 '
                          '학습 예측 분산을 1로 맞췄다. 이미지 주석을 양쪽에 사용한 집합은 '
                          f'{_direction_values(prop16)}, Sparse CCA는 {_direction_values(sparse16)}였다.')
    if cca is not None and supervised:
        best = max(supervised, key=lambda r: sum(_recall(r, direction, 10) for direction in DIRECTIONS))
        differences = [_recall(best, direction, 10) - _recall(cca, direction, 10) for direction in DIRECTIONS]
        conclusion = ("두 검색 방향 모두 높았다" if min(differences) > 0 else
                      "두 검색 방향 모두 낮았다" if max(differences) < 0 else "검색 방향에 따라 우열이 달랐다")
        budget_description = "전체 특징을 사용한" if best["k"] == "full" else f'개념당 최대 {best["k"]}개 특징을 사용한'
        statements.append(f'<strong>주석을 사용한 설정 중 관측한 평균이 가장 높은 조건도 CCA 171좌표보다 '
                          f'{conclusion}.</strong> {budget_description} 주석 기반 조건은 {_direction_values(best)}, '
                          f'CCA 171좌표는 {_direction_values(cca)}였다. 개념 표지를 예측하는 집합과 이미지·문장의 '
                          '공통 정보를 학습한 집합을 비교한 결과이며, 주석 사용 여부만 바꾼 실험은 아니다.')
    if prop16 is not None and mention16 is not None:
        delta = [_recall(prop16, direction, 10) - _recall(mention16, direction, 10) for direction in DIRECTIONS]
        conclusion = "두 방향 모두 높았다" if min(delta) > 0 else (
            "두 방향 모두 낮았다" if max(delta) < 0 else "검색 방향에 따라 우열이 달랐다")
        statements.append('<strong>문장에 해당 이미지의 범주 주석을 부여한 조건은 사전 기반 언급을 사용한 조건보다 '
                          f'{conclusion}.</strong> 개념당 특징 16개에서 이미지 주석을 양쪽에 사용하면 '
                          f'{_direction_values(prop16)}, 문장에는 사전 기반 언급을 사용하면 '
                          f'{_direction_values(mention16)}였다. 문장에 생략된 범주까지 예측하도록 학습한 조건이 '
                          '현재 검색 평가에서 유리했으며, 캡션 사전의 누락 영향과 실제 미언급의 영향은 분리하지 않았다.')
    return statements


def _cases(out: Path, records: list[Record]) -> str:
    selected = [r for r in records if r["label_target"] != "unsupervised" and r.get("k") == 8]
    parts = []
    for record in selected:
        path = out / "cases" / f'{record["key"]}.json'
        if not path.exists():
            continue
        cases = _read(path)
        if isinstance(cases, dict):
            cases = cases.get("cases", [])
        named = {case["name"]: case for case in cases}
        if "skis" not in named or "snow" not in named:
            continue
        parts.append(f'<h3>{TARGET_NAMES[record["label_target"]]} 조건에서 눈과 스키를 예측한 특징</h3>')
        for side, side_name in (("image", "이미지"), ("text", "문장")):
            skis, snow = named["skis"], named["snow"]
            ski_features = {x["feature"] for x in skis[f"{side}_weights"] if x["weight"] != 0}
            snow_features = {x["feature"] for x in snow[f"{side}_weights"] if x["weight"] != 0}
            shared = sorted(ski_features & snow_features)
            parts.append(f'<p>{side_name}에서 스키를 예측하는 집합은 {len(ski_features)}개, 눈을 예측하는 집합은 '
                         f'{len(snow_features)}개 특징을 사용했고, 그중 {len(shared)}개 특징을 공유했다. '
                         '같은 번호를 공유하더라도 가중치가 다르면 두 집합의 출력값은 다를 수 있다.</p>')
            parts.append('<div class="table-scroll"><table class="case-table"><thead><tr><th>개념</th>'
                         '<th>원래 SAE 특징 번호와 학습한 가중치</th></tr></thead><tbody>')
            for name, case in (("스키", skis), ("눈", snow)):
                weights = ", ".join(f'{int(w["feature"])}번 {float(w["weight"]):+.4f}'
                                    for w in case[f"{side}_weights"])
                parts.append(f'<tr><th>{name}</th><td>{weights}</td></tr>')
            parts.append("</tbody></table></div>")
    if not parts:
        return ""
    return ('<section><h2>주석으로 이름을 붙인 집합도 일부 activation을 공유할 수 있다</h2>'
            '<p>아래는 개념마다 특징을 최대 8개 사용한 조건이다. 특징 번호는 변환 안의 행 번호가 아니라 '
            '기존 SAE의 원래 activation 번호이다. 가중치는 학습 자료로 표준화한 activation에 곱하는 값이며 '
            '음수도 허용했다. 서로 다른 개념 표지를 학습했다고 해서 선택된 모든 특징이 그 개념에만 반응한다고 '
            '판단할 수는 없다.</p>' + "".join(parts) + "</section>")


def _conditional_table(records: list[Record]) -> str:
    parts = []
    names = {"snow": "눈", "skis": "스키"}
    for record in records:
        if record["label_target"] == "unsupervised" or record.get("k") != 8:
            continue
        rows = [row for row in record.get("conditional", [])
                if row["name"] in names and row["held_concept"] in names]
        if not rows:
            continue
        parts.append(f'<h3>{TARGET_NAMES[record["label_target"]]} 조건의 개념당 특징 8개 결과</h3>')
        parts.append('<div class="table-scroll"><table><thead><tr><th>평가 자료와 구별할 개념</th>'
                     '<th>동반 개념을 고정한 조건</th><th>양성 예시 수</th><th>음성 예시 수</th>'
                     '<th>고정한 집합의 AUROC</th></tr></thead><tbody>')
        for row in rows:
            side = "이미지" if row["side"] == "image" else "문장"
            target = names[row["name"]]
            other = names[row["held_concept"]]
            condition = f'{other} 표지가 {"있는" if row["held_present"] else "없는"} 예시만 사용'
            parts.append(f'<tr><th>{side}에서 {target} 있음·없음 구별</th><td>{condition}</td>'
                         f'<td>{row["positive_count"]:,}개</td><td>{row["negative_count"]:,}개</td>'
                         f'<td>{_number(row["auroc"])}</td></tr>')
        parts.append('</tbody></table></div>')
    if not parts:
        return ""
    return ('<section><h2>눈과 스키가 함께 등장하는 정도만으로 구별하는지 추가로 확인했다</h2>'
            '<p>학습한 집합과 가중치를 고정한 뒤, 동반 개념의 표지가 같은 예시 안에서 목표 개념을 구별했다. '
            '예를 들어 눈이 있는 이미지 안에서 스키가 있는 이미지와 없는 이미지를 구별했다. 이 평가에서 '
            '높은 AUROC를 얻으면 눈의 존재만으로 스키를 예측했다는 설명은 충분하지 않다. 다른 물체나 '
            '장면의 영향까지 모두 통제한 실험은 아니다.</p>'
            '<p>이미지 평가는 COCO 존재 주석을 사용했다. 이미지 주석을 양쪽에 사용한 조건의 문장 평가는 '
            '해당 이미지의 존재 주석을 사용했고, 문장 언급으로 학습한 조건의 문장 평가는 사전 기반 언급 '
            '표지를 사용했다. 따라서 두 조건의 문장 AUROC는 서로 다른 예시 집단에서 측정한 값이다.</p>'
            + "".join(parts) + '</section>')


def _label_reference(out: Path) -> str:
    path = out / "label_reference.json"
    if not path.exists():
        return ""
    reference = _read(path)
    if reference.get("kind") != "privileged_test_annotation_reference":
        raise ValueError("Unknown exact-annotation diagnostic")
    if not reference.get("uses_test_annotations") or reference.get("learned_model"):
        raise ValueError("Exact-label reference must be explicitly privileged and not a learned model")
    parts = ['<section><h2>171개 범주 조합 자체로는 많은 평가 이미지를 구별할 수 있었다</h2>',
             '<p>추가 진단에서는 SAE 예측값을 사용하지 않고, 평가 이미지의 정답 범주 벡터를 이미지와 '
             '그 이미지의 모든 캡션에 그대로 부여했다. 범주 조합이 같은 예시는 같은 점수를 받으므로 '
             '동률 예시의 순서를 무작위로 정했을 때의 검색 성공률을 계산했다.</p>',
             f'<p>이미지 {reference["image_count"]:,}장의 정답 범주 조합은 '
             f'{reference["distinct_label_sets"]:,}가지였고, {reference["images_with_unique_label_set"]:,}장은 '
             '같은 범주 조합을 가진 다른 이미지가 없었다. 따라서 현재의 검색 성능 차이를 '
             '범주 점수가 171개라는 이유만으로 설명하기는 어렵다.</p>',
             '<div class="table-scroll"><table><thead><tr><th>정답 주석을 직접 사용한 진단</th>']
    parts.extend(f'<th>{DIRECTION_NAMES[direction]} 상위 {k}개</th>'
                 for direction in DIRECTIONS for k in (1, 5, 10))
    parts.append('</tr></thead><tbody><tr><th>평가 이미지의 범주 벡터를 연결된 캡션에도 부여</th>')
    for direction in DIRECTIONS:
        for k in (1, 5, 10):
            value = float(reference["retrieval"][direction]["recall"][str(k)])
            parts.append(f'<td>{100 * value:.2f}%</td>')
    parts.append('</tr></tbody></table></div>')
    parts.append('<p class="note">이 진단만 평가 자료의 정답 주석을 직접 사용했다. 앞의 학습된 검색 방법은 '
                 '평가 주석을 검색 입력으로 사용하지 않았다. 이 수치는 실제 모델의 검색 성능이나 성능 상한이 '
                 '아니며, 모든 캡션이 연결된 이미지의 범주 정보를 완전히 알고 있다는 가정에서 얻었다. '
                 '실제 캡션에 해당 범주가 모두 언급됐다는 뜻은 아니다.</p></section>')
    return "".join(parts)


def _diagnostic_table(records: list[Record]) -> str:
    parts = ['<div class="table-scroll"><table><thead><tr><th rowspan="2">고정한 예측값에 적용한 변경</th>'
             '<th rowspan="2">개념당 특징 수</th><th rowspan="2">검색 좌표 수</th>'
             '<th colspan="3">이미지로 문장 검색</th><th colspan="3">문장으로 이미지 검색</th></tr><tr>']
    parts.extend(f'<th>상위 {k}개 성공률</th>' for _ in DIRECTIONS for k in (1, 5, 10))
    parts.append('</tr></thead><tbody>')
    previous_budget = None
    for record in records:
        if record["k"] != previous_budget:
            parts.append(f'<tr class="family"><th colspan="9">{_budget(record["k"])}로 학습한 집합을 고정했다</th></tr>')
            previous_budget = record["k"]
        parts.append(f'<tr><th>{DIAGNOSTIC_NAMES[record["condition"]]}</th>'
                     f'<td>{_budget(record["k"])}</td><td>{record["dimensions"]}개</td>')
        for direction in DIRECTIONS:
            for k in (1, 5, 10):
                parts.append(f'<td>{_recall(record, direction, k):.2f}%</td>')
        parts.append('</tr>')
    parts.append('</tbody></table></div>')
    return "".join(parts)


def _diagnostics(out: Path, records: list[Record], protocol: Record) -> str:
    path = out / "diagnostics" / "results.json"
    if not path.exists():
        return ""
    diagnostics = _read(path)
    _validate([{**r, "key": f'{r["condition"]}_{r["k"]}', "label_target": "unsupervised"}
               for r in diagnostics], protocol)
    ordinary = [r for r in diagnostics if not r["uses_test_annotations"]]
    privileged = [r for r in diagnostics if r["uses_test_annotations"]]
    lookup = {(r["k"], r["condition"]): r for r in diagnostics}
    original = {(r["label_target"], r["k"]): r for r in records}
    for record in diagnostics:
        if record["condition"] != "prediction_unit_variance":
            continue
        source = original[("propagated_presence", record["k"])]
        for direction in DIRECTIONS:
            for k in (1, 5, 10):
                np.testing.assert_allclose(_recall(record, direction, k), _recall(source, direction, k))
    exported = []
    for record in diagnostics:
        row = {k: record[k] for k in ("condition", "k", "dimensions", "uses_test_annotations")}
        row["description"] = DIAGNOSTIC_NAMES[record["condition"]]
        for direction in DIRECTIONS:
            for k in (1, 5, 10):
                row[f"{direction}_recall_at_{k}_percent"] = _recall(record, direction, k)
        exported.append(row)
    _csv(out / "diagnostics" / "summary.csv", exported)
    parts = ['<section id="diagnostics"><h2>출력 크기 보정이나 배경 좌표 제거만으로 성능 차이를 설명하기는 어려웠다</h2>',
             '<p>기존 집합과 가중치를 다시 학습하지 않고, 이미 계산한 예측값의 크기 보정과 검색에 사용할 '
             '범주만 바꿨다. 이 추가 분석은 본 실험 결과를 확인한 뒤 진행한 탐색 진단이며, 여기서 좋은 조건을 '
             '골라 기존 결과를 대체하지 않았다. 정답 표지나 예측값의 표준편차는 모두 학습 자료에서 계산했다.</p>']
    baseline16 = lookup.get((16, "prediction_unit_variance"))
    baseline_full = lookup.get(("full", "prediction_unit_variance"))
    centered16 = lookup.get((16, "centered_prediction"))
    centered_full = lookup.get(("full", "centered_prediction"))
    objects16 = lookup.get((16, "objects_prediction_unit_variance"))
    cca = _cca_control(records)
    if all(r is not None for r in (baseline16, baseline_full, centered16, centered_full, cca)):
        assert baseline16 is not None and baseline_full is not None and centered16 is not None
        assert centered_full is not None and cca is not None
        parts.append('<p class="lead"><strong>예측값을 표준편차로 나누는 처리가 성능 차이의 전부는 아니었다.</strong> '
                     f'특징 16개에서 평균만 빼면 {_direction_values(centered16)}로, 기존 '
                     f'{_direction_values(baseline16)}보다 낮았다. 전체 특징에서는 평균만 빼면 '
                     f'{_direction_values(centered_full)}로 기존 {_direction_values(baseline_full)}보다 '
                     f'높아졌지만, CCA의 {_direction_values(cca)}에는 미치지 못했다.</p>')
    if baseline16 is not None and objects16 is not None:
        parts.append('<p class="lead"><strong>배경 좌표를 빼는 것만으로 검색이 좋아지지는 않았다.</strong> '
                     f'특징 16개에서 물체 80개 좌표만 사용한 결과는 {_direction_values(objects16)}로, '
                     f'물체와 배경 171개를 함께 사용한 {_direction_values(baseline16)}보다 낮았다. '
                     '따라서 배경의 낮은 개념 구별 성능만 보고 배경 좌표 전체가 검색에 해롭다고 판단할 수는 없다.</p>')
    parts.extend([_diagnostic_table(ordinary),
                  '<p class="caption">모든 행은 이미지 주석을 양쪽 학습에 사용한 집합의 예측값을 고정한 결과이다. '
                  '정답 표지의 표준편차로 나눈 조건도 전체 CCA와의 차이를 해소하지 못했다. 물체와 배경을 '
                  '제외하는 조건은 검색 벡터의 차원과 포함한 정보를 함께 바꾼 비교이다.</p>',
                  '<p><strong>현재 남는 가설은 개념을 예측하는 목표와 이미지·문장을 서로 맞추는 목표가 '
                  '다르다는 것이다.</strong> 주석 기반 방법은 이미지와 문장에서 각 범주의 정답을 따로 예측했다. '
                  'CCA는 두 모달리티의 좌표가 함께 변하도록 학습했다. 각 범주의 양성·음성 순위가 좋아도 '
                  '같은 이미지·문장 쌍의 여러 예측값이 함께 일치할 필요는 없다. 이 차이가 실제 성능 차이에 '
                  '얼마나 기여했는지는 현재 진단에서 분리해 측정하지 않았다.</p>'])
    if privileged:
        parts.extend(['<details><summary>한쪽 예측값을 평가 정답 주석으로 바꾼 추가 진단을 확인한다</summary>',
                      '<p class="note">아래 조건은 한쪽 모달리티에 평가 자료의 정답 범주를 직접 제공했다. '
                      '실제로 사용할 수 있는 검색 방법이나 성능 상한으로 해석하지 않았다. 다른 쪽의 예측값도 '
                      '학습 정답 표지의 표준편차로 나눈 값이므로, 기존 조건에서 한쪽 예측 오류만 제거한 '
                      '비교가 아니다.</p>',
                      _diagnostic_table(privileged),
                      '<p>평가 정답 주석을 대입하면 점수의 분포와 벡터 간 유사도, 캡션 사이의 동률도 함께 바뀐다. '
                      '특히 연결된 이미지의 정답 주석을 문장에 부여하면 같은 이미지에 속한 여러 캡션이 동일한 '
                      '벡터를 갖는다. 따라서 이미지로 문장을 검색할 때 상위 1개와 5개의 성공률이 같아질 수 있다. '
                      '이 결과의 상승·하락을 이미지나 문장 한쪽의 예측 오류가 차지하는 비율로 해석할 수는 없다.</p>',
                      '</details>'])
    parts.append('</section>')
    return "".join(parts)


def report(out: Path) -> Path:
    out = out.resolve()
    protocol = _read(out / "protocol.json")
    protocol["evaluation_images"] = protocol.get("evaluation_images", protocol.get("test_images"))
    protocol["evaluation_captions"] = protocol.get("evaluation_captions", protocol.get("test_captions"))
    records = [_read(path) for path in sorted((out / "results").glob("*.json"))]
    if not records:
        raise ValueError("No completed experiment results")
    _validate(records, protocol)
    records.sort(key=_order)
    _export(out, records)
    _fonts()
    retrieval_plot = _retrieval_plot(out, records)
    semantic_plot = _semantic_plot(out, records)
    supervised = [r for r in records if r["label_target"] != "unsupervised"]
    controls171 = [r for r in records if r["label_target"] == "unsupervised" and r["dimensions"] == 171]
    references256 = [r for r in records if r["label_target"] == "unsupervised" and r["dimensions"] == 256]
    expected = protocol.get("expected_conditions")
    status = (f'계산한 {len(records)}개 조건을 반영했다.' if expected is None else
              f'예정한 {expected}개 조건 중 {len(records)}개 결과를 반영했다.')
    fit_images = protocol.get("fit_images", 0)
    tune_images = protocol.get("calibration_images", protocol.get("tune_images", 0))
    diagnostic_navigation = ('<a href="#diagnostics">4. 성능 차이의 가능한 원인을 확인한다. '
                             '같은 예측값의 크기 보정과 범주 구성을 바꾼 결과를 비교한다.</a>'
                             if (out / "diagnostics" / "results.json").exists() else "")
    css = """
    *{box-sizing:border-box}body{margin:0;background:#f2f4f5;color:#203442;
    font:15px/1.8 -apple-system,BlinkMacSystemFont,'Apple SD Gothic Neo','Malgun Gothic',sans-serif}
    main{max-width:1440px;margin:auto;padding:35px 28px 65px}h1{font-size:32px;line-height:1.4}
    h2{font-size:24px;line-height:1.5;margin:0 0 18px}h3{font-size:18px;margin-top:28px}
    p{margin:14px 0;max-width:1180px}section{background:#fff;border:1px solid #dbe1e5;border-radius:8px;
    padding:27px;margin:24px 0}.lead{font-size:17px}.note{background:#eaf0f3;padding:14px 18px;border-radius:5px}
    .table-scroll{overflow-x:auto;margin:20px 0}table{width:100%;border-collapse:collapse;min-width:950px;font-size:13px}
    th,td{padding:11px 10px;border-bottom:1px solid #e3e7eb;text-align:left;vertical-align:top}
    thead th{background:#244359;color:white}thead tr:nth-child(2) th{background:#34596f}
    tbody th{font-weight:500;min-width:220px}.family th{background:#e8eef2;font-weight:650;padding:8px 12px}
    td{font-variant-numeric:tabular-nums}strong{font-weight:750}td strong{color:#1f6189}
    small{display:block;font-size:11px;color:#607685}.chart{width:100%;height:auto;margin-top:20px}
    .caption{font-size:13px;color:#526c7c}.case-table{min-width:600px}.case-table tbody th{min-width:65px;width:90px}
    details{margin:20px 0}summary{cursor:pointer;color:#245a79;font-weight:650}code{overflow-wrap:anywhere}
    nav{display:grid;gap:7px;margin:20px 0}a{color:#245f84}nav a{text-decoration:none}
    @media(max-width:720px){main{padding:20px 12px}section{padding:18px}h1{font-size:26px}h2{font-size:21px}}
    """
    body = ['<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" '
            'content="width=device-width, initial-scale=1"><title>COCO 주석으로 만든 개념 집합의 검색 성능</title>'
            f'<style>{css}</style></head><body><main><h1>COCO 주석으로 만든 개념 집합의 검색 성능</h1>',
            f'<p class="note">{status} 검색 성공률은 COCO val2017의 이미지 '
            f'{protocol["evaluation_images"]:,}장과 문장 {protocol["evaluation_captions"]:,}개에서 측정했다.</p>',
            '<p class="lead">이미지와 문장에서 같은 COCO 범주를 예측하도록 SAE activation 집합을 각각 학습한 '
            '다음, 그 예측값들로 이미지·문장을 검색했다. 개념의 이름을 주석으로 지정한 집합이 검색에도 유리한지 '
            '확인하는 실험이다. 검색할 때는 평가 자료의 범주 주석을 사용하지 않았다.</p>',
            '<nav><a href="#results">1. 주석으로 집합을 만든 검색 결과를 확인한다. 특징 수와 문장 표지의 효과를 비교한다.</a>'
            '<a href="#controls">2. 주석 없이 만든 같은 크기의 표현과 비교한다. CCA와 Sparse CCA의 결과를 확인한다.</a>'
            '<a href="#semantics">3. 학습한 집합이 개념을 구별하는지 확인한다. 범주별 AUROC 분포를 비교한다.</a>'
            f'{diagnostic_navigation}</nav>',
            '<section><h2>같은 개념의 이름을 주석으로 지정하고 양쪽 특징 집합을 학습했다</h2>',
            '<p>각 COCO 범주에 대해 이미지 SAE activation의 작은 가중합과 문장 SAE activation의 작은 가중합을 '
            '각각 학습했다. 예를 들어 스키가 있는지 예측하는 이미지 특징 집합과 스키에 대응하는 문장 특징 집합을 '
            '만들었다. 물체 80개와 배경 91개를 사용하므로 이미지와 문장은 각각 171개 개념 점수로 표현된다.</p>',
            '<p>특징을 하나씩 추가하면서 범주 존재 표지의 예측 오차를 줄이는 선형 회귀를 학습했다. 개념 하나에 '
            '사용할 특징 수를 최대 1개·4개·8개·16개로 제한했고, 전체 특징을 사용하는 조건도 비교했다. '
            '음수 가중치를 허용했다. 이미지 쪽과 문장 쪽이 같은 activation 번호를 선택할 필요는 없다.</p>',
            '<p>문장 표지는 두 방식으로 만들었다. 첫 조건은 이미지에 스키가 있으면 연결된 모든 캡션에도 스키의 '
            '존재 표지를 부여했다. 문장이 스키를 생략했어도 양성으로 학습한다. 두 번째 조건은 이미지에는 같은 '
            '범주 주석을 사용하고, 문장에는 기존 사전으로 확인한 스키 언급 여부를 사용했다. 두 조건은 문장에 '
            '생략된 개념을 예측하도록 요구하는지가 다르다.</p>',
            f'<p>가중치와 특징 선택에는 이미지 {fit_images:,}장과 그 캡션을 사용했다. '
            f'분리해 둔 학습 이미지 {tune_images:,}장은 이번 특징 선택이나 설정 조정에 사용하지 않았다. '
            '회귀 계수의 크기를 제한하는 벌점은 0.01로 고정했다. '
            '각 개념 점수는 학습 자료에서 구한 평균과 표준편차로 보정한 뒤, 전체 171개 점수 벡터의 '
            '코사인 유사도로 검색했다. 이미지 검색의 정답은 해당 이미지에 연결된 원래 캡션 중 하나이며, '
            '문장 검색의 정답은 해당 캡션의 원래 이미지이다.</p>',
            '<p class="note">여기서 주석 기반 oracle은 올바른 개념 이름을 미리 제공한다는 뜻이다. 각 activation의 '
            '완벽한 의미 정답을 알고 있다는 뜻은 아니며 검색 성능의 수학적 상한도 아니다. 한 개념을 예측하도록 '
            '학습해도 동반 물체나 장면을 사용하는 특징이 선택될 수 있다.</p></section>',
            '<section id="results"><h2>주석으로 만든 집합의 검색 성능을 비교했다</h2>']
    body.extend(f'<p class="lead">{lead}</p>' for lead in _leads(records))
    body.extend([_retrieval_table(supervised),
                 '<p class="caption">Recall@1·5·10은 정답이 각각 검색 결과 상위 1개·5개·10개 안에 포함된 '
                 '질문의 비율이다. 표의 굵은 수치는 각 검색 지표에서 가장 높은 관측값이다. 동일 이미지의 여러 '
                 '캡션은 독립적인 학습 이미지로 세지 않았다.</p>'])
    if retrieval_plot:
        body.append(_image(out, retrieval_plot, "주석 조건과 개념당 특징 수에 따른 양방향 Recall@10"))
    body.extend(['</section><section id="controls"><h2>동일한 171좌표를 사용한 CCA와 비교했다</h2>',
                 '<p>CCA는 이미지와 문장 사이의 공통 변화를 잘 나타내는 두 선형 변환을 학습하며 개념 주석은 '
                 '사용하지 않는다. Sparse CCA는 그 변환의 좌표마다 사용할 원래 SAE 특징 수를 제한한다. '
                 '주석 기반 방법은 범주 하나당 좌표 하나를 만들지만 CCA의 좌표에는 범주 이름이 지정되지 않는다.</p>',
                 '<p>학습 예측 분산을 1로 보정한 CCA 조건은 주석 기반 방법과 같은 출력 크기 보정을 사용했다. '
                 'CCA에서 원래 학습한 좌표 크기를 유지한 조건도 구분해 표시했다. CCA는 같은 학습 자료의 '
                 '표준화된 SAE activation을 입력받았다.</p>',
                 _retrieval_table(supervised + controls171),
                 '<p class="caption">위 표는 출력 좌표 수를 171개로 맞췄다. 특징 집합을 학습하는 목표와 좌표별 '
                 '중복 정도까지 동일하게 맞춘 비교는 아니므로 결과 차이를 주석 사용 하나의 효과로 해석하지 않았다.</p>'])
    if references256:
        body.extend(['<details><summary>기존 256좌표 결과를 함께 확인한다</summary>',
                     '<p>이전 실험의 256좌표 결과는 출력 크기가 다르므로 참고값으로 구분했다.</p>',
                     _retrieval_table(references256), '</details>'])
    body.extend(['</section><section id="semantics"><h2>개념 구별 능력과 개별 이미지 검색 성능을 구분했다</h2>',
                 '<p>AUROC는 개념이 있는 예시가 없는 예시보다 높은 점수를 받을 확률이며, 같은 점수에는 절반을 '
                 '반영한다. 0.5는 두 집단을 구별하지 못하는 수준이며 1은 완전한 순위 구별을 뜻한다. 여기서는 '
                 '개념의 있음·없음을 평가했고 원본·가림 이미지를 비교하지 않았다.</p>',
                 '<p>두 학습 조건을 비교할 때 이미지에는 동일한 범주 존재 주석을, 문장에는 동일한 사전 기반 '
                 '언급 표지를 사용했다. 문장으로 해당 이미지의 범주 존재를 예측하는 AUROC도 별도로 표시했다. '
                 '이미지에 있는 범주와 문장에서 언급한 범주는 다른 표지이므로 이 두 AUROC를 같은 정답에 대한 '
                 '성능으로 비교하면 안 된다.</p>'])
    if semantic_plot:
        body.extend([_image(out, semantic_plot, "같은 개념 표지로 평가한 이미지와 문장의 범주별 AUROC 분포"),
                     '<p class="caption">상자는 25%·75% 분위수를, 가운데 선은 중앙값을 표시한다. 수염은 '
                     '상자 길이의 1.5배 안에 있는 관측값까지 표시하며, 그 밖의 값은 그림에서만 생략했다. '
                     '양성이나 음성이 없는 범주는 AUROC를 계산할 수 없어 그림에서 제외했고 표와 CSV에 남겼다.</p>'])
    body.extend([_semantic_table(supervised),
                 '<p>AUROC는 범주 하나의 양성·음성 순위를 평가한다. 검색은 여러 범주 점수를 함께 사용해 '
                 '정확한 이미지·문장 쌍을 찾아야 하므로 다른 지표이다. 각 개념 AUROC가 높다는 사실만으로 '
                 '두 모달리티의 전체 점수 벡터가 검색에 적합하게 대응한다고 판단할 수는 없다.</p>',
                 '</section>', _diagnostics(out, records, protocol), _label_reference(out),
                 _cases(out, records), _conditional_table(records),
                 '<section><h2>이 결과로 판단할 수 있는 범위를 제한했다</h2>',
                 '<p>주석 기반 방법은 학습 자료의 범주 정답을 사용한 지도학습 조건이다. 주석 없는 대응 방법보다 '
                 '유리한 정보를 제공받았으므로 같은 감독 조건의 방법 비교로 해석하지 않았다. 평가 자료의 주석은 '
                 '검색 벡터를 만드는 데 사용하지 않았고 개념 구별 AUROC를 계산할 때만 사용했다.</p>',
                 '<p>문장 언급 정답은 기존 사전에서 얻은 자동 표지이며 모든 문장을 사람이 검수한 의미 정답이 '
                 '아니다. COCO val2017은 이전 분석에서 반복해서 사용한 탐색 평가 자료이며, 이번에 처음 확인한 '
                 '최종 시험 자료가 아니다. 결과는 기존 SAE와 현재 자료에 대한 비교이며 새 SAE 학습 결과가 아니다.</p>',
                 '<details><summary>결과 파일의 위치를 확인한다</summary>'])
    for name, description in (("summary.csv", "모든 조건의 검색 성공률과 범주군별 AUROC를 저장했다"),
                              ("semantic_per_concept.csv", "각 범주의 개념 구별 AUROC와 유효 표본 수를 저장했다"),
                              ("protocol.json", "학습·평가 자료와 계산 설정을 저장했다")):
        path = out / name
        body.append(f'<p><a href="{html.escape(str(path))}">{html.escape(str(path))}</a>에 {description}.</p>')
    diagnostic_csv = out / "diagnostics" / "summary.csv"
    if diagnostic_csv.exists():
        body.append(f'<p><a href="{html.escape(str(diagnostic_csv))}">{html.escape(str(diagnostic_csv))}</a>에 '
                    '추가 진단의 양방향 Recall@1·5·10을 저장했다.</p>')
    body.extend(['<p>그래프는 HTML 내부에 포함했으므로 이 파일만 열어도 결과를 확인할 수 있다.</p>',
                 '</details></section></main></body></html>'])
    destination = out / "report.html"
    destination.write_text("".join(body), encoding="utf-8")
    print(json.dumps({"report": str(destination), "completed_conditions": len(records)}, ensure_ascii=False))
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    report(parser.parse_args().run)


if __name__ == "__main__":
    main()
