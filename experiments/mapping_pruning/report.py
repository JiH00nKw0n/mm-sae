"""Render the saved frozen-pruning comparison without rerunning retrieval."""

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
from matplotlib.lines import Line2D

from experiments.mapping_pruning.sinkhorn_report import render_sinkhorn


DIRECTIONS = ("image_to_text", "text_to_image")
RANKS = (1, 5, 10)
FAMILIES = ("cca", "procrustes")
NAMES = {"cca": "CCA", "procrustes": "Procrustes"}
COLORS = ("#163b59", "#2675ac", "#5aabd0", "#762746", "#af467c", "#d982ad")
DEFAULT_RUN = Path(__file__).resolve().parents[2] / "runs/mapping-pruning-2026-10-04"
Record = dict[str, Any]


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _condition(record: Record) -> str:
    k = record["metadata"]["k"]
    if k is None:
        return "계수를 모두 유지한 기준"
    if record["family"] == "cca":
        return f"공통 좌표마다 이미지·문장 입력을 각각 최대 {k}개 유지"
    return f"이미지·문장 특징마다 상대 특징을 최대 {k}개 유지"


def _counts(record: Record) -> tuple[int, int]:
    structure = record["metadata"]["structure"]
    return (sum(value["nonzero_coefficients"] for value in structure.values()),
            sum(value["total_coefficients"] for value in structure.values()))


def _recall(record: Record, direction: str, rank: int) -> float:
    return 100 * float(record["retrieval"][direction]["recall"][str(rank)])


def _load(out: Path) -> tuple[dict[str, list[Record]], Record, Record, Record]:
    population, manifest = _read(out / "population.json"), _read(out / "manifest.json")
    verification = _read(out / "reference_verification.json")
    results: dict[str, list[Record]] = {}
    expected_k = [None, *manifest["config"]["k_grid"]]
    for family in FAMILIES:
        records = [_read(out / "results" / f'{family}_{"full" if k is None else k}.json')
                   for k in expected_k]
        for record, k in zip(records, expected_k, strict=True):
            if record["family"] != family or record["metadata"]["k"] != k:
                raise ValueError("Result identity does not match the requested condition")
            if not record["metadata"]["preserve_sign_and_weights"]:
                raise ValueError("This report requires unchanged retained signed coefficients")
            for direction in DIRECTIONS:
                metric = record["retrieval"][direction]
                expected = (population["test_images"], population["test_captions"])
                if direction == "text_to_image":
                    expected = expected[::-1]
                if (metric["query_count"], metric["candidate_count"]) != expected:
                    raise ValueError("A result uses a different retrieval population")
        for direction in DIRECTIONS:
            if any(abs(verification[family][direction][str(k)]) > 1e-12 for k in RANKS):
                raise ValueError("The full-retention recall does not match its original reference")
        results[family] = records
    return results, population, manifest, _read(out / "verification.json")


def _write_csv(out: Path, results: dict[str, list[Record]]) -> None:
    rows = []
    for family in FAMILIES:
        for record in results[family]:
            count, total = _counts(record)
            structure = record["metadata"]["structure"]
            row = {
                "method": NAMES[family],
                "k": "full" if record["metadata"]["k"] is None else record["metadata"]["k"],
                "condition_definition": _condition(record),
                "coefficient_definition": ("이미지·문장 SAE 입력에서 공통 좌표로 가는 계수의 합계"
                                           if family == "cca" else "이미지 SAE 특징과 문장 SAE 특징 사이의 계수"),
                "retained_coefficients": count,
                "original_coefficients": total,
                "retained_coefficient_percent": 100 * count / total,
                "retained_negative_coefficients": sum(v["negative_coefficients"] for v in structure.values()),
                "image_coefficients": structure.get("image", {}).get("nonzero_coefficients", ""),
                "text_coefficients": structure.get("text", {}).get("nonzero_coefficients", ""),
                "used_image_features": (structure["image"]["covered_rows"] if family == "cca"
                                        else structure["image_by_text"]["covered_rows"]),
                "used_text_features": (structure["text"]["covered_rows"] if family == "cca"
                                       else structure["image_by_text"]["covered_columns"]),
            }
            for direction in DIRECTIONS:
                row[f"{direction}_query_count"] = record["retrieval"][direction]["query_count"]
                row[f"{direction}_candidate_count"] = record["retrieval"][direction]["candidate_count"]
                row[f"{direction}_zero_norm_queries"] = record["retrieval"][direction]["zero_norm_query_count"]
                for rank in RANKS:
                    row[f"{direction}_recall_at_{rank}_percent"] = _recall(record, direction, rank)
            rows.append(row)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _figure(out: Path, results: dict[str, list[Record]]) -> None:
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True)
        handles = []
        for axis, family in zip(axes, FAMILIES, strict=True):
            reference = results[family][0]
            pruned = sorted(results[family][1:], key=lambda r: r["metadata"]["k"])
            budgets = [r["metadata"]["k"] for r in pruned]
            for index, (direction, rank) in enumerate((d, k) for d in DIRECTIONS for k in RANKS):
                direction_name = "Image to text" if direction == "image_to_text" else "Text to image"
                line, = axis.plot(budgets, [_recall(r, direction, rank) for r in pruned],
                                  color=COLORS[index], marker="o" if index < 3 else "s", markersize=4.5,
                                  linewidth=1.8, label=f"{direction_name}, Recall@{rank}")
                axis.axhline(_recall(reference, direction, rank), color=COLORS[index],
                             linestyle=(0, (5, 4)), linewidth=1.2, alpha=0.72)
                if family == "cca":
                    handles.append(line)
            axis.set_xscale("log", base=2)
            axis.set_xticks(budgets, labels=[str(k) for k in budgets])
            axis.set_xlim(min(budgets) * 0.91, max(budgets) * 1.10)
            axis.set_ylim(bottom=0)
            axis.grid(axis="y", color="#dce2e7", linewidth=0.7)
            axis.set_axisbelow(True)
            axis.set_title(NAMES[family], fontweight="bold", fontsize=14, loc="left", pad=15)
            axis.set_xlabel("Maximum SAE inputs per common coordinate and modality (k)" if family == "cca"
                            else "Maximum cross-modal neighbors per SAE feature (k)", labelpad=10, fontsize=9)
        axes[0].set_ylabel("Recall (%)")
        fig.suptitle("Frozen coefficient pruning with signed weights preserved", fontsize=16, y=0.98)
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.062),
                   ncol=3, frameon=False, columnspacing=2.8, fontsize=10)
        fig.legend(handles=[Line2D([0], [0], color="#57636c", marker="o", label="Pruned transform"),
                            Line2D([0], [0], color="#57636c", linestyle=(0, (5, 4)),
                                   label="Full transform reference")],
                   loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2, frameon=False, fontsize=9)
        fig.subplots_adjust(top=0.86, bottom=0.28, left=0.06, right=0.985, wspace=0.09)
        fig.savefig(out / "recall_curves.png", dpi=180, facecolor="white")
        plt.close(fig)


def _table(records: list[Record]) -> str:
    family = records[0]["family"]
    maxima = {(direction, rank): max(_recall(r, direction, rank) for r in records)
              for direction in DIRECTIONS for rank in RANKS}
    header = ("공통 좌표의 입력 계수 수" if family == "cca" else "이미지·문장 특징 쌍의 계수 수")
    parts = [f'<div class="table-scroll"><table><thead><tr><th rowspan="2" scope="col">유지 조건</th>'
             f'<th rowspan="2" scope="col">{header}</th>'
             '<th rowspan="2" scope="col">전체 계수 중 유지 비율</th>'
             '<th colspan="3" scope="colgroup">이미지로 문장 검색</th>'
             '<th colspan="3" scope="colgroup">문장으로 이미지 검색</th></tr><tr>']
    parts += [f'<th scope="col">Recall@{rank}</th>' for _ in DIRECTIONS for rank in RANKS]
    parts.append("</tr></thead><tbody>")
    for record in records:
        count, total = _counts(record)
        structure = record["metadata"]["structure"]
        if family == "cca":
            detail = (f'이미지 계수 {structure["image"]["nonzero_coefficients"]:,}개<br>'
                      f'문장 계수 {structure["text"]["nonzero_coefficients"]:,}개')
        else:
            detail = f'음수 계수 {structure["image_by_text"]["negative_coefficients"]:,}개를 포함한다'
        css = ' class="reference"' if record["metadata"]["k"] is None else ""
        parts.append(f'<tr{css}><th scope="row">{_condition(record)}</th>'
                     f'<td>{count:,}개<span class="detail">{detail}</span></td>'
                     f'<td>{100 * count / total:.2f}%</td>')
        for direction in DIRECTIONS:
            for rank in RANKS:
                value = _recall(record, direction, rank)
                formatted = f"{value:.2f}%"
                if abs(value - maxima[direction, rank]) < 1e-10:
                    formatted = f"<strong>{formatted}</strong>"
                parts.append(f"<td>{formatted}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _lead(results: dict[str, list[Record]]) -> str:
    sentences = []
    for family in FAMILIES:
        reference = results[family][0]
        record = max(results[family][1:], key=lambda r: r["metadata"]["k"])
        count, total = _counts(record)
        k = record["metadata"]["k"]
        difference_i = _recall(reference, "image_to_text", 1) - _recall(record, "image_to_text", 1)
        difference_t = _recall(reference, "text_to_image", 1) - _recall(record, "text_to_image", 1)
        change_i = f"{abs(difference_i):.2f}%포인트 {'낮았다' if difference_i >= 0 else '높았다'}"
        change_t = f"{abs(difference_t):.2f}%포인트 {'낮았다' if difference_t >= 0 else '높았다'}"
        definition = (f"공통 좌표마다 이미지·문장 입력을 각각 최대 {k}개 유지했을 때" if family == "cca"
                      else f"각 이미지·문장 특징에 연결된 상대 특징을 최대 {k}개 유지했을 때")
        sentences.append(f'<p><strong>{NAMES[family]}는 {definition} 전체 계수의 '
                         f'{100 * count / total:.2f}%를 사용했다.</strong> '
                         f'첫 번째 검색 결과에 정답이 포함된 비율은 전체 계수를 유지한 기준보다 '
                         f'이미지로 문장을 검색할 때 {change_i}. 문장으로 이미지를 검색할 때는 {change_t}.</p>')
    return "".join(sentences)


def report(out: Path) -> None:
    out = out.resolve()
    results, population, manifest, verification = _load(out)
    _write_csv(out, results)
    _figure(out, results)
    sinkhorn = render_sinkhorn(out, population)
    has_sinkhorn = bool(sinkhorn["body"])
    title = ("CCA·Procrustes·Sinkhorn의 연결 제한 비교" if has_sinkhorn
             else "CCA와 Procrustes의 계수 제거 비교")
    dimensions = results["cca"][0]["metadata"]["dimensions"]
    config = manifest["config"]
    image_count, text_count = population["test_images"], population["test_captions"]
    feature_i, feature_t = population["image_features"], population["text_features"]
    image_data = base64.b64encode((out / "recall_curves.png").read_bytes()).decode("ascii")
    independent_text = ""
    independent_path = out / "independent_blas_verification.json"
    if independent_path.exists():
        independent = _read(independent_path)
        keys = [record["key"] for records in results.values() for record in records]
        if (independent["all_scores_finite_and_bounded"]
                and all(independent["rank_mismatches"][key][direction] == 0
                        for key in keys for direction in DIRECTIONS)):
            independent_text = (f"{len(keys)}개 조건마다 {image_count + text_count:,}개 질문의 정답 순위를 "
                                "별도로 다시 계산했고, 저장된 순위와 모두 일치했다.")
    css = """
    :root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f5f7f8;color:#23333f;
    font:16px/1.75 -apple-system,BlinkMacSystemFont,'Apple SD Gothic Neo','Malgun Gothic',sans-serif}
    main{max-width:1420px;margin:44px auto;padding:0 28px 56px}h1{font-size:32px;letter-spacing:-1px;
    line-height:1.4;margin:8px 0 22px}h2{font-size:23px;line-height:1.5;margin:0 0 13px}
    p{max-width:1120px;margin:12px 0}.date{font-size:13px;color:#637583;letter-spacing:.05em}
    .lead{padding:20px 28px;background:#e7eef3;border-left:4px solid #3f6b88;margin:24px 0 28px}
    .lead p{margin:8px 0}section{background:white;border:1px solid #dde4e8;border-radius:10px;
    padding:26px;margin:22px 0}.note{color:#526674;font-size:14px}figure{margin:26px 0}
    figure img{display:block;width:100%;height:auto;background:white;border:1px solid #e0e6e9}
    figcaption{font-size:14px;color:#526674;margin-top:10px}.table-scroll{overflow-x:auto;margin-top:22px}
    table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}
    th,td{padding:12px 10px;border-bottom:1px solid #dde4e8;text-align:right;white-space:nowrap}
    thead th{background:#edf2f5;color:#344e60;text-align:center;font-size:13px}
    tbody th{text-align:left;font-weight:400;white-space:normal;min-width:190px;max-width:260px}
    tbody .reference{background:#f1f6f8}.reference th{font-weight:600}td strong{color:#102f46;font-weight:800}
    .detail{display:block;font-size:11px;line-height:1.55;color:#687d8b;margin-top:3px}
    h3{font-size:18px;line-height:1.6;margin:26px 0 8px}.sinkhorn-details{margin:20px 0;
    border:1px solid #dbe4e9;border-radius:6px;padding:14px 18px}.sinkhorn-details summary{
    cursor:pointer;font-weight:600;color:#244e6b}.structure-table thead th{white-space:normal;min-width:92px}
    .structure-table tbody th{min-width:170px}code{font-size:12px;overflow-wrap:anywhere;white-space:normal;color:#466173}
    .sources p{max-width:none}.foot{font-size:13px;color:#687d8b}@media(max-width:700px){main{padding:0 16px;
    margin-top:24px}section{padding:18px}.lead{padding:15px 18px}h1{font-size:26px}h2{font-size:20px}}
    @media print{body{background:white}main{margin:0;max-width:none;padding:0}section{break-inside:avoid;
    border:0;padding:16px 0}figure{break-inside:avoid}.table-scroll{overflow:visible}th,td{padding:8px 5px}}
    """
    body = f"""
    <main><div class="date">2026년 10월 4일 · 저장된 실험 결과로 작성했다</div>
    <h1>{title}</h1>
    <p>희소 자동부호화기(SAE)는 입력 표현에서 각 특징이 나타나는 정도를 수치로 표현하는 모델이다.
    정준상관분석(CCA)은 짝지어진 이미지와 문장 사이의 상관이 높은 공통 좌표를 구하는 방법이다.
    공통 좌표 하나는 각 입력 특징의 값에 계수를 곱해 더한 값이다.
    Procrustes는 표현의 길이와 각도를 보존하는 직교 변환으로 두 표현을 정렬하는 방법이다.
    이번 비교에서는 두 방법에서 구한 작은 계수를 0으로 바꾼 뒤 검색 성능을 측정했다.</p>
    {'<p>Sinkhorn은 이미지 특징과 문장 특징의 각 쌍에 0 이상의 대응 가중치를 부여하는 방법이다. '
     'ε는 가중치를 여러 특징 쌍에 분산하려는 정도를 조절하는 값이다. '
     'Sinkhorn에서는 ε를 바꾼 결과와 작은 가중치를 0으로 바꾼 결과를 함께 확인했다.</p>' if has_sinkhorn else ''}
    {_lead(results)}
    {sinkhorn["lead"]}
    <p>계수를 삭제해도 두 방법의 검색 성능을 평가할 수 있었다. 다만 두 방법에서 계수를 세는 단위가 다르므로
    같은 제한값을 같은 수의 특징 쌍으로 해석할 수 없다. CCA와 Procrustes의 위 수치는 미리 정한 제한값 중 가장 큰 값의 결과이며,
    평가 결과로 ε나 연결 수를 최종 선택하지 않았다. SAE를 재학습하지 않았으며,
    Sinkhorn의 대응 가중치만 ε별로 다시 계산했다.</p>
    <section><h2>평가 대상과 계산 방법을 고정했다</h2>
    <p>이번 비교에서는 기존 SAE의 이미지 특징 {feature_i:,}개와 문장 특징 {feature_t:,}개를 그대로 사용했다.
    학습 자료에서 구한 평균을 빼고 표준편차로 나눈 값에 저장된 변환 계수를 적용했다.</p>
    <p>모든 조건에서 이미지 {image_count:,}개와 문장 {text_count:,}개를 사용했다.
    이미지로 문장을 검색할 때는 이미지 {image_count:,}개 각각에 대해 문장 {text_count:,}개를 비교했다.
    문장으로 이미지를 검색할 때는 문장 {text_count:,}개 각각에 대해 이미지 {image_count:,}개를 비교했다.
    계수를 제거한 뒤에도 검색 질문이나 후보를 제외하지 않았다.</p>
    <p>Recall@1·5·10은 각각 상위 1·5·10개 검색 결과에 정답이 하나 이상 포함된 질문의 비율이다.
    이미지에 연결된 모든 문장을 정답으로 인정했고, 문장 검색의 정답 이미지는 해당 문장이 속한 이미지로 정했다.
    검색 순위는 두 표현의 내적을 각각의 길이로 나눈 코사인 유사도로 정했다.</p>
    <p class="note">CCA와 Procrustes에서는 계수의 절댓값으로 유지할 항목을 고른 뒤 나머지 계수를 0으로 만들었다.
    남긴 계수의 값과 부호는 바꾸지 않았다. CCA와 Procrustes 표의 굵은 수치는 각 방법 안에서 해당 검색 지표가 가장 높은 값이다.</p>
    </section>
    <figure><img src="data:image/png;base64,{image_data}" alt="CCA와 Procrustes에서 유지 제한값별 검색 성능을 표시한 두 그래프">
    <figcaption>실선과 점은 계수를 제거한 결과이고, 같은 색의 수평 점선은 전체 계수를 유지한 기준이다.
    가로축의 제한값은 1·2·4·8·16·32·64로 두 배씩 증가한다. CCA의 제한값은 공통 좌표마다 사용하는 입력 특징 수이고,
    Procrustes의 제한값은 원래 특징마다 연결되는 상대 특징 수이다.</figcaption></figure>
    <section><h2>CCA에서는 각 공통 좌표를 계산하는 입력 계수를 줄였다</h2>
    <p>저장된 {dimensions:,}개 공통 좌표를 고정하고, 각 좌표에 들어가는 이미지 입력과 문장 입력을 각각 제한했다.
    예를 들어 제한값 8에서는 각 좌표를 이미지 특징 최대 8개와 문장 특징 최대 8개로 계산한다.
    두 표현을 각각 공통 좌표로 변환한 뒤 코사인 유사도를 구했다.</p>
    <p class="note">계수 수는 이미지 입력 계수와 문장 입력 계수의 합이다. 이미지·문장 특징 쌍의 수가 아니다.
    계수를 모두 유지한 경우에는 이미지 계수 {feature_i * dimensions:,}개와 문장 계수 {feature_t * dimensions:,}개가 있다.</p>
    {_table(results["cca"])}</section>
    <section><h2>Procrustes에서는 원래 이미지 특징과 문장 특징 사이의 계수를 줄였다</h2>
    <p>기존 정렬 결과를 이미지 특징 {feature_i:,}개와 문장 특징 {feature_t:,}개 사이의 계수로 나타냈다.
    각 이미지 특징과 각 문장 특징에서 절댓값이 큰 계수를 각각 고른 뒤, 양쪽에서 모두 선택된 계수만 남겼다.
    따라서 제한값 8에서는 각 원래 특징에 연결된 상대 특징이 최대 8개다.</p>
    <p>현재 자료에서는 문장을 이미지 특징 공간으로 변환하고 원래 이미지 표현과 비교했다.
    계수를 모두 유지하면 원래 Procrustes 계산과 같은 검색 결과를 얻는다.
    계수를 제거한 뒤에는 직교성이 보장되지 않으며, 변환 후 표현의 길이로 코사인 유사도를 다시 계산했다.</p>
    <p class="note">계수 하나는 원래 이미지 특징 하나와 문장 특징 하나의 쌍에 해당한다.
    전체 계수는 {feature_i:,}개 이미지 특징과 {feature_t:,}개 문장 특징을 조합한 {feature_i * feature_t:,}개다.</p>
    {_table(results["procrustes"])}</section>
    {sinkhorn["body"]}
    <section><h2>이번 결과가 확인하는 범위</h2>
    <p>CCA와 Procrustes의 비교는 저장된 계수를 삭제했을 때의 성능 변화를 확인한다.
    제한된 입력 수를 고려해 다시 최적화하는 희소 CCA를 학습한 결과는 아니다.
    CCA에서는 각 좌표의 입력을 제한했고, Procrustes에서는 원래 특징 사이의 연결을 제한했다.
    두 방법의 제한값이나 남은 계수 비율만으로 특징 대응의 품질이 같다고 판단할 수 없다.</p>
    <p>음수 계수는 반대 방향의 가중 기여를 나타내며, 비음수 대응 가중치와 같은 의미가 아니다.</p>
    {sinkhorn["scope"]}
    <p>전체 계수를 유지한 CCA와 Procrustes는 각각 기존 결과의 여섯 검색 지표와 정확히 일치했다.
    저장된 질문별 정답 순위에서 {verification["verified_recalls"]:,}개 검색 지표를 다시 계산해 확인했다.
    {independent_text}
    이 결과만으로 작은 성능 차이의 통계적 유의성이나 새로운 자료에서의 성능은 판단할 수 없다.</p>
    </section>
    <section class="sources"><h2>결과를 다시 확인할 수 있는 파일</h2>
    {sinkhorn["sources"]}
    <p>조건별 검색 결과와 계수 수는 <code>{html.escape(str(out / "results"))}</code>에 있다.</p>
    <p>숫자 표는 <code>{html.escape(str(out / "summary.csv"))}</code>에 저장했다.
    그림은 <code>{html.escape(str(out / "recall_curves.png"))}</code>에 저장했다.</p>
    <p>원래 CCA와 Procrustes 결과는 <code>{html.escape(str(Path(config["projection_run"]) / "results"))}</code>에 있다.
    기준값 차이는 <code>{html.escape(str(out / "reference_verification.json"))}</code>에서 확인했다.</p>
    <p>학습에 사용한 통계와 특징 선택은 <code>{html.escape(str(Path(config["parent_run"]) / "moments.npz"))}</code>에서,
    평가 활성값은 <code>{html.escape(str(Path(config["source_run"]) / "activations/val2017"))}</code>에서 읽었다.</p>
    </section></main>
    """
    document = ('<!doctype html><html lang="ko"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width, initial-scale=1">'
                f'<title>{title}</title><style>{css}</style></head>'
                f'<body>{body}</body></html>')
    (out / "report.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    report(args.run)
    print(args.run.resolve() / "report.html")


if __name__ == "__main__":
    main()
