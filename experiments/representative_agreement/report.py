"""Write an offline Korean report of independently selected representatives."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


Record = dict[str, Any]
DEFAULT_RUN = Path(__file__).resolve().parents[2] / "runs/representative-agreement-2026-10-04"
DIRECTIONS = ("image_to_text", "text_to_image")
DIRECTION_NAMES = {
    "image_to_text": "이미지 대표가 텍스트의 상위 후보에 포함됨",
    "text_to_image": "텍스트 대표가 이미지의 상위 후보에 포함됨",
}


def _read(path: Path) -> Record:
    return json.loads(path.read_text(encoding="utf-8"))


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _metric(record: Record) -> Record:
    for key in ("agreement", "metrics", "evaluation"):
        if isinstance(record.get(key), dict) and "per_category" in record[key]:
            return record[key]
    return record


def _method(record: Record) -> str:
    value = record.get("method", record.get("key", record.get("model", "")))
    if isinstance(value, dict):
        value = value.get("key", value.get("name", ""))
    return str(value)


def _population(record: Record) -> str:
    value = record.get("population", "tune")
    return str(value.get("key", value.get("name", "tune"))) if isinstance(value, dict) else str(value)


def _label(record: Record) -> str:
    key = _method(record)
    if "sparse" in key:
        support = record.get("support", record.get("k"))
        if support is None:
            numbers = [int(part) for part in key.replace("-", "_").split("_") if part.isdigit()]
            support = next((number for number in numbers if number in (4, 8, 16)), "제한한")
        return f"Sparse CCA, 좌표마다 양쪽 특징을 각각 최대 {support}개 사용"
    if "hungarian" in key:
        return "헝가리안, activation을 하나씩 연결"
    if "cca" in key:
        return f'CCA, 공통 좌표 {_metric(record)["n_coordinates"]}개 사용'
    return str(record.get("label", key))


def _subject(record: Record) -> str:
    key = _method(record)
    if "hungarian" in key:
        return "헝가리안에서는"
    if "sparse" in key:
        support = next((number for number in (4, 8, 16) if str(number) in key.split("_")), None)
        return f"가중합마다 양쪽 특징을 각각 최대 {support}개 사용한 Sparse CCA에서는"
    return f'공통 좌표 {_metric(record)["n_coordinates"]}개를 사용한 CCA에서는'


def _is_candidate_control(record: Record) -> bool:
    return bool(record.get("candidate_control", False)) or (
        "cca" in _method(record) and "sparse" not in _method(record)
        and _metric(record)["n_coordinates"] > 256
    )


def _order(record: Record) -> tuple[int, int]:
    key = _method(record)
    if "hungarian" in key:
        return (0, 0)
    if "sparse" not in key:
        return (1, int(_metric(record)["n_coordinates"]))
    for support in (4, 8, 16):
        if str(support) in key.split("_"):
            return (2, support)
    return (2, 0)


def _percent(value: float | None, digits: int = 2) -> str:
    return "계산 불가" if value is None else f"{100 * value:.{digits}f}%"


def _count(count: int, total: int, bold: bool = False) -> str:
    value = f"{count}/{total}개<br><span class=rate>{_percent(count / total if total else None)}</span>"
    return f"<strong>{value}</strong>" if bold else value


def _rate_table(records: list[Record]) -> str:
    if not records:
        return "<p>이 조건은 계산되지 않았다.</p>"
    best = {
        (direction, k): max(
            (float(_metric(record)["summary"][direction][f"top{k}"])
             for record in records if _metric(record)["summary"][direction][f"top{k}"] is not None),
            default=-1,
        ) for direction in DIRECTIONS for k in (1, 5, 10)
    }
    parts = ['<div class="table-scroll"><table><thead><tr><th rowspan="2">대응 방법</th>'
             '<th rowspan="2">유효 후보 좌표 수</th>'
             '<th colspan="3">이미지 대표를 텍스트 후보와 비교</th>'
             '<th colspan="3">텍스트 대표를 이미지 후보와 비교</th></tr><tr>']
    parts.extend(f"<th>상위 {k}개 안의 일치</th>" for _ in DIRECTIONS for k in (1, 5, 10))
    parts.append("</tr></thead><tbody>")
    for record in records:
        metric = _metric(record)
        parts.append(f'<tr><th>{_escape(_label(record))}</th><td>{metric["n_candidate_coordinates"]}개</td>')
        for direction in DIRECTIONS:
            entry = metric["summary"][direction]
            for k in (1, 5, 10):
                value = entry[f"top{k}"]
                is_best = value is not None and abs(value - best[direction, k]) < 1e-12
                parts.append(f'<td>{_count(entry[f"top{k}_count"], entry["n_categories"], is_best)}</td>')
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _controls_table(records: list[Record]) -> str:
    parts = ['<div class="table-scroll"><table><thead><tr><th>대응 방법</th>'
             '<th>이미지·텍스트 대표 일치</th><th>연결을 섞은 일치율<br>평균 / 95백분위수</th>'
             '<th>범주를 섞은 일치율<br>평균 / 95백분위수</th>'
             '<th>독립된 이미지 자료끼리 대표 일치</th>'
             '<th>독립된 텍스트 자료끼리 대표 일치</th></tr></thead><tbody>']
    for record in records:
        metric = _metric(record)
        control = metric["controls"]
        parts.append(f'<tr><th>{_escape(_label(record))}</th><td>'
                     f'{_count(metric["summary"]["agree_at1_count"], metric["n_evaluated_categories"])}</td>')
        for key in ("random_pairing", "category_label_shuffle"):
            entry = control[key]
            parts.append(f'<td>{_percent(entry["mean"])} / {_percent(entry["p95"])}</td>')
        for key in ("image_split_stability", "text_split_stability"):
            entry = control.get(key)
            parts.append("<td>" + (_count(entry["agree_at1_count"], entry["n_categories"])
                                   if entry else "계산하지 않음") + "</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _names(protocol: Record, records: list[Record]) -> dict[Any, str]:
    names = {}
    for entry in protocol.get("concepts", protocol.get("categories", [])):
        if isinstance(entry, dict):
            names[entry.get("id", entry.get("category_id"))] = entry.get("name", "")
    for record in records:
        for row in _metric(record)["per_category"]:
            name = row.get("name", row.get("category_name"))
            if name:
                names[row["category_id"]] = name
    return names


def _name(row: Record, names: dict[Any, str]) -> str:
    return names.get(row["category_id"], f'범주 {row["category_id"]}')


def _representative(row: Record, side: str, record: Record) -> str:
    coordinate = row[f"{side}_coordinate"]
    if coordinate is None:
        return "계산 불가"
    sign = "+" if row[f"{side}_sign"] == 1 else "−"
    if "hungarian" in _method(record):
        identities = record.get("coordinate_identities", [])
        feature = next((entry.get(side + "_feature") for entry in identities
                        if entry["coordinate"] == coordinate), None)
        if feature is not None:
            return f"원래 SAE 특징 {feature}번<br>연결쌍 {coordinate}번, {sign} 방향"
        return f"연결쌍 {coordinate}번, {sign} 방향"
    return f"가중합 좌표 {coordinate}번, {sign} 방향"


def _cases(records: list[Record], names: dict[Any, str]) -> str:
    parts = ['<div class="table-scroll"><table><thead><tr><th>대응 방법</th><th>범주</th>'
             '<th>이미지에서 고른 대표와 AUROC</th><th>텍스트에서 고른 대표와 AUROC</th>'
             '<th>좌표와 방향의 일치</th></tr></thead><tbody>']
    for record in records:
        rows = [row for row in _metric(record)["per_category"] if row["status"] == "ok"]
        examples = []
        for matched in (True, False):
            eligible = [row for row in rows if row["agree_at1"] == matched]
            if eligible:
                examples.append(max(eligible, key=lambda row: min(row["image_auc"], row["text_auc"])))
        for row in examples:
            parts.append(f'<tr><th>{_escape(_label(record))}</th><td>{_escape(_name(row, names))}</td>')
            for side in ("image", "text"):
                parts.append(f'<td>{_representative(row, side, record)}<br>AUROC {row[f"{side}_auc"]:.3f}</td>')
            parts.append(f'<td>{"일치함" if row["agree_at1"] else "일치하지 않음"}</td></tr>')
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _sharing(records: list[Record], names: dict[Any, str]) -> str:
    parts = ['<div class="table-scroll"><table><thead><tr><th>대응 방법</th><th>주석으로 대표를 고른 쪽</th>'
             '<th>평가 범주 / 서로 다른 좌표·방향 조합</th><th>대표를 공유하는 범주 수</th>'
             '<th>공유 사례</th></tr></thead><tbody>']
    for record in records:
        rows = [row for row in _metric(record)["per_category"] if row["status"] == "ok"]
        for side, side_name in (("image", "이미지"), ("text", "텍스트")):
            groups: dict[tuple[int, int], list[str]] = defaultdict(list)
            for row in rows:
                groups[(row[f"{side}_coordinate"], row[f"{side}_sign"])].append(_name(row, names))
            shared = sorted(((key, members) for key, members in groups.items() if len(members) > 1),
                            key=lambda item: (-len(item[1]), item[0]))
            shared_count = sum(len(members) for _, members in shared)
            examples = []
            for (coordinate, sign), members in shared[:3]:
                polarity = "+" if sign == 1 else "−"
                examples.append(f'좌표 {coordinate}번 {polarity} 방향에서 {_escape(", ".join(members))}')
            examples_text = "<br>".join(examples) if examples else "같은 좌표와 방향을 공유한 범주가 없음"
            parts.append(f'<tr><th>{_escape(_label(record))}</th><td>{side_name}</td>'
                         f'<td>{len(rows)}개 범주 / {len(groups)}개 대표</td>'
                         f'<td>{shared_count}/{len(rows)}개 범주<small>공유 대표 {len(shared)}개</small></td>'
                         f'<td class="left">{examples_text}</td></tr>')
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _export(output: Path, records: list[Record], names: dict[Any, str]) -> None:
    rows = []
    for record in records:
        for row in _metric(record)["per_category"]:
            rows.append({"population": _population(record), "method": _method(record),
                         "method_name": _label(record), "category_name": _name(row, names), **row})
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "representative_per_category.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _main_message(records: list[Record]) -> str:
    selected = []
    hungarian = next((record for record in records if "hungarian" in _method(record)), None)
    cca = next((record for record in records if "cca" in _method(record) and "sparse" not in _method(record)), None)
    sparse = [record for record in records if "sparse" in _method(record)]
    if hungarian is not None:
        selected.append(hungarian)
    if cca is not None:
        selected.append(cca)
    if sparse:
        selected.append(max(sparse, key=lambda record: _metric(record)["summary"]["agree_at1"] or 0))
    summaries = []
    for record in selected:
        metric = _metric(record)
        count = metric["summary"]["agree_at1_count"]
        total = metric["n_evaluated_categories"]
        summaries.append(f'{_escape(_subject(record))} {total}개 범주 중 <strong>{count}개 '
                         f'({_percent(count / total if total else None)})</strong>에서 일치했다.')
    return " ".join(summaries)


def _secondary_message(primary: list[Record], all_records: list[Record], primary_name: str) -> str:
    sparse = [record for record in primary if "sparse" in _method(record)]
    if not sparse:
        return ""
    best = max(sparse, key=lambda record: _metric(record)["summary"]["agree_at1"] or 0)
    secondary = [record for record in all_records if _population(record) != primary_name
                 and (_method(record) == _method(best) or _method(record) == "cca_256")]
    if not secondary:
        return ""
    sentences = []
    for record in sorted(secondary, key=_order):
        metric = _metric(record)
        count = metric["summary"]["agree_at1_count"]
        total = metric["n_evaluated_categories"]
        sentences.append(f'{_escape(_subject(record))} 대표가 일치한 범주가 {count}/{total}개 '
                         f'({_percent(count / total if total else None)})였다')
    return ('<p><strong>주 분석에서 가장 높았던 희소 조건도 검증 자료에서 따로 확인했다.</strong> '
            + ". ".join(sentences)
            + '. 자료별 표본 수와 평가 가능한 범주가 달라 이 차이를 성능의 일반화 저하로만 해석할 수는 없다.</p>')


def _same_categories_table(primary: list[Record], secondary: list[Record]) -> str:
    by_method = {_method(record): record for record in primary}
    parts = ['<div class="table-scroll"><table><thead><tr><th>대응 방법</th>'
             '<th>동일 범주만 남긴 주 분석의 대표 일치</th><th>동일 범주의 검증 자료 대표 일치</th>'
             '</tr></thead><tbody>']
    for record in secondary:
        source = by_method.get(_method(record))
        if source is None:
            continue
        valid = {row["category_id"] for row in _metric(record)["per_category"] if row["status"] == "ok"}
        source_rows = [row for row in _metric(source)["per_category"]
                       if row["status"] == "ok" and row["category_id"] in valid]
        if {row["category_id"] for row in source_rows} != valid:
            raise ValueError("Secondary categories are not all present in the primary population")
        count = sum(bool(row["agree_at1"]) for row in source_rows)
        target = _metric(record)["summary"]["agree_at1_count"]
        parts.append(f'<tr><th>{_escape(_label(record))}</th><td>{_count(count, len(valid))}</td>'
                     f'<td>{_count(target, len(valid))}</td></tr>')
    parts.append("</tbody></table></div>")
    return "".join(parts)


def _validate(records: list[Record]) -> None:
    by_population: dict[str, set[Any]] = {}
    for record in records:
        metric = _metric(record)
        rows = [row for row in metric["per_category"] if row["status"] == "ok"]
        ids = {row["category_id"] for row in rows}
        population = _population(record)
        if population in by_population and by_population[population] != ids:
            raise ValueError(f"Compared methods have different category denominators in {population}")
        by_population[population] = ids
        if len(rows) != metric["n_evaluated_categories"]:
            raise ValueError("Saved category denominator does not match evaluated rows")
        strict = sum(bool(row["agree_at1"]) for row in rows)
        if strict != metric["summary"]["agree_at1_count"]:
            raise ValueError("Saved representative agreement count does not match category rows")
        for direction, rank_key in (("image_to_text", "rank_in_text"), ("text_to_image", "rank_in_image")):
            entry = metric["summary"][direction]
            for k in (1, 5, 10):
                count = sum(row[rank_key] is not None and row[rank_key] <= k for row in rows)
                if count != entry[f"top{k}_count"] or entry["n_categories"] != len(rows):
                    raise ValueError("Saved top-k count differs from category ranks")


STYLE = """
:root{color-scheme:light;--ink:#19323a;--muted:#566b73;--line:#dce5e8;--accent:#176b64}
*{box-sizing:border-box}body{margin:0;background:#f4f7f7;color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,
'Apple SD Gothic Neo','Noto Sans KR',sans-serif;font-size:16px;line-height:1.75}
main{max-width:1320px;margin:0 auto;padding:50px 32px 80px}header{margin-bottom:32px}h1{font-size:34px;line-height:1.4;
letter-spacing:-.8px;margin:8px 0 18px}h2{font-size:24px;line-height:1.5;margin:0 0 16px}h3{font-size:18px;margin:24px 0 10px}
p{margin:12px 0}section{background:white;border:1px solid var(--line);border-radius:14px;padding:28px;margin:22px 0}
.eyebrow{color:var(--accent);font-weight:700;font-size:14px}.lead{font-size:19px;line-height:1.8}.note{color:var(--muted);
font-size:14px}.callout{background:#edf6f3;border-left:4px solid var(--accent);padding:16px 20px;margin:18px 0}
.table-scroll{overflow-x:auto;margin:18px 0}table{width:100%;border-collapse:collapse;font-size:14px;line-height:1.5}
th,td{padding:12px 10px;border-bottom:1px solid var(--line);text-align:center;vertical-align:middle;min-width:90px}
thead th{background:#eef3f4;font-weight:700}tbody th{text-align:left;min-width:210px;font-weight:500}strong{font-weight:750}
td.left{text-align:left;min-width:210px}.rate{white-space:nowrap}small{display:block;color:var(--muted);font-size:12px}
details{border-top:1px solid var(--line);padding-top:16px;margin-top:18px}summary{cursor:pointer;font-weight:650}
ul,ol{padding-left:24px}li{margin:8px 0}.file{word-break:break-all;color:var(--muted);font-size:13px}code{font-size:.9em}
@media(max-width:700px){main{padding:24px 14px 50px}section{padding:20px 16px}h1{font-size:27px}h2{font-size:21px}}
"""


def build_report(output: Path) -> Path:
    """Build the report using completed results without changing their values."""
    output = output.resolve()
    records = []
    for path in sorted((output / "results").glob("*.json")):
        record = _read(path)
        if "per_category" not in _metric(record):
            continue
        if "__" in path.stem:
            population, method = path.stem.split("__", 1)
            record.setdefault("population", population)
            record.setdefault("method", method)
        records.append(record)
    if not records:
        raise FileNotFoundError(f"No completed representative evaluations in {output / 'results'}")
    _validate(records)
    protocol = _read(output / "protocol.json") if (output / "protocol.json").exists() else {}
    names = _names(protocol, records)
    _export(output, records, names)
    populations = list(dict.fromkeys(_population(record) for record in records))
    primary_name = next((name for name in populations if "tune" in name or "train" in name), populations[0])
    primary = sorted((record for record in records if _population(record) == primary_name), key=_order)
    main = [record for record in primary if not _is_candidate_control(record)]
    control = [record for record in primary if _is_candidate_control(record) or "hungarian" in _method(record)]
    evaluated = _metric(main[0])["n_evaluated_categories"]
    excluded_rows = [row for row in _metric(main[0])["per_category"] if row["status"] != "ok"]
    exclusions = ", ".join(_name(row, names) for row in excluded_rows) or "없음"
    sections = [f'<section><h2>양쪽 주석으로 따로 고른 대표가 일치하는지 확인했다</h2>'
                f'<p class="lead">{_main_message(main)}</p>'
                + _secondary_message(main, records, primary_name) +
                '<div class="callout"><strong>이 비율은 좌표 번호와 점수 방향이 모두 같은 범주의 비율이다.</strong> '
                'CCA와 Sparse CCA의 좌표는 여러 activation의 가중합이다. 그 가중합 좌표 중에서 범주마다 대표 하나를 고른다. '
                '검색 Recall이나 상대 모달리티의 AUROC를 대응 정확도로 바꿔 부른 값이 아니다.</div>'
                '</section>']
    sections.append('<section><h2>공통 물체 범주에서 대표 좌표와 상대 후보 순위를 비교했다</h2>'
                    f'<p>COCO 물체 80개 중 같은 표본 조건을 만족한 {evaluated}개 범주를 모든 방법에서 평가했다. '
                    '표의 굵은 값은 같은 열에서 가장 높은 값이다.</p>' + _rate_table(main) +
                    '<p class="note">상위 1개 안의 일치는 양쪽 대표가 같다는 뜻이므로 두 방향의 값이 같다. '
                    '상위 5개·10개 안의 일치는 한쪽 대표가 상대 모달리티의 주석 기반 후보 순위 안에 같은 방향으로 '
                    '포함되는지를 나타낸다. 이미지·캡션 검색의 Recall@5·10이 아니다.</p>'
                    f'<p class="note">공통 평가에서 제외한 범주는 {_escape(exclusions)}이다.</p></section>')
    sections.append('<section><h2>연결을 섞은 결과와 같은 모달리티의 재선정 결과를 함께 확인했다</h2>'
                    '<p>대표 일치가 우연한 연결보다 얼마나 높은지, 같은 모달리티에서도 표본을 바꾸면 대표가 달라지는지 '
                    '비교했다. 같은 모달리티의 재선정 일치율은 선택의 안정성을 나타내며 이론적인 성능 상한은 아니다.</p>' +
                    _controls_table(main) + '<p class="note">무작위 연결은 좌표의 연결만 섞고, 범주 섞기는 상대쪽 범주 이름을 '
                    '섞는다. 두 조건 모두 이미 선택한 대표와 방향을 유지한다. 95백분위수는 무작위 반복 결과의 '
                    '95%가 그 이하였다는 뜻이다.</p></section>')
    sections.append('<section><h2>각자 개념을 잘 구별해도 대표 좌표는 다를 수 있다</h2>'
                    '<p>아래는 실제 계산에서 일치한 범주와 일치하지 않은 범주를 방법마다 하나씩 고른 사례다. '
                    '각 집단에서 이미지·텍스트 AUROC 중 작은 값이 가장 큰 범주를 골랐다. '
                    '따라서 개념 구별 성능과 대표 번호의 일치를 구분해서 볼 수 있다.</p>' + _cases(main, names) +
                    '<p class="note">헝가리안은 각 모달리티의 원래 SAE 특징 번호와 그 특징이 속한 연결쌍 번호를 함께 보인다. '
                    '원래 특징 번호가 달라도 같은 연결쌍에 속하면 대응한 것으로 센다. '
                    'CCA의 가중합 좌표 번호는 원래 SAE 특징 번호와 다르며 0번부터 시작한다. '
                    '+ 방향은 점수가 클수록 범주가 있음을, '
                    '− 방향은 점수가 작을수록 범주가 있음을 뜻한다.</p></section>')
    sections.append('<section><h2>대표 일치가 개념 하나에 고유한 좌표를 뜻하지는 않는다</h2>'
                    '<p>여러 물체가 같은 좌표와 같은 방향을 대표로 고를 수 있다. 이런 공유가 양쪽에서 반복되면 '
                    '대표 일치율이 높더라도 물체들을 별도로 표현한다고 볼 수 없다.</p>' + _sharing(main, names) +
                    '<p class="note">대표를 공유하는 범주 수는 두 범주 이상이 고른 대표에 속한 모든 범주를 센 값이다. '
                    '같은 좌표의 반대 방향은 다른 대표로 센다. 공유 사례는 공유 범주 수가 많은 순서로 최대 세 집단을 보인다.</p></section>')
    if len(control) > 1:
        sections.append('<section><h2>CCA의 후보 좌표 수를 늘린 보조 조건도 비교했다</h2>'
                        '<p>후보 좌표 수가 다르면 대표를 고를 기회와 무작위 일치 확률이 달라진다. '
                        '헝가리안의 원래 연결 수 437개에 맞춰 CCA도 437개 좌표를 사용하는 조건을 계산했다. '
                        '상수 좌표를 제외한 뒤에는 유효 후보 수가 정확히 같지 않다. 실제 유효 좌표 수는 아래에 표시했다.</p>' +
                        _rate_table(control) + '</section>')
    positive = [{**record, **record["positive_only_sensitivity"]} for record in main
                if "positive_only_sensitivity" in record]
    if positive:
        sections.append('<section><h2>점수가 큰 방향만 허용한 조건도 확인했다</h2>'
                        '<p>주 분석은 모든 방법에서 점수가 커지는 방향과 작아지는 방향을 함께 고려했다. '
                        '아래에서는 모든 방법에 동일하게 양의 방향만 허용하고 원래 AUROC로 대표를 골랐다. '
                        '가중합을 포함한 각 후보의 값이 커지는 방향만 사용했다. '
                        'CCA 가중합 자체의 계수에는 여전히 음수가 포함될 수 있다.</p>' +
                        _rate_table(positive) + '</section>')
    for population in populations:
        if population == primary_name:
            continue
        secondary = sorted((record for record in records if _population(record) == population
                            and not _is_candidate_control(record)), key=_order)
        sections.append('<section><h2>COCO 검증 자료에서도 같은 계산을 반복했다</h2>'
                        '<p>이 표는 주 분석과 분리한 COCO val2017 이미지 5,000장과 해당 캡션을 사용했다. '
                        '표본 수가 달라 공통 평가 가능한 범주 수가 주 분석과 다를 수 있다. '
                        '이 검증 자료는 이전 분석에서 이미 확인한 자료다.</p>' + _rate_table(secondary) +
                        '<h3>평가 범주를 같게 맞춰도 자료에 따라 수치가 달랐다</h3>'
                        '<p>아래는 검증 자료에서 평가 가능한 범주만 주 분석에서도 남긴 비교다. '
                        '대표는 각 자료에서 원래 고른 값을 유지했다. 이 조건에서도 같은 희소 설정이 항상 가장 높지는 않았다. '
                        '특징 16개가 최적이라고 확정하기보다 표본 분할에 대한 안정성을 더 확인해야 한다.</p>' +
                        _same_categories_table(main, secondary) + '</section>')
    sections.append('<section><h2>실험의 조건과 해석 범위를 명시한다</h2><ol>'
                    '<li>기존 COCO SAE와 대응 계수를 고정했다. CCA와 Sparse CCA의 학습에는 범주 주석을 쓰지 않았다. '
                    '주석은 평가할 대표 좌표를 고르고 두 선택이 일치하는지 측정하는 데 사용했다.</li>'
                    '<li>주 분석에는 대응 학습에서 제외한 COCO train2017 이미지 23,657장을 사용했다. 이미지 ID를 '
                    '기준으로 독립된 두 부분으로 나누고, 한 부분의 이미지와 다른 부분의 텍스트에서 대표를 골랐다. '
                    '같은 이미지의 캡션은 모두 같은 부분에 속한다. SAE 자체는 COCO train2017로 학습했으므로 '
                    '주 분석 자료가 SAE 학습에서도 제외된 것은 아니다.</li>'
                    '<li>물체의 중앙 자르기 영역 주석을 이미지 정답으로 사용하고, 그 이미지의 정답을 캡션에도 부여했다. '
                    '따라서 텍스트에서 캡션에 직접 언급된 물체만 평가한 것은 아니다.</li>'
                    '<li>AUROC는 물체가 있는 표본의 점수가 없는 표본보다 높을 확률이며 동점은 절반을 센다. '
                    '대표 선정에서는 양·음 방향을 비교했다. 최종 지표는 선택한 좌표와 방향의 일치 여부다.</li>'
                    '<li>양쪽에서 상수가 아닌 좌표를 사용했다. 원래 SAE activation의 활성 빈도 5% 기준은 적용하지 않았다. '
                    'COCO 물체 80개 중 이미지 A 부분의 양성이 50개 이상이고 텍스트 B 부분의 양성 캡션이 50개 이상이며 '
                    '각각 음성도 존재하는 공통 범주를 평가했다. 표본 수를 원본 결과에 저장했다.</li>'
                    '<li>이 실험은 COCO로 학습한 방법의 비교다. CC3M으로 학습한 이전 논문의 주석 평가를 그대로 재현한 '
                    '결과가 아니다. 대표 일치는 개념이 한 좌표에만 존재한다는 증거도, 개념 자체에만 반응한다는 증거도 아니다.</li>'
                    '</ol><p class="file">범주별 전체 수치는 '
                    f'{_escape(output / "representative_per_category.csv")}에 저장했다. 원본 결과는 '
                    f'{_escape(output / "results")}에 저장했다.</p></section>')
    document = '<!doctype html><html lang="ko"><head><meta charset="utf-8">' \
        '<meta name="viewport" content="width=device-width,initial-scale=1">' \
        f'<title>주석으로 고른 대표 좌표의 모달리티 간 일치</title><style>{STYLE}</style></head><body><main>' \
        '<header><div class="eyebrow">고정한 대응 방법의 주석 기반 평가</div>' \
        '<h1>이미지와 텍스트에서 같은 개념의 대표를 따로 고르면 같은 좌표가 나오는가?</h1>' \
        '<p class="lead">주석을 사용하지 않고 학습한 대응이 범주별 대표 좌표도 연결하는지 확인했다.</p></header>' \
        + "".join(sections) + '</main></body></html>'
    destination = output / "report.html"
    destination.write_text(document, encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    print(build_report(args.run))


if __name__ == "__main__":
    main()
