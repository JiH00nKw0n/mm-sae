"""Report each controlled contrast separately, without additive attribution."""
from __future__ import annotations

import argparse
import html
import itertools
import json
from pathlib import Path

from mm_sae.io import write_csv


def load_results(root):
    return [json.loads(p.read_text()) for p in sorted(root.glob("*__center_*.json"))]


def contrasts(rows):
    fields = ("support", "fit_corpus", "scale_source", "center_source", "method")
    lookup = {tuple(row[k] for k in fields): row for row in rows}
    result = []
    for factor in ("fit_corpus", "scale_source", "center_source"):
        other = [k for k in fields if k != factor]
        combinations = {tuple(r[k] for k in other) for r in rows if r["support"] == "common"}
        for values in sorted(combinations):
            condition = dict(zip(other, values, strict=True))
            left = lookup.get(tuple(({**condition, factor: "cc3m"})[k] for k in fields))
            right = lookup.get(tuple(({**condition, factor: "coco"})[k] for k in fields))
            if left is None or right is None:
                continue
            for direction, k in itertools.product(("image_to_text", "text_to_image"), (1, 5, 10)):
                a = left["retrieval"][direction]["recall"][str(k)]
                b = right["retrieval"][direction]["recall"][str(k)]
                result.append(dict(factor=factor, **condition, direction=direction, recall_k=k,
                                   cc3m_reference=a, coco_reference=b, difference_pp=100 * (b - a)))
    return result


def table(headers, rows):
    return ("<table><thead><tr>" + "".join(f"<th>{html.escape(str(x))}</th>" for x in headers)
            + "</tr></thead><tbody>" + "".join("<tr>" + "".join(
                f"<td>{html.escape(str(x))}</td>" for x in row) + "</tr>" for row in rows)
            + "</tbody></table>")


def run(root):
    rows = load_results(root)
    summary = []
    for r in rows:
        for direction, score in r["retrieval"].items():
            summary.append({**{k: r[k] for k in ("support", "fit_corpus", "scale_source", "center_source", "method")},
                            "direction": direction,
                            **{f"recall_at_{k}": score["recall"][str(k)] for k in (1, 5, 10)}})
    if summary:
        write_csv(root / "retrieval_summary.csv", summary)
    changes = contrasts(rows)
    if changes:
        write_csv(root / "controlled-differences.csv", changes)
    corpus_names = {"cc3m": "CC3M 학습 자료", "coco": "COCO 학습 자료"}
    factor_names = {"fit_corpus": "대응 학습 자료", "scale_source": "표준편차 출처", "center_source": "검색 평균 출처"}
    directions = {"image_to_text": "이미지로 텍스트 검색", "text_to_image": "텍스트로 이미지 검색"}
    fields = ("support", "fit_corpus", "scale_source", "center_source", "method")
    lookup = {tuple(r[k] for k in fields): r for r in rows}
    sections = [f"<h1>대응 학습 자료와 정규화 효과를 분리한 비교</h1><p>30개 조건 중 {len(rows)}개를 계산했습니다.</p>",
                "<p>CC3M SAE와 COCO 평가 표본을 고정했습니다. 공통 특징 조건은 두 학습 자료에서 "
                "모두 분산이 있는 같은 SAE 좌표만 사용합니다. 평가 자료의 통계로 학습하지 않았습니다. "
                "모든 CCA는 대응 학습 자료 자체의 평균을 빼고 학습했습니다. 검색 평균의 교체는 "
                "대응을 고정한 상태에서만 적용했습니다.</p>"]
    sections.append("<h2>Cross-SVD는 여러 activation의 가중합을 양쪽에 만들어 대응시킵니다</h2>"
                    "<p>이미지·텍스트 사이의 공분산 행렬 C를 UΣVᵀ로 분해하고, 이미지에는 U의 첫 256개 열을, "
                    "텍스트에는 V의 첫 256개 열을 곱합니다. 각 열은 여러 SAE activation을 합치는 계수입니다. "
                    "원래 SAE 특징 중 256개를 고르는 방법이 아닙니다. 계수는 음수일 수도 있으며 대부분의 특징이 조합에 참여할 수 있습니다. "
                    "기존 기준 조건처럼 각 입력 좌표를 자체 학습 표준편차로 나눴다면 C는 coactivation correlation 행렬입니다. "
                    "다른 자료의 표준편차를 적용한 교차 조건에서는 일반적인 공분산 행렬로 해석해야 합니다.</p>"
                    "<p>Cross-SVD는 이 계산을 구분하기 위한 코드상의 이름이며, "
                    "<a href='https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSSVD.html'>PLS-SVD</a>와 같은 형태의 분해를 사용합니다. "
                    "CCA는 모달리티 내부의 분산과 특징 간 상관까지 추가로 보정한 뒤 조합을 찾습니다. "
                    "어느 방법도 공통 좌표 하나가 특정 주석 개념 하나를 나타낸다고 보장하지 않습니다.</p>")
    controlled = [
        ("native", "cc3m", "cc3m", "cc3m", "CCA의 기존 CC3M 조건"),
        ("common", "cc3m", "cc3m", "cc3m", "공통 특징만 사용하고 나머지는 기존 CC3M 조건을 유지"),
        ("common", "cc3m", "cc3m", "coco", "공통 특징과 CC3M 대응을 유지하고 검색 평균만 COCO 것으로 교체"),
        ("common", "cc3m", "coco", "cc3m", "공통 특징과 CC3M 학습 자료를 유지하고 표준편차만 COCO 것으로 교체"),
        ("common", "coco", "cc3m", "cc3m", "공통 특징과 CC3M 평균·표준편차를 유지하고 대응만 COCO에서 학습"),
    ]
    main_rows = []
    for support, corpus, scale, center, label in controlled:
        r = lookup.get((support, corpus, scale, center, "cca"))
        if r is not None:
            main_rows.append([label, *(f'{100*r["retrieval"][d]["recall"]["10"]:.2f}%'
                                       for d in directions)])
    sections += ["<h2>CCA의 큰 격차는 특징 목록이나 정규화만으로 설명되지 않았습니다</h2>",
                 "<p>공통 특징 번호와 정규화 통계를 고정해도 대응을 COCO에서 학습한 경우의 검색 성능이 크게 높았습니다. "
                 "이 비교에서 바뀐 것은 대응 학습 자료입니다. 다만 학습 자료의 내용, 표본 수, 이미지당 캡션 수까지 따로 분리한 결과는 아닙니다.</p>",
                 table(["CCA 비교 조건", "이미지로 텍스트 검색 Recall@10", "텍스트로 이미지 검색 Recall@10"], main_rows),
                 "<p><a href='retrieval_summary.csv'>모든 조건의 Recall@1·5·10</a> · "
                 "<a href='controlled-differences.csv'>다른 조건을 고정한 요인별 성능 차이</a></p>"]
    for factor, title in (("fit_corpus", "특징과 정규화를 고정하고 대응 학습 자료만 바꿨습니다"),
                          ("scale_source", "특징과 학습 자료를 고정하고 표준편차를 바꿨습니다"),
                          ("center_source", "학습한 대응을 고정하고 검색에 쓰는 평균만 바꿨습니다")):
        subset = [r for r in changes if r["factor"] == factor and r["recall_k"] == 10]
        sections.append(f"<h2>{title}</h2>")
        sections.append(table(["방법", "고정한 나머지 조건", "검색 방향", "CC3M 통계 또는 학습 자료", "COCO 통계 또는 학습 자료", "차이(%p)"], [
            [r["method"], ", ".join(f"{factor_names[k]}는 {corpus_names[r[k]]}" for k in factor_names if k != factor),
             directions[r["direction"]], f'{100*r["cc3m_reference"]:.2f}%',
             f'{100*r["coco_reference"]:.2f}%', f'{r["difference_pp"]:+.2f}'] for r in subset]))
    feature_rows = []
    for corpus, method, direction in itertools.product(corpus_names, ("cca", "cross_svd", "procrustes"), directions):
        native = lookup.get(("native", corpus, corpus, corpus, method))
        common = lookup.get(("common", corpus, corpus, corpus, method))
        if native is not None and common is not None:
            a, b = (r["retrieval"][direction]["recall"]["10"] for r in (native, common))
            feature_rows.append([corpus_names[corpus], method, directions[direction], f"{100*a:.2f}%", f"{100*b:.2f}%", f"{100*(b-a):+.2f}"])
    sections += ["<h2>학습 자료와 정규화를 고정하고 공통 특징만 사용했습니다</h2>",
                 table(["대응 학습 자료", "방법", "검색 방향", "기존 특징", "공통 특징", "차이(%p)"], feature_rows)]
    margin_rows = []
    for r in rows:
        if r["support"] != "native":
            continue
        for direction, d in r["diagnostics"].items():
            margin_rows.append([corpus_names[r["fit_corpus"]], r["method"], directions[direction],
                                f'{d["best_positive_mean"]:.4f}', f'{d["hardest_negative_mean"]:.4f}',
                                f'{d["margin_mean"]:.4f}', f'{d["average_negative_mean"]:.4f}'])
    sections += ["<h2>정답과 가장 높은 오답의 점수를 함께 비교했습니다</h2>",
                 "<p>이미지로 텍스트를 검색할 때는 정답 캡션 중 최고점을 사용하고, 같은 이미지의 모든 캡션을 오답에서 제외했습니다. "
                 "텍스트로 이미지를 검색할 때는 해당 캡션의 원본 이미지가 정답입니다. 점수는 코사인 유사도입니다.</p>",
                 table(["대응 학습 자료", "방법", "검색 방향", "정답 점수 평균", "최고 오답 점수 평균", "정답에서 최고 오답을 뺀 값", "전체 오답 점수 평균"], margin_rows),
                 "<p>요인의 효과는 다른 조건에 따라 달라질 수 있습니다. 위 차이들을 더해 전체 격차를 설명할 수는 없습니다. "
                 "공통 특징과 정규화를 고정한 학습 자료 비교에도 표본 수와 이미지당 캡션 수의 차이가 포함됩니다. "
                 "공통 좌표별 정답·최고 오답 점수 기여도는 각 조건의 contributions.csv에 저장했습니다. "
                 "이 기여도는 좌표의 의미에 대한 주석이 아닙니다.</p>"]
    weighted_rows = []
    for corpus, method in itertools.product(corpus_names, ("cca", "cross_svd")):
        original = lookup.get(("native", corpus, corpus, corpus, method))
        if original is not None:
            for direction, score in original["retrieval"].items():
                weighted_rows.append([corpus_names[corpus], method, "기존 공통 좌표의 크기를 유지", directions[direction],
                                      *(f'{100*score["recall"][str(k)]:.2f}%' for k in (1, 5, 10))])
    for p in sorted((root / "output-weighting").glob("*__unit*.json")):
        r = json.loads(p.read_text())
        description = ("학습 자료에서 각 공통 좌표의 분산을 1로 맞춤" if r["weighting"] == "unit_training_variance"
                       else "공통 좌표의 분산을 맞추고 학습 상관의 제곱근을 곱함")
        for direction, score in r["retrieval"].items():
            weighted_rows.append([corpus_names[r["corpus"]], r["method"], description, directions[direction],
                                  *(f'{100*score["recall"][str(k)]:.2f}%' for k in (1, 5, 10))])
    if weighted_rows:
        weighted_rows.sort(key=lambda r: (r[0], r[1], r[3], r[2]))
        sections += ["<h2>공통 방향을 고정한 채 출력 좌표의 크기를 바꿨습니다</h2>",
                     "<p>기존 방법의 조합 계수 방향을 유지합니다. 방향을 다시 학습하거나 평가 자료의 통계로 크기를 정하지 않습니다. "
                     "분산을 1로 맞추면 원래 큰 값을 가지던 공통 좌표의 상대적 비중을 줄입니다. "
                     "그다음 학습 상관의 제곱근을 양쪽에 곱하면, 내적에서 각 좌표의 항에 학습 상관만큼 비중을 줍니다. 코사인 유사도의 분모도 함께 바뀝니다.</p>",
                     table(["대응 학습 자료", "방법", "출력 크기 조건", "검색 방향", "Recall@1", "Recall@5", "Recall@10"], weighted_rows)]
    (root / "report.html").write_text("<!doctype html><meta charset='utf-8'><title>대응 학습 자료 분리 비교</title>"
                                     "<style>body{font:17px/1.65 sans-serif;max-width:1350px;margin:40px auto;padding:20px}"
                                     "table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ddd;padding:8px}"
                                     "h2{margin-top:44px}</style>" + "".join(sections))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    run(parser.parse_args().run)
