"""Summarize completed cross-corpus controls with traceable, verified retrieval values."""
from __future__ import annotations

import argparse
import csv
from html import escape
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from experiments.corpus_comparison.report import paired_interval
from mm_sae.io import atomic_json, sha256, write_csv

DIRECTIONS = ("image_to_text", "text_to_image")
FIT_NAMES = {"cc3m-fit": "CC3M에서 대응 학습", "coco-fit": "COCO에서 대응 학습"}
ENGLISH_FIT_NAMES = {"cc3m-fit": "Mapping fitted on CC3M", "coco-fit": "Mapping fitted on COCO"}
K_VALUES = (4, 8, 16, 32)
EPS_VALUES = (0.003, 0.01, 0.03, 0.05, 0.1, 0.3)
SPACES = {"image": "텍스트를 이미지 좌표로 변환", "text": "이미지를 텍스트 좌표로 변환"}


class Evidence:
    def __init__(self):
        self.sources: dict[str, str] = {}
        self.checked: set[str] = set()
        self.rows: list[dict[str, Any]] = []

    def read(self, path: Path) -> Any:
        self.sources[str(path.resolve())] = sha256(path)
        return json.loads(path.read_text())

    def result(self, path: Path, label: str, group: str) -> dict[str, Any]:
        record = self.read(path)
        if str(path) not in self.checked:
            for direction, counts in zip(DIRECTIONS, ((5000, 25014), (25014, 5000)), strict=True):
                metric = record["retrieval"][direction]
                assert (metric["query_count"], metric["candidate_count"]) == counts, path
                ranks = np.asarray(metric["ranks"])
                assert len(ranks) == counts[0], path
                for k in (1, 5, 10):
                    np.testing.assert_allclose(metric["recall"][str(k)], np.mean(ranks <= k), atol=1e-12)
                self.rows.append(dict(source=str(path.resolve()), condition=group, method=label,
                                      direction=direction, **{f"recall_at_{k}": metric["recall"][str(k)]
                                                             for k in (1, 5, 10)}))
            self.checked.add(str(path))
        return dict(label=label, group=group, record=record)


def numbers(row):
    return [100 * row["record"]["retrieval"][d]["recall"][str(k)] for d in DIRECTIONS for k in (1, 5, 10)]


def retrieval_table(rows):
    maxima = {group: np.max([numbers(r) for r in rows if r["group"] == group], axis=0)
              for group in {r["group"] for r in rows}}
    body = []
    for r in rows:
        cells = []
        for j, value in enumerate(numbers(r)):
            text = f"{value:.2f}"
            if value == maxima[r["group"]][j]:
                text = f"<strong>{text}</strong>"
            cells.append(f"<td>{text}</td>")
        body.append(f'<tr><td>{escape(r["group"])}</td><th>{escape(r["label"])}</th>{"".join(cells)}</tr>')
    return ('<div class="scroll"><table><thead><tr><th rowspan="2">비교 조건</th><th rowspan="2">방법</th>'
            '<th colspan="3">이미지로 텍스트 검색 (%)</th><th colspan="3">텍스트로 이미지 검색 (%)</th></tr>'
            '<tr>' + ''.join(f'<th>Recall@{k}</th>' for _ in DIRECTIONS for k in (1, 5, 10))
            + '</tr></thead><tbody>' + ''.join(body) + '</tbody></table></div>')


def plain_table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{escape(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def section(number, title, result, contents, interpretation):
    return (f'<section id="finding-{number}"><h2>{number}. {title}</h2>'
            f'<p class="message">{result}</p>{contents}<p class="interpretation">{interpretation}</p></section>')


def ablation_result(ev, root, key, label, group):
    return ev.result(root / "results" / f"{key}.json", label, group)


def main_report(cfg):
    root = Path(cfg["run"])
    ev = Evidence()
    completed = ev.read(root / "completed-artifacts.json")
    assert completed["state"] == "completed" and completed["completed_jobs"] == 33
    missing = [name for name, state in completed["jobs"].items() if state != "completed"]
    assert not missing
    for relative, receipt in completed["files"].items():
        path = root / relative
        assert path.stat().st_size == receipt["size"], path
        assert sha256(path) == receipt["sha256"], path
    parents = np.load(cfg["evaluation_parents"])
    ev.sources[cfg["evaluation_parents"]] = sha256(Path(cfg["evaluation_parents"]))
    old = Path(cfg["coco_ablation"])
    native = Path(cfg["cc3m_ablation"])
    control = root / "coco-fit/ablation"
    populations = [ev.read(p / "population.json") for p in (old, native, control)]
    assert all(p["test_image_ids"] == populations[0]["test_image_ids"] for p in populations)
    parts = []

    corpus_rows = []
    studies = [(old, "COCO SAE·COCO 대응 학습"), (native, "CC3M SAE·CC3M 대응 학습"),
               (control, "CC3M SAE·COCO 대응 학습")]
    for folder, label in studies:
        for key, method in [("cca_256", "CCA, 공통 좌표 256개"), ("cross_svd_256", "Cross-SVD, 공통 좌표 256개")]:
            corpus_rows.append(ablation_result(ev, folder, key, method, label))
    intervals = []
    for name, a, b in [("CC3M SAE: COCO-fitted CCA minus CC3M-fitted CCA", corpus_rows[4], corpus_rows[2]),
                       ("CC3M-fitted: Cross-SVD minus CCA", corpus_rows[3], corpus_rows[2]),
                       ("COCO-fitted: Cross-SVD minus CCA", corpus_rows[5], corpus_rows[4])]:
        for direction in DIRECTIONS:
            for k in (1, 5, 10):
                intervals.append(dict(comparison=name, direction=direction, k=k,
                                      **paired_interval(a["record"]["retrieval"][direction]["ranks"],
                                                        b["record"]["retrieval"][direction]["ranks"], parents, direction, k)))
    diagnostic_rows = []
    for pre in FIT_NAMES:
        path = root / pre / "diagnostics/component-distribution.csv"
        ev.sources[str(path.resolve())] = sha256(path)
        with path.open(encoding="utf-8-sig") as stream:
            components = list(csv.DictReader(stream))
        means = [np.mean([float(r["correlation"]) for r in components
                          if r["method"] == "cca_256" and r["population"] == pop]) for pop in ("fit", "test")]
        distribution = ev.read(root / pre / "diagnostics/activation-distribution.json")
        diagnostic_rows.append([FIT_NAMES[pre], f"{means[0]:.3f}", f"{means[1]:.3f}",
                                f"{distribution['image']['test_to_fit_variance_ratio_quantiles'][2]:.3f}",
                                f"{distribution['text']['test_to_fit_variance_ratio_quantiles'][2]:.3f}"])
    parts.append(section(1, "대응 행렬을 학습한 자료에 따라 CCA 성능이 크게 달라졌다.",
        "CC3M SAE를 고정한 채 대응 학습 자료를 COCO로 바꾸자 CCA의 Recall@10이 "
        "이미지로 텍스트를 찾을 때 <strong>47.22%에서 61.28%</strong>, 텍스트로 이미지를 찾을 때 "
        "<strong>35.96%에서 52.41%</strong>로 올랐다.",
        '<p>CCA는 여러 activation의 가중합을 양쪽에서 학습해 서로 대응하는 공통 좌표를 만든다. '
        'Cross-SVD는 같은 표준화 입력에서 두 모달리티가 함께 변하는 방향을 찾되, 각 모달리티 내부의 특징 간 상관을 추가로 보정하지 않는다. '
        '아래 비교에서는 공통 좌표를 256개로 고정했다.</p>' + retrieval_table(corpus_rows)
        + '<details><summary>학습 자료와 평가 자료에서 activation 분포가 얼마나 달랐는지 확인한 결과</summary>'
        + plain_table(["대응 학습 자료", "학습 자료의 CCA 좌표 쌍 평균 상관", "평가 자료의 CCA 좌표 쌍 평균 상관",
                       "이미지 SAE 특징의 평가 분산 / 학습 분산 중앙값", "텍스트 SAE 특징의 평가 분산 / 학습 분산 중앙값"], diagnostic_rows)
        + '<p>CC3M에서 대응을 학습하면 평가 자료에서 SAE activation의 분산이 학습 때보다 작은 특징이 많았다. '
        '그러나 CCA의 대응 좌표 간 평균 상관은 0.724에서 0.723으로 거의 유지됐다. '
        '또 COCO에서 대응을 학습한 CCA는 평가 상관이 0.573으로 더 낮지만 검색은 더 잘했다. '
        '대응 좌표들의 평균 상관이 높다는 사실만으로 개별 이미지·캡션 쌍을 더 잘 구별한다고 판단할 수 없다. '
        '이 통계는 분포 차이를 관찰한 것이며, 검색 차이의 원인을 하나씩 제거한 실험은 아니다.</p></details>',
        '<strong>CC3M SAE가 COCO 검색에 필요한 정보를 제공하지 못한다는 설명은 이 결과와 맞지 않는다.</strong> '
        'COCO 학습 자료에서 대응을 다시 학습하면 더 높은 성능을 얻는다. 다만 학습 자료의 종류와 함께 '
        '이미지 수, 이미지당 캡션 수, 학습 분산으로 정한 사용 특징과 입력 보정값도 달라진다. 분포 차이 하나의 효과로 단정할 수 없다.'))

    preprocessing = []
    for folder, label in studies:
        for mode, method in [("raw", "원래 activation 사용"), ("centered", "학습 평균을 뺀 activation 사용"),
                             ("standardized", "학습 평균을 빼고 표준편차로 나눈 activation 사용")]:
            preprocessing.append(ablation_result(ev, folder, f"hungarian__{mode}__text_projected_to_image", method, label))
    parts.append(section(2, "평균 제거와 표준편차 보정의 이득은 학습 조건마다 달랐다.",
        "헝가리안 대응을 고정했을 때, CC3M에서 대응을 학습한 조건의 이미지로 텍스트 검색 Recall@10은 "
        "평균을 빼면 <strong>48.40%에서 46.24%</strong>로 낮아졌다. 표준편차까지 나누면 <strong>35.18%</strong>였다.",
        '<p>각 학습 조건 안에서 헝가리안 연결은 그대로 두고 검색에 입력하는 activation만 바꿨다. '
        '텍스트를 이미지 좌표로 변환한 동일한 검색 공간에서 비교했다.</p>' + retrieval_table(preprocessing),
        'COCO SAE에서 얻었던 “평균을 빼면 크게 좋아진다”는 결론을 모든 SAE에 적용할 수 없다. '
        '특히 이번 CC3M SAE에서는 표준편차로 나누는 조건이 헝가리안 검색 성능을 크게 낮췄다. '
        '이 사실만으로 CCA처럼 표준화된 입력으로 가중치를 학습하는 방법에서도 표준화를 제거해야 한다고 결론 내릴 수는 없다.'))

    baseline = []
    coverage = []
    exploratory = []
    for pre, folder in [("cc3m-fit", native), ("coco-fit", control)]:
        group = FIT_NAMES[pre]
        for key, label in [
            ("hungarian__standardized__text_projected_to_image", "헝가리안, 텍스트를 이미지 좌표로 변환"),
            ("sinkhorn__standardized__text_projected_to_image", "Sinkhorn, 텍스트를 이미지 좌표로 변환"),
            ("sinkhorn__standardized__image_projected_to_text", "Sinkhorn, 이미지를 텍스트 좌표로 변환"),
            ("cca_256", "CCA, 공통 좌표 256개"), ("cross_svd_256", "Cross-SVD, 공통 좌표 256개")]:
            baseline.append(ablation_result(ev, folder, key, label, group))
        baseline.append(ev.result(root / pre / "semantics/results/procrustes_full__image.json",
                                  "Procrustes, 텍스트를 이미지 좌표로 변환", group))
        for key, label in [("partial_many__standardized__text_projected_to_image", "연결 수를 제한한 다대다 대응"),
                           ("sparse_transport__standardized__text_projected_to_image", "희소한 양수 가중 대응"),
                           ("sparse_factorization__standardized__text_projected_to_image", "작은 특징 집합을 거쳐 텍스트를 이미지 좌표로 변환"),
                           ("groups__unit_variance", "작은 특징 집합 256개 사이에서 직접 검색")]:
            exploratory.append(ablation_result(ev, folder, key, label, group))
        for family, label in [("hungarian", "헝가리안"), ("sinkhorn", "Sinkhorn")]:
            for space, space_label in [("text_projected_to_image", "텍스트를 이미지 좌표로 변환"),
                                       ("image_projected_to_text", "이미지를 텍스트 좌표로 변환")]:
                coverage.append(ev.result(root / pre / "diagnostics/results" / f"{family}_standardized__{space}.json",
                                          f"{label}, {space_label}", group))
    parts.append(section(3, "여러 특징을 조합하는 방법은 유리했지만, 모든 가중 대응이 두 방향에서 좋지는 않았다.",
        "같은 평균·표준편차 보정을 적용해도 <strong>CCA와 Procrustes는 헝가리안보다 두 검색 방향 모두 좋았다.</strong> "
        "Sinkhorn은 어느 쪽 좌표로 변환하느냐에 따라 유리한 검색 방향이 달라졌다.",
        '<p>헝가리안은 특징을 하나씩 짝짓는다. Sinkhorn은 여러 특징에 양의 가중치를 나누어 준다. '
        'Procrustes는 양수와 음수 가중치를 사용하되 변환 행렬에 직교 제약을 둔다. 아래에서는 학습 평균과 표준편차로 보정한 입력을 공통으로 사용했다. '
        'Sinkhorn은 기존 비교에서 사용한 ε=0.05를 유지했다.</p>' + retrieval_table(baseline)
        + '<details><summary>헝가리안이 남긴 동일한 특징들로만 다시 비교한 결과</summary><p>헝가리안이 선택한 특징 번호로 '
        '두 방법의 입력과 연결 행렬을 제한했다. Sinkhorn을 다시 학습하거나 행렬의 가중치 합을 다시 맞추지는 않았다.</p>'
        + retrieval_table(coverage) + '</details>'
        + '<details><summary>앞서 시도한 다대다 방법 세 가지와 집합 간 직접 검색의 결과</summary>'
        '<p>연결 수 제한, 희소 가중 대응, 작은 특징 집합을 거치는 대응을 같은 표준화 입력으로 다시 비교했다. '
        '마지막 행은 작은 특징 집합의 출력 분산을 학습 자료로 맞춘 뒤 두 집합 표현끼리 직접 검색한 결과다. '
        '이 집합은 coactivation correlation 행렬을 분해해서 만들었으며, 주석 개념별로 학습한 집합과 다르다.</p>'
        + retrieval_table(exploratory) + '</details>',
        '따라서 “모든 소프트 방법이 더 좋다”보다 <strong>“양쪽 특징을 조합하는 공통 표현은 유망하지만, '
        '한쪽 activation을 다른 쪽으로 옮기는 방식에서는 검색 방향까지 분리해서 봐야 한다”</strong>가 정확하다. '
        '헝가리안과 동일한 특징만 남겨도 Sinkhorn의 방향별 차이가 남아, 사용 특징 수만으로 설명되지는 않았다.'))

    sparse_rows = []
    for key in ("cca_full", "sparse_cca_16"):
        ev.result(Path(cfg["coco_semantics"]) / "results" / f"{key}.json", key, "종전 COCO SAE·COCO 대응 학습")
    sparse_chart = {}
    for pre in FIT_NAMES:
        group = FIT_NAMES[pre]
        sparse_rows.append(ev.result(root / pre / "pruning/results/cca_full.json", "CCA, 계수 제거 없음", group))
        for k in K_VALUES:
            prune = ev.result(root / pre / "pruning/results" / f"cca_{k}.json", f"CCA 학습 후 좌표당 {k}개 계수만 유지", group)
            sparse = ev.result(root / pre / "semantics/results" / f"sparse_cca_{k}.json", f"Sparse CCA, 처음부터 좌표당 최대 {k}개 특징", group)
            sparse_rows.extend([prune, sparse])
            sparse_chart[(pre, "pruned", k)] = prune
            sparse_chart[(pre, "sparse", k)] = sparse
            meta = ev.read(root / pre / "semantics/sparse-fit" / f"sparse_cca_k{k}.json")
            assert meta["converged_components"] == meta["dimensions"] == 256
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained", sharey=True)
    for i, pre in enumerate(FIT_NAMES):
        for j, direction in enumerate(DIRECTIONS):
            ax = axes[i, j]
            for family, label, color in [("pruned", "CCA: prune after fitting", "#d27a76"),
                                          ("sparse", "Sparse CCA: fit with a limit", "#318399")]:
                ax.plot(K_VALUES, [100*sparse_chart[(pre,family,k)]["record"]["retrieval"][direction]["recall"]["10"]
                                   for k in K_VALUES], marker="o", color=color, label=label)
            full = next(r for r in sparse_rows if r["group"] == FIT_NAMES[pre] and r["label"] == "CCA, 계수 제거 없음")
            ax.axhline(100*full["record"]["retrieval"][direction]["recall"]["10"], color="#444", linestyle="--", label="Dense CCA")
            ax.set(title=ENGLISH_FIT_NAMES[pre]+" / "+("Image to text" if j == 0 else "Text to image"),
                   xlabel="Maximum SAE features per common coordinate", ylabel="Recall@10 (%)", xticks=K_VALUES, ylim=(0,70))
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.savefig(root / "sparse_comparison.png", dpi=170)
    plt.close(fig)
    parts.append(section(4, "희소하게 학습하는 방법이 학습 후 계수를 지우는 방법보다 좋았다.",
        "CC3M에서 대응을 학습하고 좌표당 특징을 16개로 제한했을 때, 이미지로 텍스트 검색 Recall@10은 "
        "CCA 계수 제거가 <strong>19.46%</strong>, Sparse CCA가 <strong>32.06%</strong>였다. "
        "COCO에서 대응을 학습하면 각각 <strong>33.92%와 49.82%</strong>였다.",
        '<p>두 방법 모두 공통 좌표 256개를 사용한다. CCA 계수 제거는 학습된 계수 중 절댓값이 큰 것만 남기고 값을 유지한다. '
        'Sparse CCA는 사용할 특징 수를 제한한 상태에서 특징 번호와 가중치를 함께 학습한다. 특징 16개는 전체에서 16개가 아니라 '
        '이미지·텍스트 각각의 공통 좌표 하나를 구성하는 최대 특징 수다.</p>'
        '<img src="sparse_comparison.png" alt="학습 후 계수 제거와 Sparse CCA의 특징 수별 검색 성능">'
        + retrieval_table(sparse_rows),
        '<strong>작은 특징 집합을 사용하려면 계수를 사후에 자르기보다 제한을 학습에 포함하는 방향이 더 유망하다.</strong> '
        '그러나 CC3M에서 대응을 학습한 Sparse CCA 16개 조건은 전체 CCA보다 Recall@10이 15.16%p·9.81%p 낮았다. '
        'COCO SAE에서 Sparse CCA 16개의 Recall@10은 46.90%·38.43%, 전체 CCA는 49.76%·40.74%였다. '
        '적은 특징으로 성능을 유지할 수 있는 정도는 학습 조건에 따라 다르며, 특징 수가 적다는 사실만으로 해석 가능성이 입증되지는 않는다.'))

    sink_rows = []
    for pre in FIT_NAMES:
        for epsilon in EPS_VALUES:
            for k in ["full", 1, 2, 4, 8, 16, 32, 64]:
                for space in ["text_projected_to_image", "image_projected_to_text"]:
                    side_label = "텍스트를 이미지 좌표로 변환" if space.startswith("text") else "이미지를 텍스트 좌표로 변환"
                    label = f"ε={epsilon}, " + ("연결 제거 없음" if k == "full" else f"특징당 연결 최대 {k}개") + ", " + side_label
                    row = ev.result(root / pre / "pruning/sinkhorn/results" / f"sinkhorn_e{epsilon}_k{k}__{space}.json", label, FIT_NAMES[pre])
                    assert row["record"]["metadata"]["solver"]["converged"]
                    sink_rows.append(row)
    sink_main = [r for r in sink_rows if r["record"]["metadata"]["epsilon"] == .05 and r["record"]["metadata"]["k"] in (None,16,32)]
    parts.append(section(5, "Sinkhorn의 작은 연결 제거는 도움이 되기도 했지만, 성능 변화는 방향마다 달랐다.",
        "CC3M에서 학습한 ε=0.05 Sinkhorn으로 텍스트를 이미지 좌표에 옮길 때, 연결을 특징당 최대 16개로 줄이면 "
        "Recall@10이 <strong>45.22%·26.81%에서 46.44%·33.44%</strong>로 올랐다. "
        "같은 조작을 COCO에서 대응을 학습한 조건에 적용하면 이미지로 텍스트 검색은 낮아지고 반대 방향은 높아졌다.",
        '<p>ε는 가중치를 넓게 분산시키는 정도를 조절한다. 작은 ε가 수치상 작은 가중치를 만들 수 있지만, '
        '연결 수의 명시적인 상한은 별도의 계수 제거로 적용했다. ε 여섯 값과 연결 수 여덟 조건을 모두 계산했다. '
        '최종 평가에서 가장 큰 수치가 나온 ε를 선택한 성능으로 제시하지 않고, 기존 ε=0.05의 변화를 본문에 둔다.</p>'
        + retrieval_table(sink_main)
        + '<details><summary>ε와 연결 수의 전체 192개 비교 조건</summary>' + retrieval_table(sink_rows) + '</details>',
        'Sinkhorn에서도 약한 연결을 줄이는 절충은 가능하다. 다만 한 조건이 두 방향에서 항상 우세하지 않았으며, '
        '현재 결과만으로 하나의 ε나 연결 수가 최적이라고 정할 수 없다.'))

    sign_rows = []
    for pre in FIT_NAMES:
        for k in (8,16):
            for sign, word in [("signed", "음수 허용"), ("nonnegative", "양수만 허용")]:
                for space in ("image", "text"):
                    sign_rows.append(ev.result(root / pre / "semantics/results" / f"procrustes_support_{k}_{sign}__{space}.json",
                                               f"연결 최대 {k}개 고정, {word}, {SPACES[space]}", FIT_NAMES[pre]))
    parts.append(section(6, "음수 계수의 허용 여부만으로 큰 성능 차이가 생기지는 않았다.",
        "Procrustes에서 고른 동일한 연결들 위에서 회귀를 다시 학습했다. 연결 상한 16개에서 이미지를 텍스트 좌표로 옮길 때, "
        "CC3M 대응의 Recall@10은 음수 허용이 <strong>43.58%·42.53%</strong>, 양수만 허용이 <strong>43.16%·42.71%</strong>였다.",
        '<p>여기서는 연결 번호와 회귀의 제곱오차 목적을 고정하고 계수의 부호 제약만 바꿨다. '
        '다시 학습한 회귀는 직교 제약을 유지하지 않으므로, 이 표는 원래 Procrustes 자체의 성능이 아니다.</p>'
        + retrieval_table(sign_rows),
        '이 연결 구조에서는 음수 계수가 성능 차이의 주된 원인이라는 근거가 약했다. '
        '양수만 사용하는 작은 집합을 검토할 근거는 되지만, CCA 전체에서 음수를 제거해도 된다는 결과는 아니다.'))

    agreement_rows = []
    agreement_export = []
    for pre in FIT_NAMES:
        for key, label in [("hungarian", "헝가리안"), ("cca_256", "CCA 256개 가중합"),
                           *[(f"sparse_cca_{k}",f"Sparse CCA, 가중합마다 최대 {k}개 특징") for k in K_VALUES],
                           ("cca_full_dimensions", "CCA 전체 공통 좌표, 후보 수 대조")]:
            record = ev.read(root / pre / "representative-agreement/results" / f"test__{key}.json")
            n = record["n_evaluated_categories"]
            assert n == 50
            signed = record["summary"]["agree_at1_count"]
            positive = record["positive_only_sensitivity"]["summary"]["agree_at1_count"]
            stats = record["selection_statistics"]
            row = [FIT_NAMES[pre],label,record["n_coordinates"],f"{signed}/{n} ({100*signed/n:.0f}%)",
                   f"{positive}/{n} ({100*positive/n:.0f}%)",f"{stats['distinct_image_coordinates']}개 / {stats['distinct_text_coordinates']}개"]
            agreement_rows.append(row)
            for direction in DIRECTIONS:
                s = record["summary"][direction]
                agreement_export.append(dict(condition=FIT_NAMES[pre], method=label, direction=direction,
                    evaluated_categories=n, candidate_coordinates=record["n_coordinates"],
                    **{f"agreement_at_{k}":s[f"top{k}"] for k in (1,5,10)}, positive_only_top1=positive/n))
    parts.append(section(7, "Sparse CCA의 주석 대표 대응은 유망하지만, 의미 순수성과는 구분해야 한다.",
        "서로 다른 이미지 집단에서 주석으로 고른 이미지·텍스트 대표가 같은 대응 좌표와 같은 부호를 선택했는지 확인했다. "
        "COCO에서 대응을 학습한 조건은 헝가리안 <strong>25/50</strong>, CCA <strong>33/50</strong>, Sparse CCA 32개 <strong>42/50</strong>였다.",
        '<p>COCO 물체 80개 중 독립된 이미지 절반마다 양성 표본이 50개 이상인 50개 범주만 평가했다. '
        '이미지 쪽 대표는 한쪽 절반에서, 텍스트 쪽 대표는 다른 절반에서 개념 있음·없음 AUROC로 독립 선택했다. '
        'CCA에서는 대표 번호가 SAE 좌표 하나의 번호가 아니라 여러 SAE activation을 합친 공통 좌표의 번호다. '
        '주 비교는 개념이 있을 때 증가하거나 감소하는 방향을 모두 허용하며, 증가하는 방향만 허용한 결과도 함께 보였다.</p>'
        + plain_table(["대응 학습 자료", "방법", "선택 후보 좌표 수", "대표 일치, 증가·감소 모두 허용", "대표 일치, 증가만 허용", "선택된 서로 다른 좌표 수, 이미지 / 텍스트"], agreement_rows)
        + '<p>상위 5개와 10개 안에 상대 대표가 포함되는 비율도 annotation_agreement.csv에 저장했다. '
        '이 값은 이미지·텍스트 검색 Recall과 다른 지표다.</p>',
        '<strong>84%를 “모든 연결 중 의미가 정확한 비율”이라고 해석하면 안 된다.</strong> '
        'Sparse CCA 32개의 COCO 대응에서 50개 범주가 선택한 서로 다른 좌표는 이미지 25개·텍스트 24개였다. '
        '여러 범주가 같은 좌표를 공유할 수 있다. 증가 방향만 허용하면 CCA는 72%, Sparse CCA 32개는 70%였다. '
        '따라서 대표 대응 결과는 좋은 단서이지만, 하나의 집합이 하나의 개념만 나타낸다는 증거나 CCA보다 항상 해석하기 쉽다는 증거는 아니다.'))

    probes = []
    probe_aucs = []
    labels = {"forward_ridge":"주석 예측 오차를 줄이는 특징을 순차 추가", "swap_ridge":"주석 예측용 특징의 교체도 허용",
              "fixed_support_logistic":"특징 번호를 고정하고 로지스틱 분류기로 재학습", "sparse_cca":"주석 없이 Sparse CCA 학습"}
    for k in (8,16,32):
        for method,label in labels.items():
            row=ev.result(root/'coco-fit/concept-probe-comparison/results'/f'{method}_{k}.json',label,f"공통 좌표 171개, 좌표당 특징 최대 {k}개")
            assert row["record"]["dimensions"] == 171
            probes.append(row)
            if method != "sparse_cca":
                values=row["record"]["semantic_summary"]
                probe_aucs.append([f"특징 최대 {k}개",label,f"{values['image']['mean']:.3f}",f"{values['text']['mean']:.3f}"])
    parts.append(section(8, "주석 기반 특징 선택을 개선해도 Sparse CCA와 검색 성능 차이가 남았다.",
        "공통 좌표 171개와 좌표당 특징 최대 16개를 맞췄다. 주석 기반 순차 선택의 Recall@10은 "
        "<strong>40.56%·32.94%</strong>, 특징 교체를 허용하면 <strong>40.70%·33.01%</strong>, "
        "로지스틱 분류기로 바꾸면 <strong>40.78%·32.18%</strong>였다. 같은 조건의 Sparse CCA는 <strong>47.64%·39.51%</strong>였다.",
        '<p>모두 CC3M SAE를 사용하고 동일한 COCO 학습 이미지·캡션에서 대응 가중치를 구했다. '
        '주석 기반 방법은 개념 171개의 존재 여부를 각각 예측하도록 가중합을 학습한다. 이미지의 범주 주석을 연결된 캡션에도 부여한다. '
        '텍스트에 생략된 범주도 이 학습 정답에 포함된다. 검색은 예측 확률이 아니라 학습 평균과 표준편차로 보정한 선형 점수 사이의 코사인 유사도로 수행했다. '
        'Sparse CCA는 같은 자료에서 주석 없이 학습한 앞 171개 공통 좌표를 사용했다.</p>'
        + retrieval_table(probes)
        + '<details><summary>주석 개념의 있음·없음을 구별한 평균 AUROC</summary>'
        + plain_table(["좌표당 특징 수", "주석 기반 학습 방법", "이미지 평균 AUROC", "텍스트 평균 AUROC"], probe_aucs) + '</details>',
        '<strong>앞선 주석 기반 검색 성능이 낮았던 이유를 단순한 특징 선택 실수나 회귀 손실 하나의 문제로 설명하기는 어렵다.</strong> '
        '주석 개념을 잘 예측하는 목적과 특정 이미지·캡션 쌍을 찾아내는 목적은 다르다. 이번 두 변경이 격차를 없애지는 못했으며, '
        '개념 171개가 검색에 필요한 정보를 얼마나 누락하는지와 캡션에 없는 범주를 정답으로 부여한 영향은 아직 분리하지 못했다.'))

    write_csv(root / "followup_retrieval.csv", ev.rows)
    write_csv(root / "paired_differences.csv", intervals)
    write_csv(root / "annotation_agreement.csv", agreement_export)
    atomic_json(root / "report-verification.json", dict(
        state="completed", completed_jobs=33, numerical_result_files_checked=len(ev.checked),
        retrieval_verification="Recall@1,5,10 recomputed from every saved query rank; 5000 image queries, 25014 caption queries",
        evaluation_population="identical COCO val2017 image IDs across three ablations",
        sparse_cca_verification="all 256 coordinates converged in each of eight fits",
        sinkhorn_verification="all epsilon conditions report convergence",
        bootstrap="1000 resamples of parent images, seed 0; fixed learned models",
        sources=ev.sources))
    navigation = ''.join(f'<li><a href="#finding-{i}">{label}</a></li>' for i,label in enumerate([
        "대응 학습 자료를 바꿨을 때 CCA가 개선되는지 비교했다.",
        "평균 제거와 표준편차 보정의 효과를 분리했다.",
        "동일한 입력 보정에서 일대일 대응과 가중 대응을 비교했다.",
        "학습 후 계수 제거와 처음부터 희소하게 학습하는 방법을 비교했다.",
        "Sinkhorn의 ε와 연결 수를 바꾼 결과를 비교했다.",
        "동일한 연결에서 음수 계수를 허용한 효과를 비교했다.",
        "주석으로 양쪽 대표를 독립 선택해 대응 일치를 확인했다.",
        "주석 기반 특징 선택과 분류기 변경이 검색 성능을 개선하는지 비교했다."], 1))
    page = '''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>CC3M 후속 실험 결과</title><style>
*{box-sizing:border-box}body{font:16px/1.75 system-ui,-apple-system,sans-serif;color:#192e3a;background:#f5f6f6;margin:0}
main{max-width:1360px;margin:auto;padding:40px 30px 70px}h1{font-size:34px;line-height:1.4}h2{font-size:24px;line-height:1.5}
p{max-width:1200px}section,nav,.intro{padding:26px 30px;background:white;border:1px solid #dbe2e6;border-radius:9px;margin:24px 0}
.message{font-size:18px}.interpretation{border-top:2px solid #dce5eb;padding-top:18px}.note{color:#52636e;font-size:14px}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;min-width:850px;font-size:13px;margin:18px 0}th,td{border-bottom:1px solid #dce2e7;padding:10px 11px;text-align:left;vertical-align:top}
thead th{background:#eaf0f4;white-space:nowrap}tbody th{font-weight:500;min-width:240px}td{font-variant-numeric:tabular-nums}strong{font-weight:800;color:#104f6d}
img{max-width:100%;height:auto}details{margin:22px 0}summary{cursor:pointer;color:#225d79}a{color:#205c79}li{margin:7px 0}code{overflow-wrap:anywhere}
@media(max-width:700px){main{padding:20px 12px}section,nav,.intro{padding:18px}h1{font-size:28px}h2{font-size:21px}}
</style></head><body><main><h1>CC3M 후속 실험 결과</h1>
<p>2026년 10월 6일 · 예정한 33개 작업을 모두 완료했다.</p>
<div class="intro"><p><strong>대응을 학습한 자료의 영향이 컸고, 작은 집합은 계수를 사후에 지우기보다 처음부터 희소하게 학습하는 편이 좋았다.</strong>
그러나 작은 집합의 검색 성능과 의미 해석 가능성을 동시에 확보했다는 결론에는 아직 이르지 못했다.</p>
<p>모든 검색 평가는 COCO val2017의 이미지 5,000장과 캡션 25,014개를 사용했다. Recall@1·5·10은 정답이 각각 상위 1·5·10개 안에 있는 질의의 비율이다.
각 표의 굵은 수치는 같은 비교 조건 안에서 해당 열의 최댓값이며, 통계적 유의성이나 최종 모델 선택을 뜻하지 않는다.</p>
<p>이번 후속 실험에서는 CC3M에서 학습한 SAE를 고정했다. 대응 행렬은 CC3M의 학습 이미지 2,324,763장과 캡션 2,324,763개에서 구한 조건,
COCO의 학습 이미지 94,630장과 캡션 473,400개에서 구한 조건으로 나누었다. COCO의 나머지 학습 이미지 23,657장은 조정·주석 평가용으로 분리했다.
평가 이미지로 대응 계수를 학습하지 않았다.</p>
<p class="note">예전 COCO SAE와 이번 CC3M SAE는 학습 데이터 외에도 한 표본에서 활성화하는 특징 수가 8개와 32개, 학습 반복 수가 30회와 10회로 다르다.
따라서 두 SAE의 결과 차이를 데이터셋만의 효과로 해석할 수 없다. COCO val2017은 앞선 탐색에도 사용한 평가 자료이며, 이번 결과도 반복 학습을 통한 최종 검증 성능은 아니다.</p></div>
'''+f'<nav><h2>비교한 내용</h2><ol>{navigation}</ol></nav>'+''.join(parts)+'''
<section><h2>이번 결과에서 얻은 방향</h2><ul>
<li>대응 학습 자료와 입력 보정 조건을 먼저 고정해야 한다. 동일한 CC3M SAE에서도 CCA의 Recall@10이 14.06%p·16.45%p 달라졌다.</li>
<li>작은 집합을 제안하려면 특징 수 제한을 학습에 포함하는 방법이 더 유망하다. Sparse CCA는 학습 후 계수 제거보다 좋았지만, 전체 CCA와 성능 차이가 남았다.</li>
<li>집합의 의미를 검증할 때 대표 일치와 의미 순수성을 구분해야 한다. 같은 개념의 대표들이 대응해도 여러 개념이 하나의 좌표를 공유할 수 있었다.</li>
</ul></section><section><h2>검증과 저장 범위</h2><p>서버의 33개 작업 완료 기록과 결과 파일을 확인했다. 로컬로 가져온 보고서·수치·실행 기록 870개 파일은 서버의 SHA-256과 모두 일치했다.
후속 실험의 큰 가중치 파일과 activation 캐시는 서버에 남아 있다. 검색 수치는 저장된 질의별 순위에서 다시 계산했고, Sparse CCA와 Sinkhorn의 수렴 상태도 확인했다.</p>
<p><a href="followup_retrieval.csv">양방향 Recall@1·5·10 전체 수치</a> · <a href="annotation_agreement.csv">주석 대표 대응의 상위 1·5·10개 일치율</a> ·
<a href="paired_differences.csv">이미지 단위 재표집으로 구한 성능 차이와 95% 구간</a> · <a href="report-verification.json">검증 기록과 근거 파일</a></p>
<p class="note">재표집 구간은 동일한 학습 모델의 평가 이미지 변동만 반영한다. 모델을 다시 학습했을 때의 변동과 여러 조건을 탐색한 효과를 반영하지 않는다.</p></section>
</main></body></html>'''
    (root / "report.html").write_text(page)
    print(json.dumps(dict(report=str(root / "report.html"), verified_result_files=len(ev.checked), retrieval_rows=len(ev.rows))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    args=parser.parse_args()
    cfg=yaml.safe_load(args.config.read_text())
    cfg={k:str((args.config.parent/v).resolve()) for k,v in cfg.items()}
    main_report(cfg)


if __name__ == "__main__":
    main()
