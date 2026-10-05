"""Self-contained report of the two controlled annotation-readout comparisons."""

from __future__ import annotations

import base64
import csv
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "forward_ridge": "기존 방식으로 특징을 추가하며 선택",
    "swap_ridge": "기존 방식에 특징 교체를 추가",
    "fixed_support_logistic": "기존 특징을 고정하고 로지스틱 분류기를 학습",
    "sparse_cca": "Sparse CCA",
}
SHORT = {"forward_ridge": "Forward ridge", "swap_ridge": "Ridge + swaps",
         "fixed_support_logistic": "Fixed-support logistic", "sparse_cca": "Sparse CCA"}
COLORS = ["#e89595", "#dfb56d", "#70a9c4", "#819e70"]
DIRECTIONS = ("image_to_text", "text_to_image")


def recall(record, direction, k):
    return 100 * record["retrieval"][direction]["recall"][str(k)]


def table(records):
    headers = "<tr><th rowspan='2'>방법</th><th rowspan='2'>출력당 특징 수</th>"
    headers += "<th colspan='3'>이미지로 텍스트 검색 (%)</th><th colspan='3'>텍스트로 이미지 검색 (%)</th></tr><tr>"
    headers += "".join(f"<th>Recall@{k}</th>" for _ in DIRECTIONS for k in (1, 5, 10)) + "</tr>"
    maxima = {(d, k): max(recall(r, d, k) for r in records) for d in DIRECTIONS for k in (1, 5, 10)}
    body = []
    for r in records:
        cells = []
        for d in DIRECTIONS:
            for k in (1, 5, 10):
                value = recall(r, d, k)
                text = f"{value:.2f}"
                if value == maxima[d, k]:
                    text = f"<strong>{text}</strong>"
                cells.append(f"<td>{text}</td>")
        body.append(f"<tr><td>{escape(LABELS[r['method']])}</td><td>{r['k']}개</td>{''.join(cells)}</tr>")
    return f"<div class='scroll'><table><thead>{headers}</thead><tbody>{''.join(body)}</tbody></table></div>"


def embed(path):
    data = base64.b64encode(path.read_bytes()).decode()
    return f"<img src='data:image/png;base64,{data}' alt='{escape(path.stem)}'>"


def plot_results(output, records, budgets):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, direction, title in zip(axes, DIRECTIONS, ("Image to text", "Text to image"), strict=True):
        for method, color in zip(LABELS, COLORS, strict=True):
            rows = [records[(method, k)] for k in budgets]
            ax.plot(budgets, [recall(r, direction, 10) for r in rows], marker="o", label=SHORT[method], color=color)
        ax.set(title=title, xlabel="Features per output coordinate", ylabel="Recall@10 (%)", xticks=budgets)
        ax.grid(axis="y", alpha=.2)
    axes[1].legend(fontsize=8)
    path = output / "retrieval_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(budgets), figsize=(12, 6.5), sharey=True, squeeze=False, constrained_layout=True)
    for row, side in enumerate(("image", "text")):
        for col, k in enumerate(budgets):
            ax = axes[row, col]
            values = [[s[side + "_auc"] for s in records[(m, k)]["semantic"] if s[side + "_auc"] is not None]
                      for m in list(LABELS)[:3]]
            boxes = ax.boxplot(values, tick_labels=["Forward", "+ swaps", "Logistic"], whis=(0, 100),
                               patch_artist=True, showmeans=True,
                               meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 4})
            for box, color in zip(boxes["boxes"], COLORS[:3], strict=True):
                box.set_facecolor(color)
            ax.set(title=f"{side.capitalize()}, k={k}", ylim=(0, 1.01))
            ax.axhline(.5, color="#999", linestyle="--", linewidth=.8)
            ax.grid(axis="y", alpha=.15)
            if col == 0:
                ax.set_ylabel("Concept presence AUROC")
    path2 = output / "concept_auroc_boxplots.png"
    fig.savefig(path2, dpi=180)
    plt.close(fig)
    return path, path2


def delta_text(records, alternative, budgets):
    messages = []
    for k in budgets:
        base, alt = records[("forward_ridge", k)], records[(alternative, k)]
        delta = [recall(alt, d, 10) - recall(base, d, 10) for d in DIRECTIONS]
        messages.append(f"특징 {k}개에서 Recall@10은 이미지로 텍스트를 찾을 때 {delta[0]:+.2f}%p, "
                        f"텍스트로 이미지를 찾을 때 {delta[1]:+.2f}%p 변했습니다.")
    return " ".join(messages)


def comparison_message(records, alternative, budgets):
    changes = [recall(records[(alternative, k)], direction, 10)
               - recall(records[("forward_ridge", k)], direction, 10)
               for k in budgets for direction in DIRECTIONS]
    if max(changes) < 0:
        result = "모든 특징 수에서 양방향 Recall@10이 기존 방식보다 낮았습니다."
    elif min(changes) > 0:
        result = "모든 특징 수에서 양방향 Recall@10이 기존 방식보다 높았습니다."
    elif all(x == 0 for x in changes):
        result = "모든 특징 수에서 양방향 Recall@10이 기존 방식과 같았습니다."
    else:
        result = "특징 수와 검색 방향을 바꾸어도 일관되게 좋아지는 결과는 나오지 않았습니다."
    return result + f" 기존 방식 대비 Recall@10의 최대 절대 변화는 {max(abs(x) for x in changes):.2f}%p였습니다."


def build_report(output):
    output = Path(output)
    protocol = json.loads((output / "protocol.json").read_text())
    budgets = protocol["budgets"]
    records = {}
    for method in LABELS:
        for k in budgets:
            r = json.loads((output / "results" / f"{method}_{k}.json").read_text())
            records[(method, k)] = r
    first, second = plot_results(output, records, budgets)
    rows = []
    semantic_rows = []
    for r in records.values():
        row = dict(method=r["method"], features_per_output=r["k"], dimensions=r["dimensions"])
        for direction in DIRECTIONS:
            for k in (1, 5, 10):
                row[f"{direction}_recall_at_{k}_percent"] = recall(r, direction, k)
        for side in ("image", "text"):
            row[side + "_presence_auroc_mean"] = r["semantic_summary"][side].get("mean")
        rows.append(row)
        semantic_rows.extend(dict(method=r["method"], features_per_output=r["k"], **s) for s in r["semantic"])
    for name, data in (("summary.csv", rows), ("semantic_per_concept.csv", semantic_rows)):
        if data:
            with (output / name).open("w", newline="") as f:
                writer = csv.DictWriter(f, list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    sections = []
    comparisons = [
        ("swap_ridge", "기존에 고른 특징을 교체하면 검색이 개선되는가?",
         "같은 주석을 예측하는 제곱오차 회귀를 유지했습니다. 기존에는 특징을 하나씩 추가하고 가중치를 다시 학습했습니다. "
         "추가 조건에서는 이미 선택한 특징 하나를 다른 특징으로 교체하는 모든 경우를 평가했습니다. "
         "학습 오차와 가중치 벌점을 가장 많이 줄이는 교체를 반복했습니다.",
         "이 비교는 기존의 순차 선택에서 고친 특징 집합이 검색에도 도움이 되는지를 보여줍니다. "
         "교체 한 번으로 더 개선할 수 없을 때 종료했으며, 가능한 모든 특징 집합 중 최적이라는 뜻은 아닙니다."),
        ("fixed_support_logistic", "같은 특징으로 개념 분류기를 학습하면 검색이 개선되는가?",
         "기존 방식에서 선택한 특징 번호를 그대로 고정했습니다. 해당 개념의 존재 여부를 예측할 때 "
         "제곱오차를 줄이는 대신 로지스틱 분류 손실을 줄였습니다. 가중치와 절편만 새로 학습했습니다. "
         "검색에는 양쪽 모두 학습 평균을 빼고 표준편차로 나눈 선형 점수를 사용했습니다.",
         "이 비교는 같은 특징을 조합하는 가중치의 학습 목표를 바꾼 효과입니다. "
         "로지스틱 분류기로 특징 선택까지 새로 수행한 결과는 아닙니다."),
    ]
    for method, title, description, interpretation in comparisons:
        selected = [records[(m, k)] for k in budgets for m in ("forward_ridge", method)]
        sections.append(f"<section><h2>{title}</h2><p class='message'><strong>"
                        f"{comparison_message(records, method, budgets)}</strong></p><p>{description}</p>"
                        f"{table(selected)}<p>{delta_text(records, method, budgets)}</p><p>{interpretation}</p></section>")
    reference = [records[(m, k)] for k in budgets for m in LABELS]
    auroc = []
    for r in reference:
        if r["method"] == "sparse_cca":
            continue
        cells = []
        for side in ("image", "text"):
            s = r["semantic_summary"][side]
            cells.append(f"<td>{s['mean']:.4f} ± {s['std']:.4f}</td>")
        auroc.append(f"<tr><td>{LABELS[r['method']]}</td><td>{r['k']}개</td>{''.join(cells)}</tr>")
    largest = max(budgets)
    original = records[("forward_ridge", largest)]["semantic_summary"]
    logistic = records[("fixed_support_logistic", largest)]["semantic_summary"]
    semantic_message = (f"특징 {largest}개에서 기존 방식의 평균 AUROC는 이미지 {original['image']['mean']:.4f}, "
                        f"텍스트 {original['text']['mean']:.4f}였습니다. 같은 특징으로 로지스틱 분류기를 학습하면 "
                        f"이미지 {logistic['image']['mean']:.4f}, 텍스트 {logistic['text']['mean']:.4f}였습니다.")
    html = f"""<!doctype html><html lang='ko'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>주석 기반 특징 선택과 분류 손실의 비교</title>
<style>body{{font-family:system-ui,sans-serif;margin:40px auto;padding:0 24px;max-width:1180px;color:#1e293b;line-height:1.75;background:#fbfcff}}h1{{font-size:30px}}h2{{margin-top:0;font-size:24px}}section{{background:white;border:1px solid #dce2ea;border-radius:10px;padding:26px;margin:24px 0}}p{{max-width:1050px}}.message{{background:#eef3f8;padding:16px;border-radius:6px}}table{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}}th,td{{border-bottom:1px solid #dde3ea;padding:10px;text-align:right}}th:first-child,td:first-child{{text-align:left;min-width:220px}}th{{background:#f2f5f8}}img{{width:100%;height:auto}}.scroll{{overflow:auto}}.small{{font-size:14px;color:#526173}}</style>
<h1>주석 기반 특징 선택과 분류 손실을 나누어 비교했습니다</h1>
<p>기존 SAE와 학습 자료를 고정한 뒤, 특징을 교체하는 효과와 분류 손실을 바꾸는 효과를 각각 확인했습니다. "주석 기반"은 개념의 존재 여부를 예측하도록 가중치를 학습했다는 뜻입니다.</p>
<p class='small'>Recall@1·5·10은 검색 결과 상위 1·5·10개 안에 정답이 포함된 질의 비율입니다. 이미지 질의에는 정답 캡션 중 하나만 포함되어도 성공으로 계산했습니다. 표의 굵은 숫자는 각 표와 열 안의 최댓값입니다.</p>
{''.join(sections)}
<section><h2>같은 출력 수에서 Sparse CCA와 비교했습니다</h2>
<p>모든 조건의 출력은 {len(protocol['concepts'])}개입니다. 주석 기반 출력 하나는 지정된 개념 하나를 예측합니다. Sparse CCA 출력 하나는 이미지와 텍스트의 점수가 함께 변하도록 학습한 가중합이며, 미리 정한 개념 이름이 없습니다. 출력당 사용한 특징 수를 각각 동일하게 맞췄습니다.</p>
{embed(first)}{table(reference)}</section>
<section><h2>검색 성능과 개념 구별 능력을 따로 확인했습니다</h2>
<p class='message'><strong>{semantic_message}</strong></p>
<p>각 범주의 존재 여부를 예측한 AUROC를 평가했습니다. AUROC는 그 개념이 있는 표본 하나와 없는 표본 하나를 각각 뽑았을 때, 있는 표본에 더 높은 점수를 주는 비율이며 동점은 절반으로 계산했습니다. 원본과 가림본을 비교한 값이 아닙니다. 텍스트의 정답도 연결된 이미지의 존재 주석입니다.</p>
{embed(second)}<p class='small'>상자는 25·50·75% 분위수이고 끝선은 최솟값과 최댓값입니다. 흰 마름모는 평균입니다. 분포의 단위는 개념 범주입니다. 아래 표는 범주별 AUROC의 평균과 표준편차입니다.</p>
<div class='scroll'><table><tr><th>방법</th><th>개념당 특징 수</th><th>이미지 개념 존재 AUROC</th><th>텍스트 개념 존재 AUROC</th></tr>{''.join(auroc)}</table></div>
<p>Sparse CCA의 출력에는 개념 이름을 지정하지 않았으므로, 출력 번호를 주석 범주 번호로 간주한 AUROC는 계산하지 않았습니다.</p></section>
<section><h2>평가 조건과 해석 범위</h2>
<p>학습 이미지 {protocol['fit_images']:,}장과 이미지·캡션 {protocol['fit_caption_pairs']:,}쌍을 사용했습니다. 평가에는 학습과 겹치지 않는 COCO val2017 이미지 {protocol['test_images']:,}장과 연결된 캡션을 사용했습니다. SAE는 재학습하지 않았습니다. 이미지의 존재 주석을 해당 캡션에도 부여했으므로, 캡션에 적혀 있지 않은 개념도 학습 정답에 포함됩니다.</p>
<p>가중치의 크기에 부과하는 벌점 계수는 제곱오차 회귀와 로지스틱 분류 모두 {protocol['ridge']}로 고정했습니다. 서로 다른 손실에서 이 값이 각각 최선이라는 뜻은 아닙니다. 특징 선택과 가중치 학습에는 평가 주석을 사용하지 않았습니다. 기존 연구 중 여러 번 확인한 평가 자료이므로, 최종 방법을 정한 뒤 별도 자료에서 재확인이 필요합니다.</p>
<p class='small'>원시 결과와 가중치, 수렴 진단, 입력 파일의 해시를 이 보고서와 같은 결과 폴더에 저장했습니다. 수치는 통계적 유의성을 주장하는 검정 결과가 아닙니다.</p></section></html>"""
    (output / "report.html").write_text(html)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    build_report(parser.parse_args().output)
