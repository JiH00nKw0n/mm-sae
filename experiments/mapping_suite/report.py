"""Create tables and distribution plots from completed correspondence comparisons."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mm_sae.io import write_csv

LABELS = {
    "hungarian": "Hungarian", "greedy": "Row maximum", "topk": "Row top-k",
    "partial_one": "Partial 1-to-1", "partial_many": "Bounded many-to-many",
    "same_budget_many": "Many-to-many, fixed edge cap", "global_top_edges": "Global highest edges",
    "sinkhorn": "Sinkhorn", "sparse_transport": "Sparse transport",
    "sparse_factorization": "Small feature groups",
}
COLORS = ["#B8BDC5", "#B8BDC5", "#B8BDC5", "#B8BDC5", "#FFADAD", "#FFD6A5",
          "#B8BDC5", "#B8BDC5", "#CAFFBF", "#9BF6FF"]
RECALL_KS = (1, 5, 10)
SPACES = {
    "text_projected_to_image": "텍스트 activation을 이미지 특징 공간으로 옮긴 비교",
    "image_projected_to_text": "이미지 activation을 텍스트 특징 공간으로 옮긴 비교",
}
DIRECTIONS = {
    "image_to_text": "이미지로 캡션 검색",
    "text_to_image": "캡션으로 이미지 검색",
}
DENSE_LABELS = {
    "procrustes": "Procrustes",
    "cca_64_ridge_0.01": "CCA 64차원",
    "cca_256_ridge_0.01": "CCA 256차원",
}
TABLE_GROUPS = (
    ("단순 비교·대조 방법", "baseline",
     ("hungarian", "greedy", "topk", "global_top_edges", "partial_one")),
    ("기존 가중 대응·선형 변환", "reference",
     ("sinkhorn", "procrustes", "cca_64_ridge_0.01", "cca_256_ridge_0.01")),
    ("이번에 시도한 다대다 대응", "explored",
     ("partial_many", "same_budget_many", "sparse_transport", "sparse_factorization")),
)
TABLE_LABELS = {
    "hungarian": "Hungarian · 일대일 대응",
    "greedy": "Greedy · 특징마다 최고점 하나",
    "topk": "특징마다 상위 점수 여러 개",
    "global_top_edges": "전체 행렬에서 최고점 연결 선택",
    "partial_one": "연결 생략을 허용한 일대일 대응",
    "sinkhorn": "Sinkhorn · 가중 대응",
    "partial_many": "연결 수를 제한한 다대다 대응",
    "same_budget_many": "연결 수 제한 다대다의 변형",
    "sparse_transport": "희소한 가중 대응",
    "sparse_factorization": "작은 특징 집합을 통한 대응",
}


def load(path):
    return json.loads(path.read_text())


def finite(values):
    return np.asarray([v for v in values if v is not None and np.isfinite(v)], float)


def save_figure(fig, out, name):
    fig.savefig(out / (name + ".svg"), bbox_inches="tight")
    fig.savefig(out / (name + ".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def source_explanation(population, config):
    """Keep the original report wording unless a dataset description is supplied."""
    if config.get("data_mode") == "paired" or any(
        config.get(key) for key in ("source_description", "training_dataset", "evaluation_dataset")
    ):
        training = config.get("training_dataset", "학습 자료")
        evaluation = config.get("evaluation_dataset", "평가 자료")
        description = config.get("source_description", "")
        explanation = (
            f'{description} {training}에서 이미지 {population["fit_images"]:,}장으로 특징 대응을 학습했습니다. '
            f'별도로 나눈 이미지 {population["tune_images"]:,}장에서 양방향 특징 활성값 예측 오차가 '
            "가장 작은 설정을 선택했습니다. 특징 활성값은 각 특징이 입력에서 얼마나 강하게 반응하는지를 나타냅니다. "
            f'{evaluation}의 이미지 {population["test_images"]:,}장과 캡션 {population["test_captions"]:,}개로 '
            "최종 검색 성능을 평가했습니다. 한 이미지의 캡션은 모두 같은 분할에 포함했습니다. "
            "특징을 추출하는 모델과 학습된 희소 오토인코더의 가중치는 대응을 비교하는 동안 고정했습니다."
        ).strip()
    else:
        explanation = (
            "같은 coactivation correlation으로 대응을 구성하고, COCO 학습 이미지의 80%로 대응을 학습했습니다. "
            "나머지 20%에서 양방향 activation 예측 오차가 가장 작은 설정을 선택했습니다. "
            f'COCO 검증 이미지 {population["test_images"]:,}장과 캡션 {population["test_captions"]:,}개로 마지막 평가 자료를 구성했습니다. '
            "이미지의 캡션은 항상 같은 분할에 포함했습니다. SAE와 임베딩 모델은 고정했습니다."
        )
    return html.escape(explanation)


def retrieval_table(records, dense, space, highlight_best=False):
    """Group all comparisons while identifying dense methods' own evaluation spaces."""
    by_name = {r["family"]: r for r in records}
    all_metrics = [r["retrieval"][space] for r in records] + list(dense.values())
    maxima = {(direction, k): max(r[direction]["recall"][str(k)] for r in all_metrics)
              for direction in DIRECTIONS for k in RECALL_KS}
    table = '<thead><tr><th rowspan="2">방법</th><th rowspan="2">연결 수</th>'
    table += "".join(f'<th colspan="{len(RECALL_KS)}">{label}</th>' for label in DIRECTIONS.values())
    table += "</tr><tr>" + "".join(f"<th>Recall@{k}</th>" for _ in DIRECTIONS for k in RECALL_KS)
    table += "</tr></thead>"
    for title, group_class, methods in TABLE_GROUPS:
        available = [name for name in methods if name in by_name or name in dense]
        if not available:
            continue
        table += f'<tbody class="{group_class}"><tr class="group-divider"><th colspan="8" scope="rowgroup">{title}</th></tr>'
        for name in available:
            if name in dense:
                label = DENSE_LABELS.get(name, name)
                note = "별도 공통 공간에서 평가한 참고값"
                edges = "해당 없음"
                retrieval = dense[name]
            else:
                record = by_name[name]
                label = TABLE_LABELS[name]
                edge_count = record["structure"]["edge_count"]
                edges = f"{edge_count:,}"
                note = ""
                if name == "topk":
                    label = f'특징마다 최고점 {record["options"]["k"]}개 선택'
                elif name == "global_top_edges":
                    label = f"전체 행렬에서 최고점 {edge_count:,}개 연결"
                elif name == "partial_one":
                    note = "연결을 생략하는 효과를 확인하는 대조 방법"
                elif name == "same_budget_many":
                    note = f"전체 연결 수를 헝가리안과 같은 {edge_count:,}개로 제한"
                retrieval = record["retrieval"][space]
            table += f'<tr data-method="{html.escape(name)}"><th scope="row">{html.escape(label)}'
            if note:
                table += f'<span class="method-note">{html.escape(note)}</span>'
            table += "</th><td>" + edges + "</td>"
            for direction in DIRECTIONS:
                for k in RECALL_KS:
                    score = retrieval[direction]["recall"][str(k)]
                    if highlight_best and score == maxima[direction, k]:
                        table += f'<td class="best"><strong>{score:.2%}</strong></td>'
                    else:
                        table += f'<td>{score:.2%}</td>'
            table += "</tr>"
        table += "</tbody>"
    return "<div class='table-scroll'><table>" + table + "</table></div>"


def report(out: Path):
    selected = load(out / "selected.json")
    records = [load(out / "evaluation" / (r["family"] + ".json")) for r in selected]
    dense = load(out / "dense_references.json")
    names = [r["family"] for r in records]
    labels = [LABELS[n] for n in names]
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    summaries = []
    for r in records:
        s = r["structure"]
        row = {"method": r["family"], "edges": s["edge_count"],
               "image_coverage": s["image_coverage"], "text_coverage": s["text_coverage"],
               "groups": s["group_count"]}
        for space, retrieval in r["retrieval"].items():
            for direction, metric in retrieval.items():
                for k, score in metric["recall"].items():
                    row[f"{space}_{direction}_r{k}"] = score
        for direction, metric in r["prediction"].items():
            row[f"{direction}_mean_r2"] = metric["mean_r2"]
        summaries.append(row)
    write_csv(out / "summary.csv", summaries)
    for k in RECALL_KS:
        fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharey=True)
        for ax, (space, direction) in zip(axes.flat, [(s, d) for s in SPACES for d in DIRECTIONS], strict=True):
            values = [100 * r["retrieval"][space][direction]["recall"][str(k)] for r in records]
            ax.barh(labels, values, color=COLORS[:len(records)], edgecolor="#657080", linewidth=.5)
            for i, value in enumerate(values):
                ax.text(value + .08, i, f"{value:.2f}", va="center", fontsize=9)
            ax.set_title(space.replace("_", " ") + "\n" + direction.replace("_", " "))
            ax.set_xlabel(f"Recall@{k} (%) across all test queries")
            ax.set_ylim(len(labels) - .5, -.5)
            ax.set_xlim(0, max(values) * 1.18 + .5)
        fig.tight_layout()
        # Preserve the existing Recall@1 figure path for links in older reports.
        save_figure(fig, figures, "retrieval" if k == 1 else f"retrieval_r{k}")

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True)
    for ax, key in zip(axes.flat, ["image_to_text", "text_to_image", "support_forward", "support_reverse"], strict=True):
        values = [finite(r["prediction"].get(key, {}).get("r2", [])) for r in records]
        ax.boxplot(values, tick_labels=labels, showfliers=False, showmeans=True, meanline=True,
                   vert=False, patch_artist=True,
                   boxprops={"facecolor": "#9BF6FF", "alpha": .7})
        ax.axvline(0, color="#888", lw=1)
        ax.set_title(key.replace("_", " ") + "; one value per target SAE feature")
        ax.set_xlabel("Held-out R² (larger is better; negative means worse than test mean)")
        ax.invert_yaxis()
    fig.tight_layout()
    save_figure(fig, figures, "prediction_distribution")

    removals = [load(path) for path in sorted((out / "removal").glob("*.json"))]
    rows = [{"category_id": c["category_id"], "category": c["name"], "kind": c.get("kind"),
             "images": c.get("images", 0), **r} for c in removals for r in c["records"]]
    write_csv(out / "removal.csv", rows)
    if rows:
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        for ax, kind in zip(axes.flat, ["weighted_forward", "weighted_reverse", "support_forward", "support_reverse"], strict=True):
            for offset, field, color in [(-.18, "relative_mse", "#9BF6FF"), (.18, "shuffled_relative_mse", "#FFD6A5")]:
                values = [finite([r[field] for r in rows if r["family"] == n and r["readout"] == kind]) for n in names]
                ax.boxplot(values, positions=np.arange(len(names)) + offset, widths=.3, showfliers=False,
                           vert=False, patch_artist=True, boxprops={"facecolor": color}, showmeans=True,
                           meanprops={"marker": ".", "markeredgecolor": "black", "markerfacecolor": "black"})
            ax.set_yticks(np.arange(len(names)), labels)
            ax.axvline(1, color="#999", lw=1)
            ax.set_title(kind.replace("_", " ") + "\nblue: real pairs; orange: within-category shuffled pairs")
            ax.set_xlabel("Removal response relative MSE (smaller is better; no-change predictor = 1)")
            ax.invert_yaxis()
        fig.tight_layout()
        save_figure(fig, figures, "removal_distribution")
    for summary in summaries:
        for kind in ["weighted_forward", "weighted_reverse", "support_forward", "support_reverse"]:
            for field in ["relative_mse", "shuffled_relative_mse"]:
                values = finite([r[field] for r in rows if r["family"] == summary["method"] and r["readout"] == kind])
                summary[kind + "_" + field] = float(values.mean()) if len(values) else None
    write_csv(out / "summary.csv", summaries)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, field, title in zip(axes, ["edges", "image_coverage", "text_coverage"],
                               ["Number of nonzero connections", "Image feature coverage", "Text feature coverage"], strict=True):
        values = [r[field] for r in summaries]
        ax.barh(labels, values, color=COLORS[:len(records)])
        ax.set_title(title)
        ax.invert_yaxis()
        if field == "edges":
            ax.set_xscale("log")
    fig.tight_layout()
    save_figure(fig, figures, "complexity")

    population = load(out / "population.json")
    manifest = out / "manifest.json"
    config = load(manifest).get("config", {}) if manifest.exists() else {}
    explanation = source_explanation(population, config)
    body = f"<h1>다대다 특징 대응 비교</h1><p>{explanation}</p>"
    body += (
        "<p>Recall@1·5·10은 정답이 각각 검색 상위 1개·5개·10개 안에 포함된 질의의 비율입니다. "
        "이미지로 캡션을 찾을 때는 해당 이미지의 정답 캡션 중 하나 이상이 포함되면 성공입니다. "
        "캡션으로 이미지를 찾을 때는 해당 캡션의 이미지가 포함되면 성공입니다. "
        f'모든 이미지 {population["test_images"]:,}장과 캡션 {population["test_captions"]:,}개를 각각 분모에 유지했습니다. '
        "이 값은 의미가 같은 특징을 찾은 정확도와는 다릅니다.</p>"
        "<p>추가 학습이나 설정 재선택 없이 같은 검색 결과의 순위에서 Recall@5와 Recall@10을 확인했습니다. "
        "<a href='retrieval_summary.csv'>검색 결과 CSV</a>에는 전체 방법과 두 검색 방향의 수치를 함께 저장했습니다.</p>"
    )
    body += (
        "<p>표의 위쪽에는 단순 비교·대조 방법을, 가운데에는 기존 가중 대응과 선형 변환을, "
        "아래쪽에는 이번에 시도한 세 가지 다대다 방법을 배치했습니다. "
        "다대다 대응의 연결 수 고정 조건은 첫 번째 방법의 변형입니다.</p>"
        "<p>Procrustes와 CCA는 여러 특징을 섞어 만든 자체 공통 공간에서 평가한 참고값입니다. "
        "두 표에 같은 결과를 함께 표시했으며, 특징 사이의 연결 수는 해당하지 않습니다.</p>"
    )
    retrieval_rows = []
    for space, title in SPACES.items():
        body += f"<h2>{title}</h2>"
        body += retrieval_table(records, dense["results"], space)
        for r in records:
            for direction, metric in r["retrieval"][space].items():
                retrieval_rows.append({"method": r["family"], "space": space, "direction": direction,
                                       "query_count": metric["query_count"],
                                       "candidate_count": metric["candidate_count"],
                                       **{f"recall_at_{k}": metric["recall"][str(k)] for k in RECALL_KS}})
    body += "<p>각 그림의 위쪽은 이미지 특징 공간에서 비교한 결과이고, 아래쪽은 텍스트 특징 공간에서 비교한 결과입니다. 왼쪽은 이미지로 캡션을 찾고, 오른쪽은 캡션으로 이미지를 찾습니다.</p>"
    for k in RECALL_KS:
        name = "retrieval" if k == 1 else f"retrieval_r{k}"
        body += f"<h2>Recall@{k} 비교</h2><img src='figures/{name}.svg' alt='Recall@{k} 양방향 검색 결과'>"
    captions = {
        "prediction_distribution": "위쪽은 대응 가중치의 비율을 유지하고 목표 특징별 배율 하나만 학습했습니다. 아래쪽은 선택한 연결 안에서 회귀 계수를 다시 학습했습니다. 아래 결과에는 회귀 학습의 효과가 포함됩니다.",
        "removal_distribution": "범주를 가리기 전후 activation 차이를 예측했습니다. 이미지당 해당 범주를 언급한 캡션의 차이를 평균했습니다. 범주 안에서 이미지·텍스트 짝을 섞은 조건보다 오차가 낮아야 개별 이미지의 대응에 대한 근거가 됩니다. 박스 하나는 평가 가능한 범주들의 분포입니다.",
        "complexity": "연결 수와 연결된 특징 비율은 해석해야 할 양을 나타냅니다. 이 값만으로 연결의 의미가 정확하다고 판단할 수는 없습니다.",
    }
    for name, caption in captions.items():
        if name == "removal_distribution" and not rows:
            body += "<p>이 보고서에는 범주 제거 평가 결과가 없습니다.</p>"
            continue
        body += f"<p>{caption}</p><img src='figures/{name}.svg' alt='{name}'>"
    for name, r in dense["results"].items():
        for direction in DIRECTIONS:
            metric = r[direction]
            retrieval_rows.append({"method": name, "space": "transformed_common_space", "direction": direction,
                                   "query_count": metric["query_count"], "candidate_count": metric["candidate_count"],
                                   **{f"recall_at_{k}": metric["recall"][str(k)] for k in RECALL_KS}})
    write_csv(out / "retrieval_summary.csv", retrieval_rows)
    body += "<p>특징 집합이 하나의 개념을 나타내는지는 별도의 주석과 사례 검토가 필요합니다. 이번 비교만으로 다대다 대응이 spurious correlation을 해결했다고 결론 내릴 수 없습니다.</p>"
    (out / "report.html").write_text("<!doctype html><html lang='ko'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>다대다 특징 대응 비교</title><style>body{font:17px/1.6 system-ui;max-width:1440px;margin:48px auto;padding:0 24px;color:#263142}.table-scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left}thead{background:#f3f5f8}thead th[colspan]{text-align:center}th[scope=row]{font-weight:500;white-space:normal;min-width:240px}.method-note{display:block;font-size:12px;font-weight:400;color:#586578;margin-top:3px}.group-divider th{border-top:3px solid #738196;border-bottom:1px solid #c6cfdb;padding:12px 10px;text-align:left;letter-spacing:.02em}.baseline .group-divider{background:#edf0f4}.reference .group-divider{background:#e8eef8}.explored .group-divider{background:#e4f5ef}.explored th[scope=row]{font-weight:700}img{width:100%;margin:24px 0}p{max-width:1100px}</style>" + body + "</html>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    report(parser.parse_args().output)
