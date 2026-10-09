"""Compare completed mapping ablations without choosing settings on test results."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from mm_sae.io import atomic_json, sha256, write_csv

DIRECTIONS = {"image_to_text": "이미지로 텍스트 검색", "text_to_image": "텍스트로 이미지 검색"}
METRICS = ["recall_at_1", "recall_at_5", "recall_at_10"]


def load_rows(root: Path):
    rows = []
    for name in ("preprocessing", "cca_components", "feature_groups"):
        with (root / f"{name}.csv").open(encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                for metric in METRICS:
                    row[metric] = float(row[metric])
                rows.append(row)
    return rows


def table(rows, keys):
    selected = [r for r in rows if r["key"] in keys]
    groups = {}
    for row in selected:
        groups.setdefault((row["dataset"], row["direction"]), []).append(row)
    maxima = {g: {m: max(r[m] for r in group) for m in METRICS} for g, group in groups.items()}
    body = []
    for key, label in keys.items():
        for row in selected:
            if row["key"] != key:
                continue
            values = []
            for metric in METRICS:
                text = f'{100 * row[metric]:.2f}%'
                if row[metric] == maxima[row["dataset"], row["direction"]][metric]:
                    text = f"<strong>{text}</strong>"
                values.append(f"<td>{text}</td>")
            body.append(f'<tr><td>{html.escape(label)}</td><td>{html.escape(row["dataset"])}</td>'
                        f'<td>{DIRECTIONS[row["direction"]]}</td>{"".join(values)}</tr>')
    return ('<div class="scroll"><table><thead><tr><th>고정한 비교 조건</th><th>SAE와 대응 학습 자료</th>'
            '<th>검색 방향</th><th>Recall@1</th><th>Recall@5</th><th>Recall@10</th></tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def paired_interval(a, b, parents, direction, k, seed=0):
    """Resample parent images so five captions never count as independent images."""
    delta = (np.asarray(a) <= k).astype(float) - (np.asarray(b) <= k)
    if direction == "text_to_image":
        counts = np.bincount(parents)
        sums = np.bincount(parents, weights=delta)
    else:
        counts, sums = np.ones(len(delta)), delta
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(1000):
        idx = rng.integers(len(counts), size=len(counts))
        samples.append(sums[idx].sum() / counts[idx].sum())
    return dict(difference=float(delta.mean()),
                low=float(np.quantile(samples, .025)), high=float(np.quantile(samples, .975)))


def build(cfg, config_path):
    base = config_path.parent
    out = (base / cfg["output"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    parents_path = (base / cfg["evaluation_parents"]).resolve()
    parents = np.load(parents_path)
    rows, sources, intervals = [], {str(parents_path): sha256(parents_path)}, []
    test_ids = None
    for study in cfg["studies"]:
        root = (base / study["ablation_run"]).resolve()
        pop = json.loads((root / "population.json").read_text())
        if test_ids is not None and pop["test_image_ids"] != test_ids:
            raise ValueError("Cannot compare different evaluation image populations")
        test_ids = pop["test_image_ids"]
        if len(test_ids) != int(parents.max()) + 1:
            raise ValueError("Caption parents and evaluation images differ")
        rows.extend({**r, "dataset": study["name"]} for r in load_rows(root))
        for name in ("preprocessing.csv", "cca_components.csv", "feature_groups.csv", "population.json"):
            sources[str(root / name)] = sha256(root / name)
        records = {}
        for key in ("cca_256", "cross_svd_256"):
            path = root / "results" / f"{key}.json"
            sources[str(path)] = sha256(path)
            records[key] = json.loads(path.read_text())["retrieval"]
        for direction in DIRECTIONS:
            for k in (1, 5, 10):
                intervals.append(dict(dataset=study["name"], direction=direction, rank=k,
                                      comparison="cross_svd_256 minus cca_256",
                                      **paired_interval(records["cross_svd_256"][direction]["ranks"],
                                                        records["cca_256"][direction]["ranks"],
                                                        parents, direction, k)))
    write_csv(out / "retrieval.csv", rows)
    write_csv(out / "whitening_difference.csv", intervals)
    atomic_json(out / "provenance.json", dict(config=cfg, sources=sources,
                                              confidence_interval="1000 parent-image bootstrap draws, seed 0"))
    selection = {
        "hungarian__raw__text_projected_to_image": "Hungarian / raw",
        "hungarian__centered__text_projected_to_image": "Hungarian / centered",
        "hungarian__standardized__text_projected_to_image": "Hungarian / standardized",
        "cross_svd_256": "Cross-SVD / 256",
        "cca_256": "CCA / 256",
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True, layout="constrained")
    x = np.arange(len(selection))
    for i, direction in enumerate(DIRECTIONS):
        for j, metric in enumerate(METRICS):
            ax = axes[i, j]
            for s, study in enumerate(cfg["studies"]):
                values = [next(r[metric] * 100 for r in rows if r["dataset"] == study["name"]
                               and r["direction"] == direction and r["key"] == key) for key in selection]
                ax.bar(x + (s - .5) * .36, values, .36, label=study["name"],
                       color=["#FFD6A5", "#9BF6FF"][s])
            ax.set_xticks(x, selection.values(), rotation=30, ha="right", fontsize=8)
            ax.set_title(f'{"Image to text" if i == 0 else "Text to image"} / Recall@{[1, 5, 10][j]}')
            ax.set_ylabel("Recall (%)")
            ax.set_ylim(0, 60)
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
    axes[0, 0].legend(frameon=False)
    fig.savefig(out / "comparison.png", dpi=180)
    fig.savefig(out / "comparison.svg")
    plt.close(fig)
    sections = []
    for section in cfg["sections"]:
        sections.append(f'<section><h2>{html.escape(section["message"])}</h2>'
                        f'<p>{html.escape(section["explanation"])}</p>'
                        + table(rows, section["keys"]) + '</section>')
    notes = ''.join(f'<li>{html.escape(s)}</li>' for s in cfg['limitations'])
    followup = cfg.get("followup_report")
    followup_note = (f'<p><strong>후속 실험을 완료했다.</strong> '
                     f'<a href="{html.escape(followup)}">CC3M 후속 실험의 전체 결과 보기</a></p>'
                     if followup else
                     '<p>이 문서는 최초 대응 비교 결과를 정리했다.</p>')
    page = ('''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>CC3M과 COCO 학습 모델의 대응 비교</title><style>
body{font-family:system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#182c37;line-height:1.7}
h1{font-size:30px}h2{font-size:24px}section{margin:44px 0}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:10px 12px;border-bottom:1px solid #d8e0e5;text-align:left}th{background:#f0f5f8}
strong{color:#004f76;font-weight:800}.scroll{overflow-x:auto}img{width:100%}.note{background:#fff8e9;padding:16px 24px}
</style><h1>CC3M과 COCO 학습 모델의 대응 비교</h1>'''
            '<p>동일한 COCO val2017 이미지 5,000장과 캡션 25,014개에서 검색했다. Recall@1·5·10은 정답이 각각 상위 1·5·10개에 포함된 질의 비율이다. '
            '표의 굵은 수치는 같은 학습 자료와 검색 방향 안에서 가장 큰 값이다.</p>'
            + followup_note +
            '<img src="comparison.svg" alt="학습 자료별 검색 성능 비교">'
            + ''.join(sections) + f'<section class="note"><h2>현재 구분할 수 없는 원인</h2><ul>{notes}</ul></section>'
            '<p>전체 수치는 retrieval.csv에, SVD와 CCA의 차이에 대한 이미지 단위 재표집 구간은 whitening_difference.csv에 저장했다. '
            '재표집 구간은 학습을 반복했을 때의 변동을 포함하지 않는다.</p></html>')
    (out / "report.html").write_text(page)
    print(out / "report.html")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    build(yaml.safe_load(args.config.read_text()), args.config.resolve())


if __name__ == "__main__":
    main()
