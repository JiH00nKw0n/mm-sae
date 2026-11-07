"""Render the baseline tables and measured Pareto diagram from completed results."""

from pathlib import Path
import argparse
import csv
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

methods = {"hungarian": "Hungarian", "pls": "PLS-SVD", "cca": "CCA (λ=1)"}
benchmarks = {
    "coco": "COCO",
    "stuff": "COCO-Stuff",
    "visual_genome": "Visual Genome",
    "cub": "CUB",
    "coco_retrieval": "COCO",
    "flickr30k": "Flickr30k",
    "docci": "DOCCI",
    "iiw": "IIW-400",
    "syncoco": "synCOCO",
    "syncub": "synCUB",
}


def main(root, dest):
    dest.mkdir(parents=True, exist_ok=True)
    data = {
        m: {p.parent.name: json.loads(p.read_text()) for p in (root / m).glob("*/result.json")}
        for m in methods
    }
    # Accept both collected server results and the repository's native run tree.
    for fit_path in root.glob("**/mappings/*/fit.json"):
        receipt = json.loads(fit_path.read_text())
        cfg = receipt["config"]
        if cfg["method"] not in data or cfg.get("rotation", "none") != "none" or cfg.get("prune") is not None:
            continue
        if cfg["rank"] != 256 or (cfg["method"] == "cca" and cfg["ridge"] != 1):
            continue
        for filename in fit_path.parent.glob("evaluation/*/results.json"):
            report = json.loads(filename.read_text())
            if report.get("synthetic"):
                continue
            for name, result in report["datasets"].items():
                previous = data[cfg["method"]].get(name)
                if previous is not None and previous != result:
                    raise ValueError(f"Multiple measurements for {name}. Select one run directory.")
                data[cfg["method"]][name] = result
    if not any(data.values()):
        raise ValueError("No measured baselines found")

    def table(cols, rows):
        return "\n".join(
            [
                "| " + " | ".join(cols) + " |",
                "|" + "---|" * len(cols),
                *["| " + " | ".join(map(str, r)) + " |" for r in rows],
            ]
        )

    def save_csv(file, cols, rows):
        with (dest / file).open("w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(cols)
            writer.writerows(rows)

    rows = []
    cr = []
    rr = []
    er = []
    for name, label in benchmarks.items():
        for m, method in methods.items():
            if name not in data[m]:
                continue
            j = data[m][name]
            if "macro_test_loss" in j:
                d = j["diagnostics"]
                cr.append(
                    [
                        label,
                        method,
                        j["eligible"],
                        round(j["macro_test_loss"], 6),
                        round(d["mixing"], 6),
                        round(d["dispersion"], 6),
                    ]
                )
            elif "image_to_text" in j:
                rr.append(
                    [
                        label,
                        method,
                        *[
                            round(100 * j[s][f"R@{k}"], 3)
                            for s in ("image_to_text", "text_to_image")
                            for k in (1, 5, 10)
                        ],
                        round(100 * j["truncated_fraction"], 3),
                    ]
                )
            else:
                for role, sub in j.items():
                    if not isinstance(sub, dict) or "mean_delta" not in sub:
                        continue
                    er.append(
                        [
                            label,
                            method,
                            role,
                            sub["count"],
                            sub["missing_probe"],
                            sub["mean_delta"],
                            sub["mean_absolute_delta"],
                        ]
                    )
    ccols = ["자료", "대응 방법", "개념 수", "개념 예측 손실", "개념 혼합도", "표현 분산도"]
    rcols = [
        "자료",
        "대응 방법",
        "이미지 R@1 (%)",
        "이미지 R@5 (%)",
        "이미지 R@10 (%)",
        "텍스트 R@1 (%)",
        "텍스트 R@5 (%)",
        "텍스트 R@10 (%)",
        "잘린 캡션 (%)",
    ]
    ecols = [
        "자료",
        "대응 방법",
        "편집 역할",
        "평가 건수",
        "분류기 없는 건수",
        "평균 확률 변화",
        "평균 절대 확률 변화",
    ]
    for file, cols, result in [
        ("concepts.csv", ccols, cr),
        ("retrieval.csv", rcols, rr),
        ("edits.csv", ecols, er),
    ]:
        save_csv(file, cols, result)
    common = set.intersection(*(set(b) for b in data.values()))
    final = {"coco", "stuff", "visual_genome", "coco_retrieval", "flickr30k", "docci", "iiw"} <= common
    im = [k for k in ("coco", "stuff") if k in common]
    text = ["visual_genome"] if "visual_genome" in common else []
    ret = [k for k in ("coco_retrieval", "flickr30k", "docci", "iiw") if k in common]
    caption = ""
    if im and ret:
        f, ax = plt.subplots(figsize=(7.6, 4.8), layout="constrained")
        for m, method in methods.items():
            als = [np.mean([data[m][ds]["macro_test_loss"] for ds in im])]
            if text:
                als.append(data[m][text[0]]["macro_test_loss"])
            x = float(np.mean(als))
            y = float(
                np.mean(
                    [100 * data[m][ds][s]["R@5"] for ds in ret for s in ("image_to_text", "text_to_image")]
                )
            )
            rows.append([method, x, y])
        front = sorted(
            [
                (x, y)
                for _, x, y in rows
                if not any((a <= x and b >= y and (a < x or b > y)) for _, a, b in rows)
            ]
        )
        if len(front) > 1:
            ax.plot(*zip(*front), color="#777", linestyle="--", linewidth=1)
        for i, (method, x, y) in enumerate(rows):
            color = ["#606875", "#146b83", "#a03f1c"][i]
            ax.scatter(
                x, y, s=100, color=color, marker=["s", "o", "D"][i], label=method, zorder=3, edgecolor="white"
            )
        ax.invert_xaxis()
        ax.margins(0.18, 0.17)
        ax.set(
            xlabel="Single-coordinate concept loss (lower is better →)",
            ylabel="Mean bidirectional Recall@5 (%)",
            title="Concept prediction and cross-modal retrieval",
        )
        ax.grid(alpha=0.17)
        ax.legend(loc="best")
        f.savefig(dest / "baseline-frontier.png", dpi=190)
        plt.close(f)
        save_csv("figure-points.csv", ["method", "concept_loss", "bidirectional_R5_percent"], rows)
        caption = (
            "가로축은 "
            + ", ".join(benchmarks[k] for k in im)
            + "의 이미지 개념 예측 손실 평균"
            + ("과 Visual Genome 텍스트 손실을 같은 비중으로 평균한 값" if text else "")
            + "입니다. 낮을수록 오른쪽에 표시합니다. 세로축은 "
            + ", ".join(benchmarks[k] for k in ret)
            + "에서 두 방향 R@5를 같은 비중으로 평균한 값입니다."
        )
    pending = [benchmarks[x] for x in benchmarks if x not in common]
    heading = "# CC3M SAE 대응 방법의 실제 데이터 평가\n\n"
    status = (
        "세 방법 모두 완료한 자료로 표와 그림을 작성했습니다. 남은 자료는 " + ", ".join(pending) + "입니다."
        if pending
        else "세 방법에 대해 슬라이드 11–16쪽의 평가를 마쳤습니다."
    )
    status += "\n\n2,905,954개의 CC3M 학습 쌍과 기존 이미지·텍스트 SAE를 사용했습니다. 헝가리안은 전체 일대일 할당에서 상관계수가 큰 256쌍을 사용했습니다. PLS-SVD와 규제 CCA는 각각 256개 조합을 사용했고, CCA의 규제 값은 1입니다. 세 방법 모두 후속 회전과 계수 절삭을 적용하지 않았습니다."
    if final:
        caption += " 이미지와 텍스트에 각각 50%를 배정했습니다. 세 방법 중 어떤 다른 방법도 예측 손실과 검색 성능을 동시에 개선하지 못하는 점을 점선으로 연결했습니다."
    plotsection = "\n\n![세 대응 방법의 측정 결과](baseline-frontier.png)\n\n" + caption if rows else ""
    if rows:
        plotsection += "\n\n" + table(
            ["대응 방법", "그림의 개념 예측 손실", "그림의 양방향 평균 R@5 (%)"],
            [[name, f"{x:.6f}", f"{y:.3f}"] for name, x, y in rows],
        )
    if final:
        status += " 그림에서 비교한 세 점의 좌표는 [CSV](figure-points.csv)에 있습니다."
    body = (
        heading
        + status
        + plotsection
        + "\n\n## 개념 예측\n\n"
        + table(ccols, cr)
        + "\n\n## 이미지·캡션 검색\n\n"
        + table(rcols, rr)
    )
    body += "\n\n이미지 R@5는 이미지로 정답 캡션을 상위 5개 안에서 찾은 비율입니다. 텍스트 R@5는 캡션으로 정답 이미지를 찾은 비율입니다."
    if er:
        body += "\n\n## 편집 전후의 개념 예측 확률\n\n" + table(
            ecols, [[*r[:5], *[f"{v:.5f}" if v is not None else "해당 없음" for v in r[5:]]] for r in er]
        )
    body += "\n\n## 재현 방법\n\n" + (dest / "REPRODUCE.md").read_text() + "\n"
    (dest / "README.md").write_text(body.rstrip() + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("docs", type=Path)
    a = p.parse_args()
    main(a.results, a.docs)
