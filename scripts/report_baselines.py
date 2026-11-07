"""Build the configured concept tables and retrieval diagram from held-out measurements."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/report/coco_val2017.yaml"


def load_definition(path):
    definition = yaml.safe_load(path.read_text())
    for dataset in definition["concepts"].values():
        for rule in dataset["groups"].values():
            if "vocabulary" in rule:
                rule["names"] = (path.parent / rule.pop("vocabulary")).read_text().splitlines()
    return definition


def summarize_groups(result, probe_file, rules):
    # Import the suite only when computing new summaries. Plotting a snapshot needs no model packages.
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from mm_sae.evaluation.concepts import entropy_scores

    with np.load(probe_file, allow_pickle=False) as probes:
        names = probes["concepts"].astype(str)
        gains = probes["importance"]
    if len(set(names)) != len(names) or set(names) != set(result["concepts"]) or gains.shape[1] != len(names):
        raise ValueError(f"Concepts differ between {probe_file} and its measured losses")
    losses = np.array([result["concepts"][name]["test_loss"] for name in names])
    groups = {}
    for label, rule in rules.items():
        mask = np.ones(len(names), dtype=bool)
        if "names" in rule:
            mask &= np.isin(names, rule["names"])
        if "prefix" in rule:
            mask &= np.char.startswith(names, rule["prefix"])
        if rule.get("complement"):
            mask = ~mask
        if not mask.any():
            raise ValueError(f"No concepts matched {label} in {probe_file}")
        groups[label] = {
            "count": int(mask.sum()),
            "loss": float(losses[mask].mean()),
            **entropy_scores(gains[:, mask]),
        }
    return {"groups": groups, "split_sizes": result["split_sizes"]}


def load_measurements(source, definition):
    signature = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
    if source.is_file():
        snapshot = json.loads(source.read_text())
        if snapshot.get("definition_sha256") != signature:
            raise ValueError("The summary was generated with a different report definition")
        return snapshot
    results = {}
    for method, criteria in definition["methods"].items():
        native = []
        for path in source.glob("**/mappings/*/fit.json"):
            cfg = json.loads(path.read_text())["config"]
            if cfg["method"] == method and all(cfg.get(k) == v for k, v in criteria.items() if k != "label"):
                for filename in path.parent.glob("evaluation/*/results.json"):
                    run = json.loads(filename.read_text())
                    if not run["synthetic"]:
                        native.append((filename.parent, run["datasets"]))
        results[method] = {}
        for name in [*definition["concepts"], *definition["retrieval"]]:
            collected = source / method / name
            if (collected / "result.json").exists():
                result = json.loads((collected / "result.json").read_text())
                path = collected / "probes.npz"
            else:
                candidates = []
                for folder, datasets in native:
                    if name in datasets:
                        candidates.append((datasets[name], folder / f"{name}-probes.npz"))
                if len(candidates) != 1:
                    raise ValueError(
                        f"Expected one measured run for {method}/{name}, found {len(candidates)}"
                    )
                result, path = candidates[0]
            if name in definition["concepts"]:
                results[method][name] = summarize_groups(result, path, definition["concepts"][name]["groups"])
            else:
                results[method][name] = {
                    k: result[k]
                    for k in ("images", "captions", "image_to_text", "text_to_image", "truncated_fraction")
                }
    return {"definition_sha256": signature, "methods": results}


def write_table(dest, name, headers, rows):
    with (dest / name).open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)

    def cell(value):
        if value is None:
            return "해당 없음"
        if isinstance(value, float):
            return f"{value:.5f}"
        return str(value)

    table = [headers, ["---"] * len(headers), *rows]
    return "\n".join("| " + " | ".join(cell(v) for v in row) + " |" for row in table)


def figure(definition, data, dest):
    layout = definition["figure"]
    panels = layout["panels"]
    fig, axes = plt.subplots(
        1, len(panels), figsize=(4.2 * len(panels), 4.2), sharey=True, squeeze=False, layout="constrained"
    )
    artists = []
    rows = []
    for ax, (group, title) in zip(axes[0], panels.items()):
        coordinates = []
        for idx, (method, style) in enumerate(definition["methods"].items()):
            measurements = data[method]
            image_loss = float(np.mean([measurements[ds]["groups"][g]["loss"] for ds, g in layout["image"]]))
            text_loss = measurements[layout["text_dataset"]]["groups"][group]["loss"]
            x = (image_loss + text_loss) / 2
            y = 100 * float(
                np.mean(
                    [
                        measurements[ds][s]["R@5"]
                        for ds in definition["retrieval"]
                        for s in ("image_to_text", "text_to_image")
                    ]
                )
            )
            rows.append([group, style["label"], image_loss, text_loss, x, y])
            coordinates.append((x, y))
            point = ax.scatter(
                x,
                y,
                s=92,
                color=plt.get_cmap("tab10")(idx % 10),
                marker=["o", "s", "D", "^"][idx % 4],
                edgecolors="white",
                zorder=3,
                label=style["label"],
            )
            if len(artists) < len(definition["methods"]):
                artists.append(point)
        frontier = sorted(
            (x, y)
            for x, y in coordinates
            if not any(a <= x and b >= y and (a < x or b > y) for a, b in coordinates)
        )
        if len(frontier) > 1:
            ax.plot(*zip(*frontier), linewidth=1.3, linestyle="--", color="#6b7278")
        ax.set_title(title)
        ax.margins(x=0.18, y=0.17)
        ax.grid(alpha=0.16)
        ax.invert_xaxis()
    axes[0, 0].set_ylabel("Mean bidirectional Recall@5 (%)")
    fig.supxlabel("Mean image and text concept loss (lower toward the right)")
    fig.legend(handles=artists, loc="outside upper center", ncols=len(artists), frameon=False)
    fig.savefig(dest / "baseline-frontier.png", dpi=190)
    plt.close(fig)
    return rows


def render(snapshot, definition, dest):
    data = snapshot["methods"]
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "measurements.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    concept_rows, retrieval_rows = [], []
    for key, spec in definition["concepts"].items():
        for group in spec["groups"]:
            for method, style in definition["methods"].items():
                s = data[method][key]["groups"][group]
                concept_rows.append(
                    [
                        spec["label"],
                        group,
                        style["label"],
                        s["count"],
                        s["loss"],
                        s["mixing"],
                        s["dispersion"],
                    ]
                )
    for key, label in definition["retrieval"].items():
        for method, style in definition["methods"].items():
            s = data[method][key]
            retrieval_rows.append(
                [
                    label,
                    style["label"],
                    s["images"],
                    s["captions"],
                    *[
                        round(100 * s[d][f"R@{k}"], 3)
                        for d in ("image_to_text", "text_to_image")
                        for k in (1, 5, 10)
                    ],
                ]
            )
    coords = figure(definition, data, dest)
    coord_cols = [
        "텍스트 개념",
        "방법",
        "이미지 평균 손실",
        "텍스트 평균 손실",
        "모달리티 평균 손실",
        "양방향 평균 R@5 (%)",
    ]
    write_table(dest, "figure-points.csv", coord_cols, coords)
    main_rows = [r[1:] for r in coords if r[0] == definition["figure"]["primary"]]
    main = write_table(dest, "baseline.csv", coord_cols[1:], main_rows)
    concepts = write_table(
        dest,
        "concepts.csv",
        ["자료", "개념 범위", "방법", "개념 수", "시험 손실", "혼합도", "분산도"],
        concept_rows,
    )
    retrieval = write_table(
        dest,
        "retrieval.csv",
        [
            "자료",
            "방법",
            "이미지 수",
            "캡션 수",
            *[f"{side} R@{k} (%)" for side in ("이미지", "텍스트") for k in (1, 5, 10)],
        ],
        retrieval_rows,
    )
    populations = []
    for key, spec in definition["concepts"].items():
        sizes = [measurements[key]["split_sizes"] for measurements in data.values()]
        if any(size != sizes[0] for size in sizes):
            raise ValueError(f"Methods evaluated on different populations for {key}")
        populations.extend([spec["label"], split, n] for split, n in sizes[0].items())
    write_table(dest, "populations.csv", ["자료", "분할", "표본 수"], populations)
    best_loss = min(main_rows, key=lambda r: r[-2])
    best_recall = max(main_rows, key=lambda r: r[-1])
    lead = f"텍스트의 {definition['figure']['primary']} 범위에서 이미지·텍스트 평균 개념 예측 손실이 가장 낮은 방법은 **{best_loss[0]}** ({best_loss[-2]:.5f})이고, 양방향 평균 R@5가 가장 높은 방법은 **{best_recall[0]}** ({best_recall[-1]:.2f}%)입니다."
    body = [
        f"# {definition['title']}",
        lead,
        main,
        "![개념 예측과 이미지·텍스트 검색](baseline-frontier.png)",
        (dest / "PROTOCOL.md").read_text().strip(),
        "## 범주별 개념 예측",
        concepts,
        "## 이미지·텍스트 검색",
        retrieval,
        "## 재현 방법",
        (dest / "REPRODUCE.md").read_text().strip(),
    ]
    (dest / "README.md").write_text("\n\n".join(body) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("docs", type=Path)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = p.parse_args()
    definition = load_definition(args.config)
    render(load_measurements(args.results, definition), definition, args.docs)
