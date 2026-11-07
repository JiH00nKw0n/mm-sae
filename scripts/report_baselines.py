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
    collected_protocol = source / "evaluation.json"
    for method in definition["methods"]:
        if (source / method).is_dir() and any((source / method).glob("*/result.json")):
            budget = definition.get("concept_feature_budget")
            actual = json.loads(collected_protocol.read_text()) if collected_protocol.exists() else {}
            if actual.get("concept_feature_budget") != budget:
                raise ValueError(f"{source}: concept feature budget differs from the report")
            break
    results = {}
    for method, criteria in definition["methods"].items():
        expected = {k: v for k, v in criteria.items() if k not in ("label", "sae_objective")}
        expected.setdefault("method", method)
        native = []
        for path in source.glob("**/mappings/*/fit.json"):
            cfg = json.loads(path.read_text())["config"]
            if all(cfg.get(k) == v for k, v in expected.items()):
                for filename in path.parent.glob("evaluation/*/results.json"):
                    recipe = json.loads((filename.parent / "recipe.json").read_text())
                    if recipe.get("concept_feature_budget") != definition.get("concept_feature_budget"):
                        continue
                    if "sae_objective" in criteria:
                        if (recipe["training"].get("shared") or {}).get("method") != criteria[
                            "sae_objective"
                        ]:
                            continue
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
    return {
        "definition_sha256": signature,
        "methods": results,
        "panels": list(definition["figure"]["panels"]),
    }


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
        1,
        len(panels),
        figsize=(max(8.0, 4.2 * len(panels)), 4.8),
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    artists = []
    rows = []
    for ax, (group, title) in zip(axes[0], panels.items()):
        coordinates = []
        detail_points = []
        detail_methods = layout.get("detail_methods", [])
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
            rows.append([group, style["label"], style["rank"], image_loss, text_loss, x, y])
            shared = "sae_objective" in style
            if not shared:
                coordinates.append((x, y))
            marker = "*" if shared else ["o", "s", "D", "^"][idx % 4]
            color = plt.get_cmap("tab10")(idx % 10)
            if method in detail_methods:
                detail_points.append((x, y, marker, color))
            point = ax.scatter(
                x,
                y,
                s=155 if shared else 92,
                color=color,
                marker=marker,
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
        if len(panels) == 1 and len(detail_points) >= 2:
            from matplotlib.ticker import LinearLocator, FormatStrFormatter

            inset = ax.inset_axes([0.12, 0.40, 0.29, 0.26])
            for x, y, marker, color in detail_points:
                inset.scatter(x, y, marker=marker, color=color, s=36)
            inset.margins(x=0.16, y=0.25)
            inset.invert_xaxis()
            inset.set_title("Matched features (detail)", fontsize=8)
            for axis, fmt in ((inset.xaxis, "%.5f"), (inset.yaxis, "%.2f")):
                axis.set_major_locator(LinearLocator(3))
                axis.set_major_formatter(FormatStrFormatter(fmt))
            inset.tick_params(labelsize=7)
            inset.grid(alpha=0.15)
        ax.set_title(title if len(panels) > 1 else "COCO val2017")
        ax.margins(x=0.18, y=0.17)
        ax.grid(alpha=0.16)
        ax.invert_xaxis()
    axes[0, 0].set_ylabel("Mean bidirectional Recall@5 (%)")
    budget = definition.get("concept_feature_budget")
    xlabel = (
        f"Mean concept loss with at most {budget} SAE features per axis"
        if budget
        else "Mean image and text concept loss"
    )
    fig.supxlabel(xlabel + " (lower toward the right)")
    fig.legend(
        handles=artists,
        loc="outside upper center",
        ncols=min(3 if len(panels) == 1 else 4, len(artists)),
        frameon=False,
    )
    fig.savefig(dest / "baseline-frontier.png", dpi=190, bbox_inches="tight", pad_inches=0.12)
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
                        style["rank"],
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
                    style["rank"],
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
        "평가 좌표 수",
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
        ["자료", "개념 범위", "방법", "평가 좌표 수", "개념 수", "시험 손실", "혼합도", "분산도"],
        concept_rows,
    )
    retrieval = write_table(
        dest,
        "retrieval.csv",
        [
            "자료",
            "방법",
            "평가 좌표 수",
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
    best_loss = min(row[-2] for row in main_rows)
    best_recall = max(row[-1] for row in main_rows)
    loss_names = ", ".join(row[0] for row in main_rows if row[-2] == best_loss)
    recall_names = ", ".join(row[0] for row in main_rows if row[-1] == best_recall)
    lead = (
        f"텍스트의 {definition['figure']['primary']} 범위에서 이미지·텍스트 평균 개념 예측 손실이 "
        f"가장 낮은 방법은 **{loss_names}** ({best_loss:.5f})이고, "
        f"양방향 평균 R@5가 가장 높은 방법은 **{recall_names}** ({best_recall:.2f}%)입니다."
    )
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
    p.add_argument("--methods", nargs="+", help="Report only these configured methods, in this order")
    p.add_argument("--primary-panel", action="store_true", help="Draw only the primary text concept group")
    args = p.parse_args()
    definition = load_definition(args.config)
    if args.results.is_file():
        snapshot = json.loads(args.results.read_text())
        if args.methods is None:
            args.methods = list(snapshot["methods"])
        if "panels" in snapshot:
            layout = definition["figure"]
            layout["panels"] = {key: layout["panels"][key] for key in snapshot["panels"]}
    if args.methods:
        unknown = set(args.methods) - definition["methods"].keys()
        if unknown:
            p.error(f"Methods not defined by the report configuration: {sorted(unknown)}")
        definition["methods"] = {key: definition["methods"][key] for key in args.methods}
    if args.primary_panel:
        layout = definition["figure"]
        primary = layout["primary"]
        layout["panels"] = {primary: layout["panels"][primary]}
    render(load_measurements(args.results, definition), definition, args.docs)
