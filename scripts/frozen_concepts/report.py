"""Join fresh concept probes, verified reusable probes, and four-benchmark retrieval."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
PUBLISHED = ROOT / "docs/baselines-2026-11-11"


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(output, config):
    with config.open() as stream:
        protocol = json.load(stream)
    reference = read_rows(PUBLISHED / "uncapped/per-concept.csv")
    retrieval = {
        row["method"]: row
        for row in read_rows(PUBLISHED / "additional/retrieval.csv")
        if row["dataset"] == "macro_average"
    }
    results = []
    for method in [*protocol["methods"], *protocol["reuse_methods"]]:
        result = {"method": method}
        for dataset, scope in protocol["datasets"].items():
            if method in protocol["reuse_methods"]:
                rows = [r for r in reference if r["method"] == method and r["dataset"] == dataset]
            else:
                rows = read_rows(output / "complete-probes" / method / dataset / "per-concept.csv")
            if len(rows) != scope["concepts"] or len({r["concept"] for r in rows}) != len(rows):
                raise ValueError(f"Wrong concept population: {method}/{dataset}")
            for metric in ["ce", "precision", "recall", "f1"]:
                result[f"{scope['side']}_{metric}"] = mean(float(r[metric]) for r in rows)
        for side, direction in [("image", "image_to_text"), ("text", "text_to_image")]:
            result[f"{side}_R5"] = float(retrieval[method][f"{direction}_R5"])
        for metric in ["ce", "R5"]:
            result[metric] = mean([result[f"image_{metric}"], result[f"text_{metric}"]])
        results.append(result)
    dest = output / "summary.csv"
    with dest.with_suffix(".tmp").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)
    dest.with_suffix(".tmp").replace(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/report/frozen_uncapped.json")
    args = parser.parse_args()
    summarize(args.output, args.config)
