"""Paired image-cluster bootstrap comparisons for the three fixed-map ablations.

Run with ``python -m experiments.mapping_ablation.contrasts OUTPUT_DIRECTORY``.
The same sampled image multiplicities are used in every comparison. Caption
queries belonging to a sampled image travel together, and their denominator is
recomputed in each replicate. Intervals describe test-sample uncertainty only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import linalg

from mm_sae.analysis.data import save_json
from mm_sae.io import sha256, write_csv


SPACES = ("text_projected_to_image", "image_projected_to_text")
DIRECTIONS = ("image_to_text", "text_to_image")
RANKS = (1, 5, 10)


@dataclass(frozen=True)
class Contrast:
    experiment: str
    comparison: str
    before: str
    after: str


def planned_contrasts(records: dict[str, dict[str, Any]]) -> list[Contrast]:
    """Construct declared comparisons; never select a condition by test score."""
    result: list[Contrast] = []

    def add(experiment: str, comparison: str, before: str, after: str) -> None:
        missing = [key for key in (before, after) if key not in records]
        if missing:
            raise ValueError(f"Missing completed result for {comparison}: {missing}")
        result.append(Contrast(experiment, comparison, before, after))

    families = sorted({r["family"] for r in records.values() if r["experiment"] == "preprocessing"})
    for family in families:
        for space in SPACES:
            for before, after in (("raw", "centered"), ("centered", "standardized"),
                                  ("raw", "standardized")):
                add("preprocessing", f"{family}: {before} to {after}, {space}",
                    f"{family}__{before}__{space}", f"{family}__{after}__{space}")

    dimensions = sorted({int(r["metadata"]["dimensions"]) for r in records.values()
                         if r["family"] in ("cross_svd", "cca")})
    if not dimensions:
        raise ValueError("No completed common-space comparisons were found")
    for dimension in dimensions:
        add("cca_components", f"Within-modality covariance correction at {dimension} dimensions",
            f"cross_svd_{dimension}", f"cca_{dimension}")
    for family in ("cross_svd", "cca"):
        for before, after in zip(dimensions[:-1], dimensions[1:], strict=True):
            add("cca_components", f"{family}: {before} to {after} dimensions",
                f"{family}_{before}", f"{family}_{after}")
    # The highest requested rank is full in the complete protocol. Smoke runs
    # may request a smaller rank, which remains explicit in the comparison name.
    add("cca_components", f"Procrustes to common SVD at {dimensions[-1]} dimensions",
        "procrustes", f"cross_svd_{dimensions[-1]}")

    for space in SPACES:
        before = f"sparse_factorization__standardized__{space}"
        for condition in ("raw_factors", "weighted_average", "unit_variance"):
            add("feature_groups", f"Normalized graph in {space} to direct groups with {condition}",
                before, f"groups__{condition}")
        add("feature_groups", f"Unnormalized graph in {space} to direct groups with raw factors",
            f"groups__unnormalized_mapping__{space}", "groups__raw_factors")
    add("feature_groups", "Raw group factors to group weighted averages",
        "groups__raw_factors", "groups__weighted_average")
    add("feature_groups", "Group weighted averages to unit-variance groups",
        "groups__weighted_average", "groups__unit_variance")
    group_rank = int(records["groups__unit_variance"]["metadata"]["dimensions"])
    if group_rank in dimensions:
        add("feature_groups", f"CCA to unit-variance groups at {group_rank} dimensions",
            f"cca_{group_rank}", "groups__unit_variance")
    return result


def cluster_hits(record: dict[str, Any], parents: np.ndarray, images: int) -> dict[str, np.ndarray]:
    """Return per-image successful-query counts for every direction and cutoff."""
    result = {}
    for direction in DIRECTIONS:
        metric = record["retrieval"][direction]
        ranks = np.asarray(metric["ranks"], dtype=np.int64)
        queries, candidates = (images, len(parents)) if direction == "image_to_text" else (len(parents), images)
        if (ranks.shape != (queries,) or metric["query_count"] != queries
                or metric["candidate_count"] != candidates or np.any(ranks < 1)):
            raise ValueError(f"Inconsistent query population for {record['key']}/{direction}")
        values = []
        for rank in RANKS:
            hits = (ranks <= min(rank, candidates)).astype(np.float64)
            np.testing.assert_allclose(metric["recall"][str(rank)], hits.mean(), atol=1e-14, rtol=0)
            values.append(hits if direction == "image_to_text" else
                          np.bincount(parents, weights=hits, minlength=images))
        result[direction] = np.column_stack(values)
    return result


def paired_intervals(differences: np.ndarray, denominators: np.ndarray,
                     multiplicities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap ratios of summed per-image hit differences to query counts.

    Each column is one metric. ``denominators`` has the same shape and contains
    one query per image or all caption queries belonging to that image.
    """
    differences = np.asarray(differences, dtype=np.float64)
    denominators = np.asarray(denominators, dtype=np.float64)
    multiplicities = np.asarray(multiplicities, dtype=np.float64)
    if (differences.ndim != 2 or denominators.shape != differences.shape
            or multiplicities.ndim != 2 or multiplicities.shape[1] != len(differences)):
        raise ValueError("Cluster values and bootstrap multiplicities have inconsistent dimensions")
    gemm = linalg.get_blas_funcs(("gemm",), dtype=np.float64)[0]
    numerator = gemm(alpha=1, a=multiplicities, b=differences)
    denominator = gemm(alpha=1, a=multiplicities, b=denominators)
    if np.any(denominator <= 0):
        raise ValueError("A bootstrap sample has no evaluable queries")
    estimates = 100 * numerator / denominator
    interval = np.percentile(estimates, [2.5, 97.5], axis=0)
    return interval[0], interval[1]


def contrasts(out: Path, *, resamples: int = 1000, seed: int = 20261004) -> list[dict[str, Any]]:
    if resamples < 1:
        raise ValueError("Bootstrap resamples must be positive")
    paths = sorted((out / "results").glob("*.json"))
    records = {}
    for path in paths:
        record = json.loads(path.read_text())
        if record["key"] in records:
            raise ValueError(f"Duplicate result key: {record['key']}")
        records[record["key"]] = record
    parents = np.load(out / "parents.npy")
    if parents.ndim != 1 or parents.dtype.kind not in "iu" or not len(parents):
        raise ValueError("parents.npy must contain one integer image index per caption")
    images = int(json.loads((out / "population.json").read_text())["test_images"])
    np.testing.assert_array_equal(np.unique(parents), np.arange(images))
    counts = np.bincount(parents, minlength=images).astype(np.float64)
    hits = {key: cluster_hits(record, parents, images) for key, record in records.items()}
    plans = planned_contrasts(records)
    rows: list[dict[str, Any]] = []
    differences, denominators = [], []
    for plan in plans:
        for direction in DIRECTIONS:
            denominator = np.ones(images) if direction == "image_to_text" else counts
            before, after = hits[plan.before][direction], hits[plan.after][direction]
            for column, rank in enumerate(RANKS):
                change = after[:, column] - before[:, column]
                differences.append(change)
                denominators.append(denominator)
                rows.append({"experiment": plan.experiment, "comparison": plan.comparison,
                             "before_key": plan.before, "after_key": plan.after,
                             "direction": direction, "recall_at": rank,
                             "query_count": int(denominator.sum()),
                             "before_percent": float(100 * before[:, column].sum() / denominator.sum()),
                             "after_percent": float(100 * after[:, column].sum() / denominator.sum()),
                             "difference_percentage_points": float(100 * change.sum() / denominator.sum())})
    rng = np.random.default_rng(seed)
    multiplicities = rng.multinomial(images, np.full(images, 1 / images), size=resamples)
    lower, upper = paired_intervals(np.column_stack(differences), np.column_stack(denominators), multiplicities)
    for row, low, high in zip(rows, lower, upper, strict=True):
        row.update(ci95_lower_percentage_points=float(low), ci95_upper_percentage_points=float(high))
    metadata = {"resamples": resamples, "seed": seed, "cluster_unit": "parent image",
                "image_clusters": images, "caption_queries": len(parents),
                "direction_of_difference": "after minus before",
                "units": "percentage points",
                "confidence_interval": "Nominal uncorrected 95% percentile interval",
                "caption_weighting": "All captions follow their sampled parent; denominator resampled too",
                "shared_resamples": True,
                "scope": "Fixed fitted models and fixed predeclared conditions; no training-seed uncertainty",
                "planned_comparisons": len(plans), "metric_comparisons": len(rows),
                "source_hashes": {str(path.relative_to(out)): sha256(path)
                                  for path in [*paths, out / "parents.npy", out / "population.json"]},
                "code_hash": sha256(Path(__file__))}
    write_csv(out / "contrasts.csv", rows)
    save_json(out / "contrasts.json", {"protocol": metadata, "comparisons": rows})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    rows = contrasts(args.output.resolve(), resamples=args.resamples, seed=args.seed)
    print(json.dumps({"metric_comparisons": len(rows), "bootstrap_resamples": args.resamples,
                      "output": str(args.output.resolve() / "contrasts.json")}))


if __name__ == "__main__":
    main()
