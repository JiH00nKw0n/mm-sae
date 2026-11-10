#!/usr/bin/env python3
"""Validate, bootstrap, and report the full controlled multi-caption run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METHODS = (
    "cca",
    "cca_cov",
    "cca_cov64",
    "hungarian",
    "pls",
    "procrustes",
    "sinkhorn",
    "sparse_cca",
    "sparse_pls",
)
AGGREGATIONS = ("clip_mean_pre_sae", "activation_mean_post_sae")
KS = (1, 2, 3, 5)
CONTROLS = ("same_image", "duplicate", "cross_image")
REFERENCE_NAMES = {
    "hungarian": "Hungarian",
    "pls": "PLS-SVD",
    "cca": "CCA (λ=1)",
    "sparse_pls": "Sparse PLS-SVD (k=16)",
    "sparse_cca": "Sparse CCA (k=16)",
    "cca_cov": "CCA + covariance rotation",
    "cca_cov64": "CCA + rotation (top 64)",
    "sinkhorn": "Sinkhorn (256 pairs)",
    "procrustes": "Procrustes (256 pairs)",
}
REFERENCE_COLUMNS = {
    "image_to_text_R@1": "이미지 R@1 (%)",
    "image_to_text_R@5": "이미지 R@5 (%)",
    "image_to_text_R@10": "이미지 R@10 (%)",
    "text_to_image_R@1": "텍스트 R@1 (%)",
    "text_to_image_R@5": "텍스트 R@5 (%)",
    "text_to_image_R@10": "텍스트 R@10 (%)",
}


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def prefix(seed: int, k: int, control: str, aggregation: str) -> str:
    return f"s{seed}_k{k}_{control}_{aggregation}"


def paired_r5(npz, seed: int, k: int, control: str, aggregation: str) -> np.ndarray:
    name = prefix(seed, k, control, aggregation)
    return 0.5 * (
        (npz[name + "_image_ranks"] <= 5).astype(np.float32)
        + (npz[name + "_text_ranks"] <= 5).astype(np.float32)
    )


def hierarchical_ci(deltas: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    seeds, images = deltas.shape
    values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        seed_rows = rng.integers(0, seeds, seeds)
        image_rows = rng.integers(0, images, images)
        values[draw] = deltas[np.ix_(seed_rows, image_rows)].mean()
    return tuple(np.quantile(values, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2_000)
    args = parser.parse_args()
    root = args.input.resolve()
    manifest = json.loads((root / "run_manifest.json").read_text())
    rows = read_rows(root / "summary_by_seed.csv")
    expected_rows = len(METHODS) + len(METHODS) * 10 * len(KS) * len(CONTROLS) * len(AGGREGATIONS)
    errors = []
    warnings = []
    if manifest.get("state") != "completed":
        errors.append("run_manifest state is not completed")
    if tuple(manifest.get("mapping_methods", [])) != tuple(sorted(METHODS)):
        errors.append("mapping method set differs from the locked nine-method scope")
    if manifest.get("seeds") != 10 or manifest.get("rows") != expected_rows or len(rows) != expected_rows:
        errors.append(f"expected {expected_rows} summary rows with exactly 10 seeds")
    reference_rows = read_rows(args.reference.resolve())
    reference = {row["방법"]: row for row in reference_rows if row["자료"] == "COCO val2017"}
    anchors = {row["method"]: row for row in rows if row["benchmark"] == "original_anchor"}
    baseline_max_absolute_error = 0.0
    baseline_method_max_error_pp = {}
    for method, display in REFERENCE_NAMES.items():
        if method not in anchors or display not in reference:
            errors.append(f"missing published anchor comparison for {method}")
            continue
        method_max = 0.0
        for metric, column in REFERENCE_COLUMNS.items():
            difference = abs(float(anchors[method][metric]) - float(reference[display][column]) / 100.0)
            baseline_max_absolute_error = max(baseline_max_absolute_error, difference)
            method_max = max(method_max, difference)
        baseline_method_max_error_pp[method] = method_max * 100.0
    if baseline_max_absolute_error > 0.00011:
        warnings.append(
            f"original benchmark does not reproduce the 2026-11-08 table: max error {baseline_max_absolute_error}"
        )

    bootstrap_rows = []
    duplicate_max_rank_difference = 0
    k1_control_max_rank_difference = 0
    rng = np.random.default_rng(20261110)
    for method in METHODS:
        path = root / f"per_query_{method}.npz"
        if not path.exists():
            errors.append(f"missing {path.name}")
            continue
        with np.load(path) as packed:
            for aggregation in AGGREGATIONS:
                for seed in range(10):
                    base = prefix(seed, 1, "same_image", aggregation)
                    for control in CONTROLS[1:]:
                        other = prefix(seed, 1, control, aggregation)
                        for direction in ("image_ranks", "text_ranks"):
                            k1_control_max_rank_difference = max(
                                k1_control_max_rank_difference,
                                int(np.max(np.abs(packed[base + "_" + direction] - packed[other + "_" + direction]))),
                            )
                    for k in KS:
                        other = prefix(seed, k, "duplicate", aggregation)
                        for direction in ("image_ranks", "text_ranks"):
                            duplicate_max_rank_difference = max(
                                duplicate_max_rank_difference,
                                int(np.max(np.abs(packed[base + "_" + direction] - packed[other + "_" + direction]))),
                            )
                for control in CONTROLS:
                    baseline = np.stack([paired_r5(packed, seed, 1, control, aggregation) for seed in range(10)])
                    for k in (2, 3, 5):
                        target = np.stack([paired_r5(packed, seed, k, control, aggregation) for seed in range(10)])
                        delta = target - baseline
                        low, high = hierarchical_ci(delta, rng, args.bootstrap_draws)
                        bootstrap_rows.append(
                            {
                                "method": method,
                                "aggregation": aggregation,
                                "control": control,
                                "comparison": f"k{k}-k1",
                                "mean_R@5_difference": float(delta.mean()),
                                "ci95_low": float(low),
                                "ci95_high": float(high),
                            }
                        )
    if duplicate_max_rank_difference:
        errors.append("same-caption duplicate changed retrieval ranks")
    if k1_control_max_rank_difference:
        errors.append("k=1 controls are not identical")

    fields = list(bootstrap_rows[0]) if bootstrap_rows else []
    with (root / "paired_bootstrap.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(bootstrap_rows)

    primary = [r for r in bootstrap_rows if r["control"] == "same_image" and r["comparison"] == "k5-k1"]
    duplicate = [r for r in bootstrap_rows if r["control"] == "duplicate" and r["comparison"] == "k5-k1"]
    shuffled = [r for r in bootstrap_rows if r["control"] == "cross_image" and r["comparison"] == "k5-k1"]
    primary_mean = float(np.mean([r["mean_R@5_difference"] for r in primary])) if primary else float("nan")
    duplicate_mean = float(np.mean([r["mean_R@5_difference"] for r in duplicate])) if duplicate else float("nan")
    shuffled_mean = float(np.mean([r["mean_R@5_difference"] for r in shuffled])) if shuffled else float("nan")
    method_spread = float(np.std([r["mean_R@5_difference"] for r in primary])) if primary else float("nan")
    if primary_mean >= 0.005 and primary_mean > duplicate_mean + 0.005 and primary_mean > shuffled_mean + 0.005:
        interpretation = "부분 관측 병목이 실제 retrieval 실패의 유의미한 구성요소라는 방향을 지지한다."
    elif abs(primary_mean) < 0.002:
        interpretation = "현재 방법들의 주 병목을 단일 캡션의 부분 관측으로 설명하기 어렵다."
    else:
        interpretation = "관측량 효과는 있으나 대조군과 방법별 이질성을 함께 보지 않고 병목으로 단정할 수 없다."
    interaction = (
        "방법별 이동량 차이가 보여 기존 방법 순위/Pareto의 observation sensitivity를 추가 해석해야 한다."
        if method_spread >= 0.002
        else "방법들이 거의 평행 이동해 observation은 공통 난이도지만 방법 간 격차의 주원인은 아닐 가능성이 크다."
    )

    qc = {
        "state": "failed" if errors else ("verified_with_warnings" if warnings else "verified"),
        "errors": errors,
        "warnings": warnings,
        "expected_summary_rows": expected_rows,
        "observed_summary_rows": len(rows),
        "duplicate_max_rank_difference": duplicate_max_rank_difference,
        "k1_control_max_rank_difference": k1_control_max_rank_difference,
        "bootstrap_draws": args.bootstrap_draws,
        "published_anchor_max_absolute_error": baseline_max_absolute_error,
        "published_anchor_method_max_error_pp": baseline_method_max_error_pp,
    }
    (root / "VERIFICATION.json").write_text(json.dumps(qc, indent=2) + "\n")
    baseline_lines = "\n".join(
        f"- {method}: {error:.3f}%p" for method, error in sorted(baseline_method_max_error_pp.items())
    )
    report = f"""# Full controlled multi-caption 결과 보고서

> 검증 상태: **{qc['state']}**  
> 범위: 독립 SAE 기반 9개 방법, k=1/2/3/5, 10 seeds, pre/post-SAE aggregation  
> 제외: Group-Sparse와 Iso-Energy는 shared-SAE checkpoint 부재로 이 실행에 포함되지 않음

## 핵심 결과

- same-image k=5−k=1 평균 양방향 R@5 변화: **{primary_mean:+.4f}**
- duplicate control 변화: **{duplicate_mean:+.4f}** (정확 불변성 최대 rank 차이 {duplicate_max_rank_difference})
- cross-image control 변화: **{shuffled_mean:+.4f}**
- 방법×aggregation 간 primary effect 표준편차: **{method_spread:.4f}**

## So what

{interpretation} {interaction}

이 결론은 original 5,000×25,014 anchor와 controlled 5,000×5,000의 raw recall 차이가 아니라, 후보 수를 고정한 controlled benchmark 내부의 paired k 변화에만 근거한다.

## 검증

- summary rows: {len(rows)}/{expected_rows}
- k=1 세 대조군 최대 rank 차이: {k1_control_max_rank_difference}
- 11월 8일 published anchor 최대 절대 오차: {baseline_max_absolute_error:.6f}
- hierarchical paired bootstrap: seed와 image를 함께 재표집, {args.bootstrap_draws} draws
- 상세 수치: `summary_by_seed.csv`, `paired_bootstrap.csv`, `per_query_*.npz`

### 11월 8일 anchor 재현 오차

{baseline_lines}

Sparse CCA의 최대 1.30%p 드리프트 때문에 이 실행을 11월 8일 Pareto의 정확한 재현으로 취급하지 않는다. 다만 동일하게 새로 fit한 각 mapping 내부의 k=1→5 paired intervention과 대조군 비교는 유효하며, 기존 Pareto 위치와의 직접 연결만 제한한다.

## 한계

- COCO의 추가 캡션을 쓰는 diagnostic intervention이며 single-caption 배포 성능 개선법이 아니다.
- mapping은 CC3M 단일 image-caption pair로 학습된 채 고정되어 train-time missing-view 가정은 바꾸지 않는다.
- shared-SAE 두 방법이 빠져 있어 11개 방법 전체 Pareto에 대한 최종 결론은 보류한다.
- concat aggregation과 의미 coverage/객체 수에 따른 이질성 분석은 후속 localization 항목이다.
"""
    (root / "REPORT_FULL_CONTROLLED.md").write_text(report)
    if errors:
        raise SystemExit("; ".join(errors))
    (root / "FULL_CONTROLLED_VERIFIED.COMPLETE").touch()
    print(root / "REPORT_FULL_CONTROLLED.md")


if __name__ == "__main__":
    main()
