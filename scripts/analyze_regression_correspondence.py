"""Evaluate frozen-feature ridge correspondence; fit/tune splits are grouped by image."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg, sparse
import yaml

from mm_sae.io import atomic_json, sha256, write_csv
from mm_sae.metrics.regression import (
    Moments, RidgePath, prediction_mse, score_matrices, sparse_moments, varying_columns,
)
from mm_sae.progress import ProgressReporter, iter_progress, stage_progress

METHODS = {
    "pearson": "기존 상관점수",
    "ridge_image_to_text": "이미지로 텍스트 예측",
    "ridge_text_to_image": "텍스트로 이미지 예측",
    "ridge_bidirectional_mean": "양방향 회귀 평균",
}
COLORS = ["#FFADAD", "#FFD6A5", "#CAFFBF", "#9BF6FF"]
LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D"]
DIRECTIONS = {"image": "이미지 기준", "text": "텍스트 기준"}


def blocks(g: np.ndarray, ni: int):
    return g[:ni, :ni], g[:ni, ni:], g[ni:, ni:]


def load_rows(path: Path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def tune(fit: Moments, validation: Moments, ni: int, penalties: list[float]):
    ii, it, tt = blocks(fit.standardized(fit), ni)
    vi, vit, vt = blocks(validation.standardized(fit), ni)
    paths = {"image_to_text": RidgePath.from_moments(ii, it),
             "text_to_image": RidgePath.from_moments(tt, it.T)}
    rows, selected = [], {}
    for direction, path in paths.items():
        xx, xy, yy = (vi, vit, vt) if direction == "image_to_text" else (vt, vit.T, vi)
        targets = fit.variance[ni:] > 0 if direction == "image_to_text" else fit.variance[:ni] > 0
        for penalty in iter_progress(penalties, f"Tune {direction}", unit="penalties"):
            mse = prediction_mse(path.coefficients(penalty), xx, xy, yy)
            rows.append({"direction": direction, "penalty": penalty,
                         "mean_standardized_mse": float(mse[targets].mean()),
                         "targets_varying_in_fit": int(targets.sum())})
        best = min((r for r in rows if r["direction"] == direction),
                   key=lambda r: (r["mean_standardized_mse"], r["penalty"]))
        selected[direction] = best["penalty"]
    return selected, rows


def labels_and_coverage(root: Path, split: str, ids: list[int]):
    base = root / "index" / split
    columns = {int(c): j for j, c in enumerate(json.loads((base / "concept_ids.json").read_text()))}
    presence = np.load(base / "presence.npy")[:, [columns[c] for c in ids]]
    mentions = np.load(base / "mentions.npy", mmap_mode="r")
    parents = np.load(base / "parents.npy")
    weights = np.bincount(parents, minlength=len(presence)).astype(float)
    p = sparse.csr_matrix(presence, dtype=np.float64)
    mean = np.asarray(p.T @ weights).ravel() / len(parents)
    covariance = (p.T @ p.multiply(weights[:, None])).toarray() / len(parents) - np.outer(mean, mean)
    scale = np.sqrt(mean * (1 - mean))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = covariance / np.outer(scale, scale)
    coverage = {}
    for j, cid in enumerate(ids):
        present = presence[parents, j]
        coverage[cid] = float(np.sum(present & mentions[:, columns[cid]]) / present.sum())
    return corr, coverage


def ranking_flags(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows are anchors; diagonals always belong to the row's own category."""
    valid = np.isfinite(scores) & np.isfinite(np.diag(scores))[:, None]
    np.fill_diagonal(valid, False)
    return valid, valid & (scores > np.diag(scores)[:, None])


def category_matrix(matrix, reps, ids, image_ids, text_ids):
    ii, tt = {int(v): i for i, v in enumerate(image_ids)}, {int(v): i for i, v in enumerate(text_ids)}
    result = np.full((len(ids), len(ids)), np.nan)
    for a, cid in enumerate(ids):
        i = ii.get(reps[str(cid)]["image"])
        for b, other in enumerate(ids):
            j = tt.get(reps[str(other)]["text"])
            if i is not None and j is not None:
                result[a, b] = matrix[i, j]
    return result


def summarize(matrices, ids, names, labels, coverage, auc, source, bin_width):
    categories, pairs, summaries, bins = [], [], [], []
    raw = matrices["pearson"]
    for method, scores in matrices.items():
        for direction in DIRECTIONS:
            s = scores if direction == "image" else scores.T
            r = raw if direction == "image" else raw.T
            for a, cid in enumerate(ids):
                valid = np.isfinite(s[a]) & np.isfinite(r[a])
                valid[a] = False
                if not np.isfinite(s[a, a]) or not np.isfinite(r[a, a]):
                    continue
                opponents = np.flatnonzero(valid)
                if not len(opponents):
                    continue
                higher = s[a, opponents] > s[a, a]
                raw_higher = r[a, opponents] > r[a, a]
                q = [auc.get((cid, side)) for side in ["image", "text"]]
                quality = min(q) if all(v is not None for v in q) else None
                categories.append({"source": source, "method": method, "direction": direction,
                    "category_id": cid, "name": names[cid], "category_type": "object" if cid < 91 else "background",
                    "opponents": len(opponents), "higher_opponents": int(higher.sum()),
                    "any_higher": bool(higher.any()), "strict_rank": int(1 + higher.sum()),
                    "tied_opponents": int(np.sum(s[a, opponents] == s[a, a])),
                    "same_score": float(s[a, a]), "same_score_positive": bool(s[a, a] > 0),
                    "raw_any_higher": bool(raw_higher.any()),
                    "image_auroc_val": q[0], "text_auroc_val": q[1], "min_auroc_val": quality,
                    "caption_mention_rate": coverage[cid]})
                for k, b in enumerate(opponents):
                    other = ids[b]
                    pairs.append({"source": source, "method": method, "direction": direction,
                        "anchor_id": cid, "candidate_id": other, "anchor_name": names[cid],
                        "candidate_name": names[other], "annotation_correlation": float(labels[a, b]),
                        "anchor_type": "object" if cid < 91 else "background",
                        "candidate_type": "object" if other < 91 else "background",
                        "same_score": float(s[a, a]), "other_score": float(s[a, b]),
                        "higher": bool(higher[k]), "raw_higher": bool(raw_higher[k]),
                        "resolved": bool(raw_higher[k] and not higher[k]),
                        "introduced": bool(not raw_higher[k] and higher[k])})
            cr = [row for row in categories if row["method"] == method and row["direction"] == direction]
            pr = [row for row in pairs if row["method"] == method and row["direction"] == direction]
            for group in ["all", "object", "background", "min_auc_ge_0.6", "min_auc_ge_0.7", "min_auc_ge_0.8"]:
                selected = cr if group == "all" else [row for row in cr if (
                    row["min_auroc_val"] is not None and row["min_auroc_val"] >= float(group[-3:])
                    if group.startswith("min_auc") else row["category_type"] == group)]
                keep = {row["category_id"] for row in selected}
                ps = [row for row in pr if row["anchor_id"] in keep]
                summaries.append({"source": source, "method": method, "direction": direction,
                    "group": group, "categories": len(selected),
                    "categories_with_higher": sum(row["any_higher"] for row in selected),
                    "category_percent": 100 * np.mean([row["any_higher"] for row in selected]) if selected else None,
                    "pairs": len(ps), "higher_pairs": sum(row["higher"] for row in ps),
                    "pair_percent": 100 * np.mean([row["higher"] for row in ps]) if ps else None,
                    "resolved_pairs": sum(row["resolved"] for row in ps),
                    "introduced_pairs": sum(row["introduced"] for row in ps),
                    "mean_strict_rank": float(np.mean([row["strict_rank"] for row in selected])) if selected else None,
                    "positive_same_scores": sum(row["same_score_positive"] for row in selected)})
            edges = np.round(np.arange(0, 1 + bin_width / 2, bin_width), 8)
            for lo, hi in zip(edges[:-1], edges[1:]):
                ps = [row for row in pr if lo <= row["annotation_correlation"]
                      and (row["annotation_correlation"] < hi or hi == 1 and row["annotation_correlation"] <= 1)]
                values = [row["other_score"] for row in ps]
                bins.append({"source": source, "method": method, "direction": direction,
                    "left": float(lo), "right": float(hi), "pairs": len(ps),
                    "higher_pairs": sum(row["higher"] for row in ps),
                    "percent": 100 * np.mean([row["higher"] for row in ps]) if ps else None,
                    "mean_other": float(np.mean(values)) if values else None,
                    "std_other": float(np.std(values)) if values else None})
    return categories, pairs, summaries, bins


def plot_results(out, summaries, bins, categories, matrices, ids, names, examples, font):
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False,
                         "font.size": 11, "pdf.fonttype": 42, "svg.fonttype": "path"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for column, direction in enumerate(DIRECTIONS):
        for row_no, metric in enumerate(["category_percent", "pair_percent"]):
            ax = axes[row_no, column]
            for k, (method, label) in enumerate(METHODS.items()):
                rows = [next(r for r in summaries if r["method"] == method and r["direction"] == direction
                             and r["group"] == group) for group in ["all", "object", "background"]]
                x = np.arange(3) + (k - 1.5) * .19
                bars = ax.bar(x, [r[metric] for r in rows], .18, color=COLORS[k],
                              edgecolor="#56616b", linewidth=.4, label=label)
                for bar, row in zip(bars, rows):
                    text = f"{row['categories_with_higher']}개" if row_no == 0 else f"{row[metric]:.1f}"
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (1 if row_no == 0 else .3),
                            text, ha="center", fontsize=8)
            ax.set(xticks=np.arange(3),
                   xticklabels=(["전체 171개", "물체 80개", "배경 91개"] if row_no == 0 else
                                ["전체\n29,070쌍", "물체 기준\n13,600쌍", "배경 기준\n15,470쌍"]),
                   title=DIRECTIONS[direction] if row_no == 0 else "",
                   ylim=(0, 105 if row_no == 0 else 30),
                   ylabel="더 높은 다른 상대가 있는 범주 (%)" if row_no == 0 else "같은 범주 점수를 넘은 개별 쌍 (%)")
            ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.suptitle("기존 대표 특징 고정 · 나머지 170개 범주와 비교")
    fig.tight_layout(rect=(0, .1, 1, .96))
    save_plot(fig, out, "01_ranking_comparison")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, direction in zip(axes, DIRECTIONS):
        for k, (method, label) in enumerate(METHODS.items()):
            rows = [r for r in bins if r["method"] == method and r["direction"] == direction]
            ax.plot(np.arange(len(rows)), [r["percent"] for r in rows], marker=MARKERS[k], color=COLORS[k],
                    linestyle=LINE_STYLES[k], markeredgecolor="#56616b", label=label, linewidth=2)
            ax.set(xticks=np.arange(len(rows)), xticklabels=[f"{r['left']:.1f}–{r['right']:.1f}\nn={r['pairs']}" for r in rows])
        ax.set(title=DIRECTIONS[direction], xlabel="원본 이미지 주석 상관 구간",
               ylabel="같은 범주 점수를 넘은 다른 범주 쌍 (%)", ylim=(-2, 102))
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, .1, 1, 1))
    save_plot(fig, out, "02_cooccurrence_bins")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, direction in zip(axes, DIRECTIONS):
        for k, (method, label) in enumerate(METHODS.items()):
            values, labels = [], []
            for lo, hi in zip([0, .5, .6, .7, .8, .9], [.5, .6, .7, .8, .9, 1.01]):
                rows = [r for r in categories if r["method"] == method and r["direction"] == direction
                        and r["min_auroc_val"] is not None and lo <= r["min_auroc_val"] < hi]
                values.append(100 * np.mean([r["any_higher"] for r in rows]) if rows else np.nan)
                labels.append(("<0.5" if lo == 0 else f"{lo:.1f}–{min(hi, 1):.1f}") + f"\nn={len(rows)}")
            ax.plot(range(6), values, marker=MARKERS[k], color=COLORS[k], linestyle=LINE_STYLES[k],
                    markeredgecolor="#56616b", label=label)
            ax.set(xticks=range(6), xticklabels=labels)
        ax.set(title=DIRECTIONS[direction], xlabel="기준 범주의 검증 AUROC\n이미지·텍스트 중 작은 값 · 구간별 범주 수 n",
               ylabel="더 높은 다른 상대가 있는 범주 (%)", ylim=(-2, 105))
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, .1, 1, 1))
    save_plot(fig, out, "03_auroc_strata")
    lookup = {name: ids.index(cid) for cid, name in names.items()}
    a, b = (lookup[name] for name in examples[0])
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6))
    for ax, (method, label) in zip(axes, METHODS.items()):
        m = matrices[method][np.ix_([a, b], [a, b])]
        ax.imshow(m, cmap="Blues", vmin=min(0, float(m.min())), vmax=max(float(m.max()), .01))
        for row in range(2):
            for col in range(2):
                ax.text(col, row, f"{m[row, col]:.3f}", ha="center", va="center",
                        color="white" if m[row, col] > .6*m.max() else "black", fontsize=13)
        ax.set(xticks=[0, 1], xticklabels=examples[0], yticks=[0, 1], yticklabels=examples[0],
               title=label, xlabel="텍스트 범주", ylabel="이미지 범주")
    fig.suptitle("눈·스키 사례 · 상관계수와 회귀 계수의 절댓값은 직접 비교 불가")
    fig.tight_layout(rect=(0, 0, 1, .94))
    save_plot(fig, out, "04_snow_skis_example")


def save_plot(fig, out, name):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    root = (args.config.parent / cfg["run"]).resolve()
    out = (args.config.parent / cfg["output"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stages = ["load", "tune", "fit", "evaluate", "sensitivity", "figures"]
    with ProgressReporter(out, stages, [], cfg["progress_interval_seconds"]):
        with stage_progress("load"):
            split, test_split = cfg["fit_split"], cfg["test_split"]
            raw_i = sparse.load_npz(root / "activations" / split / "image.npz").tocsr()
            raw_t = sparse.load_npz(root / "activations" / split / "text.npz").tocsr()
            parents = np.load(root / "index" / split / "parents.npy")
            image_ids, text_ids = varying_columns(raw_i), varying_columns(raw_t)
            ni = len(image_ids)
            x, y = raw_i[parents][:, image_ids], raw_t[:, text_ids]
            order = np.random.default_rng(cfg["seed"]).permutation(raw_i.shape[0])
            tune_images = order[:round(len(order) * cfg["tune_image_fraction"])]
            is_tune = np.isin(parents, tune_images)
            np.savez_compressed(out / "image_split.npz", fit_image_rows=order[len(tune_images):],
                                tune_image_rows=tune_images)
            fit, validation = sparse_moments(x[~is_tune], y[~is_tune]), sparse_moments(x[is_tune], y[is_tune])
            full = sparse_moments(x, y)
            test_parents = np.load(root / "index" / test_split / "parents.npy")
            test_i = sparse.load_npz(root / "activations" / test_split / "image.npz").tocsr()
            test_t = sparse.load_npz(root / "activations" / test_split / "text.npz").tocsr()
            heldout = sparse_moments(test_i[test_parents][:, image_ids], test_t[:, text_ids])
            reps = json.loads((root / "representatives.json").read_text())
            ids = sorted(int(cid) for cid in reps)
            names = {int(c["id"]): c["name"] for c in json.loads((root / "dataset.json").read_text())["concepts"]}
            auc = {(int(r["concept_id"]), r["side"]): float(r["auroc"]) if r["auroc"] else None
                   for r in load_rows(root / "representatives.csv") if r["split"] == test_split}
            labels, coverage = labels_and_coverage(root, split, ids)
        with stage_progress("tune"):
            selected, tuning = tune(fit, validation, ni, cfg["penalties"])
            write_csv(out / "penalty_tuning.csv", tuning)
            atomic_json(out / "selected_penalties.json", selected)
        with stage_progress("fit"):
            ii, it, tt = blocks(full.standardized(full), ni)
            matrices = score_matrices(ii, it, tt, selected["image_to_text"], selected["text_to_image"])
            residuals = {}
            for name, xx, cross, penalty, coefficient in [
                ("image_to_text", ii, it, selected["image_to_text"], matrices["ridge_image_to_text"]),
                ("text_to_image", tt, it.T, selected["text_to_image"], matrices["ridge_text_to_image"].T),
            ]:
                residual = linalg.get_blas_funcs(("gemm",), dtype=np.float64)[0](
                    alpha=1, a=xx + penalty*np.eye(len(xx)), b=coefficient) - cross
                residuals[name] = float(np.max(np.abs(residual)))
                assert residuals[name] < 1e-9
            atomic_json(out / "normal_equation_verification.json", residuals)
            panel = np.load(root / "panel.npz")
            np.testing.assert_allclose(it, panel["C"][np.ix_(image_ids, text_ids)], atol=1e-7)
            np.savez_compressed(out / "scores.npz", image_feature_ids=image_ids, text_feature_ids=text_ids,
                                mean=full.mean, scale=full.scale, **matrices)
            vi, vit, vt = blocks(heldout.standardized(full), ni)
            prediction = []
            for direction, b, xx, xy, yy in [
                ("image_to_text", matrices["ridge_image_to_text"], vi, vit, vt),
                ("text_to_image", matrices["ridge_text_to_image"].T, vt, vit.T, vi),
            ]:
                mse = prediction_mse(b, xx, xy, yy)
                baseline = np.diag(yy)
                prediction.append({"direction": direction, "split": test_split,
                    "mean_standardized_mse": float(mse.mean()), "mean_intercept_only_mse": float(baseline.mean()),
                    "relative_mse_reduction": float(1 - mse.sum()/baseline.sum())})
            write_csv(out / "heldout_prediction.csv", prediction)
        with stage_progress("evaluate"):
            category_scores = {m: category_matrix(s, reps, ids, image_ids, text_ids) for m, s in matrices.items()}
            for name, s in category_scores.items():
                if not np.isfinite(s).all():
                    raise ValueError(f"Original 171-category comparison contains undefined scores: {name}")
            cats, pairs, summary, bins = summarize(category_scores, ids, names, labels, coverage, auc,
                                                  split, cfg["annotation_bin_width"])
            write_csv(out / "per_category.csv", cats)
            write_csv(out / "per_pair.csv", pairs)
            write_csv(out / "summary.csv", summary)
            write_csv(out / "annotation_bins.csv", bins)
            # Independent vectorized check of the row/column denominators and all transitions.
            checks = []
            for direction in DIRECTIONS:
                oriented = {m: s if direction == "image" else s.T for m, s in category_scores.items()}
                _, raw_higher = ranking_flags(oriented["pearson"])
                for method, s in oriented.items():
                    valid, higher = ranking_flags(s)
                    result = next(r for r in summary if r["direction"] == direction
                                  and r["method"] == method and r["group"] == "all")
                    assert result["higher_pairs"] == int(higher.sum())
                    assert result["categories_with_higher"] == int(higher.any(axis=1).sum())
                    assert result["pairs"] == int(valid.sum()) == len(ids)*(len(ids)-1)
                    assert result["resolved_pairs"] == int((raw_higher & ~higher).sum())
                    assert result["introduced_pairs"] == int((~raw_higher & higher).sum())
                    checks.append({"method": method, "direction": direction, "verified": True})
            atomic_json(out / "ranking_verification.json", checks)
            raw_scores = category_scores["pearson"]
            shrink = (1 / (1 + selected["image_to_text"]) + 1 / (1 + selected["text_to_image"])) / 2
            for s in [raw_scores, raw_scores.T]:
                np.testing.assert_array_equal(s > np.diag(s)[:, None],
                                              s*shrink > np.diag(s*shrink)[:, None])
            examples = []
            lookup = {name: cid for cid, name in names.items()}
            for a, b in cfg["examples"]:
                ac, bc = lookup[a], lookup[b]
                for r in pairs:
                    if {r["anchor_id"], r["candidate_id"]} == {ac, bc}:
                        examples.append(r)
            write_csv(out / "prespecified_examples.csv", examples)
            # Independent score-estimation replication, not prediction by the frozen train map.
            hi, hit, ht = blocks(heldout.standardized(heldout), ni)
            replicated = score_matrices(hi, hit, ht, selected["image_to_text"], selected["text_to_image"])
            valid_i, valid_t = heldout.variance[:ni] > 0, heldout.variance[ni:] > 0
            valid = valid_i[:, None] & valid_t[None, :]
            for matrix in replicated.values():
                matrix[~valid] = np.nan
            cs = {m: category_matrix(s, reps, ids, image_ids, text_ids) for m, s in replicated.items()}
            rep_cats, rep_pairs, replicated_summary, _ = summarize(
                cs, ids, names, labels, coverage, auc,
                test_split + "_separate_estimation", cfg["annotation_bin_width"])
            write_csv(out / "heldout_score_replication.csv", replicated_summary)
            write_csv(out / "heldout_categories.csv", rep_cats)
            example_ids = [{lookup[a], lookup[b]} for a, b in cfg["examples"]]
            write_csv(out / "heldout_examples.csv", [r for r in rep_pairs
                if {r["anchor_id"], r["candidate_id"]} in example_ids])
        with stage_progress("sensitivity"):
            sensitivity = []
            forward, backward = RidgePath.from_moments(ii, it), RidgePath.from_moments(tt, it.T)
            for penalty in iter_progress(cfg["penalties"], "Penalty sensitivity", unit="penalties"):
                avg = (forward.coefficients(penalty) + backward.coefficients(penalty).T)/2
                cm = category_matrix(avg, reps, ids, image_ids, text_ids)
                _, _, ss, _ = summarize({"pearson": raw_scores, "ridge_bidirectional_mean": cm}, ids, names,
                                        labels, coverage, auc, split, cfg["annotation_bin_width"])
                sensitivity.extend(dict(r, common_penalty=penalty) for r in ss if r["method"] != "pearson")
            write_csv(out / "penalty_sensitivity.csv", sensitivity)
        with stage_progress("figures"):
            plot_results(out, summary, bins, cats, category_scores, ids, names, cfg["examples"], cfg["font"])
            inputs = [root / "representatives.json", root / "representatives.csv", root / "panel.npz"]
            for sp in [split, test_split]:
                inputs += [root / "activations" / sp / f"{side}.npz" for side in ["image", "text"]]
                inputs += [root / "index" / sp / "parents.npy"]
            inputs += [root / "index" / split / name for name in ["presence.npy", "mentions.npy", "concept_ids.json"]]
            atomic_json(out / "manifest.json", {"config": cfg, "source_run": root,
                "selected_penalties": selected, "images": raw_i.shape[0], "captions": raw_t.shape[0],
                "image_features": len(image_ids), "text_features": len(text_ids),
                "tune_images": len(tune_images), "tune_captions": int(is_tune.sum()),
                "fit_images": len(order)-len(tune_images), "fit_captions": int((~is_tune).sum()),
                "test_images": test_i.shape[0], "test_captions": test_t.shape[0],
                "penalty_selection": "Independent per direction; mean standardized prediction MSE on image-held-out train split",
                "score_estimation": "Refit on all train2017 after prediction-only tuning; representative IDs fixed",
                "replication": "val2017 coefficients re-estimated with fixed penalties and representatives; not frozen-map prediction",
                "unit": "Each image-caption pair has equal weight; zero activations retained",
                "semantic_labels": "Existing masking-AUROC representatives, not human-verified semantic ground truth",
                "score_units": "Pearson and standardized ridge coefficients are not numerically commensurate",
                "shrink_control_ranks_identical": True,
                "inputs": {str(p): sha256(p) for p in inputs},
                "code": {str(p): sha256(p) for p in [Path(__file__), Path(__file__).parents[1]/"src/mm_sae/metrics/regression.py", args.config]}})
    print(json.dumps({"output": str(out), "penalties": selected,
                      "overall": [r for r in summary if r["group"] == "all"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
