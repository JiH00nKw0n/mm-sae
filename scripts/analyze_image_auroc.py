"""Describe fixed image representatives by category type and removal AUROC."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.stats import mannwhitneyu, spearmanr

from mm_sae.io import atomic_json, sha256, write_csv

BIN_LABELS = ["0.5 미만", "0.5–0.6", "0.6–0.7", "0.7–0.8", "0.8–0.9", "0.9–1.0"]
COLORS = {"object": "#FFD6A5", "background": "#9BF6FF"}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def auc_bin(value):
    return int(np.searchsorted([.5, .6, .7, .8, .9], value, side="right"))


def analyze_split(run, split, names, representatives, selection):
    index = run / "index" / split
    ids = json.loads((index / "concept_ids.json").read_text())
    presence = np.load(index / "presence.npy", mmap_mode="r")
    areas = np.load(index / "areas.npy", mmap_mode="r")
    original = sparse.load_npz(run / "activations" / split / "image.npz").tocsc()
    shared = Counter(r["image"] for r in representatives.values())
    assert not np.any(original.data < 0)
    records = []
    for col, cid in enumerate(ids):
        feature = representatives[str(cid)]["image"]
        selected = selection[split, cid]
        assert int(selected["feature"]) == feature
        folder = run / "counterfactual" / split / str(cid)
        rows = np.load(folder / "image_rows.npy")
        # Both pools use the same concept-present images, with one edited copy each.
        assert np.array_equal(np.sort(rows), np.flatnonzero(presence[:, col]))
        full = original.getcol(feature).toarray().ravel()
        before = full[rows]
        after = sparse.load_npz(folder / "image_activations.npz").getcol(feature).toarray().ravel()
        assert len(rows) == int(selected["n_pairs"]) == len(after) and len(rows) > 0
        assert not np.any(after < 0)
        auc = float(selected["auroc"])
        recomputed = float(mannwhitneyu(before, after, method="asymptotic").statistic) / len(rows)**2
        assert np.isclose(auc, recomputed, rtol=0, atol=1e-12), (cid, auc, recomputed)
        p, q = float(np.mean(before > 0)), float(np.mean(after > 0))
        ceiling = .5 + .5 * p
        tight_ceiling = p + .5 * (1-p) * (1-q)
        assert auc <= tight_ceiling + 1e-12 <= ceiling + 2e-12
        records.append(dict(
            split=split, category_id=cid, name=names[cid],
            kind="object" if cid < 91 else "background", feature=feature,
            shared_category_count=shared[feature], auroc=auc, bin_index=auc_bin(auc),
            n_images=len(rows), image_prevalence=float(len(rows) / len(full)),
            original_active_given_category=p, masked_active_given_category=q,
            global_original_active=float(np.mean(full > 0)),
            paired_decrease_fraction=float(np.mean(before > after)),
            paired_both_zero_fraction=float(np.mean((before == 0) & (after == 0))),
            turned_off_given_original_active=(float(np.mean(after[before > 0] == 0)) if p else None),
            mean_mask_area=float(np.mean(areas[rows, col])),
            median_mask_area=float(np.median(areas[rows, col])),
            auroc_ceiling_from_original_firing=ceiling,
            auroc_ceiling_from_both_firing=tight_ceiling,
            gap_to_firing_ceiling=ceiling-auc,
        ))
    return records


def aggregate(records):
    bins, summaries, associations = [], [], []
    for split in sorted({r["split"] for r in records}):
        rows = [r for r in records if r["split"] == split]
        assert len(rows) == 171
        for index, label in enumerate(BIN_LABELS):
            selected = [r for r in rows if r["bin_index"] == index]
            obj = sum(r["kind"] == "object" for r in selected)
            bg = len(selected) - obj
            bins.append(dict(split=split, bin_index=index, interval=label,
                             object_count=obj, background_count=bg, total=len(selected),
                             object_fraction=obj/len(selected) if selected else None,
                             background_fraction=bg/len(selected) if selected else None))
        for kind in ["all", "object", "background"]:
            group = [r for r in rows if kind == "all" or r["kind"] == kind]
            summaries.append(dict(split=split, kind=kind, categories=len(group),
                median_auroc=float(np.median([r["auroc"] for r in group])),
                mean_auroc=float(np.mean([r["auroc"] for r in group])),
                auroc_ge_07=sum(r["auroc"] >= .7 for r in group),
                median_original_active_given_category=float(np.median([
                    r["original_active_given_category"] for r in group])),
                median_mask_area=float(np.median([r["median_mask_area"] for r in group])),
                median_n_images=float(np.median([r["n_images"] for r in group])),
                categories_fewer_than_30_images=sum(r["n_images"] < 30 for r in group),
            ))
            for metric in ["original_active_given_category", "masked_active_given_category",
                           "global_original_active", "mean_mask_area", "median_mask_area",
                           "n_images", "shared_category_count"]:
                rho = cast(float, spearmanr([r["auroc"] for r in group], [r[metric] for r in group])[0])
                associations.append(dict(split=split, kind=kind, metric=metric,
                                         categories=len(group), spearman=float(rho)))
    return bins, summaries, associations


def save_figure(fig, out, name):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out / f"{name}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def draw(bins, records, out, font):
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "svg.fonttype": "path", "font.size": 12})
    rows = [r for r in bins if r["split"] == "val2017"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(rows))
    obj = np.array([r["object_count"] for r in rows])
    bg = np.array([r["background_count"] for r in rows])
    ax.bar(x-.18, obj, color=COLORS["object"], width=.34, label="물체 80개 범주")
    ax.bar(x+.18, bg, color=COLORS["background"], width=.34, label="배경 91개 범주")
    for i in range(len(rows)):
        ax.text(i-.18, obj[i]+1, str(obj[i]), ha="center", fontsize=11)
        ax.text(i+.18, bg[i]+1, str(bg[i]), ha="center", fontsize=11)
    ax.set(xticks=x, xticklabels=BIN_LABELS, ylabel="범주 수", xlabel="이미지 대표 특징의 검증 AUROC",
           ylim=(0, max(max(obj), max(bg))*1.16), title="AUROC 구간별 물체와 배경의 분포")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.1, .04, "학습 자료에서 고른 특징을 고정하고 검증 이미지에서 평가했습니다. 각 범주를 한 번씩 집계했습니다.", fontsize=10)
    fig.text(.1, .005, "각 구간은 하한을 포함하고 상한을 제외합니다. 마지막 구간에는 1.0을 포함합니다.", fontsize=10)
    fig.tight_layout(rect=(0, .09, 1, 1))
    save_figure(fig, out, "image_auroc_by_category_type")

    fig, ax = plt.subplots(figsize=(8, 6))
    for kind, label in [("object", "물체"), ("background", "배경")]:
        group = [r for r in records if r["split"] == "val2017" and r["kind"] == kind]
        ax.scatter([r["original_active_given_category"] for r in group], [r["auroc"] for r in group],
                   color=COLORS[kind], edgecolor="#58616A", linewidth=.5, s=40, label=label)
    ax.plot([0, 1], [.5, 1], color="#61676D", linestyle="--", label="활성 빈도로 결정되는 AUROC 상한")
    ax.set(xlim=(-.02, 1.02), ylim=(.44, 1.02), xlabel="범주가 있는 원본 이미지에서 특징이 켜진 비율",
           ylabel="이미지 대표 특징의 검증 AUROC", title="낮은 활성 빈도에 따른 AUROC 제한")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.1, .015, "활성값이 음수가 아니므로 AUROC는 0.5 + 0.5 × 원본 활성 비율을 넘을 수 없습니다.", fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, 1))
    save_figure(fig, out, "image_auroc_and_firing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--font", default="AppleGothic")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = {c["id"]: c["name"] for c in json.loads((args.run / "dataset.json").read_text())["concepts"]}
    representatives = json.loads((args.run / "representatives.json").read_text())
    selection = {(r["split"], int(r["concept_id"])): r
                 for r in read_csv(args.run / "representatives.csv") if r["side"] == "image"}
    records = []
    for split in ["train2017", "val2017"]:
        records.extend(analyze_split(args.run, split, names, representatives, selection))
        print(f"{split}: verified cached AUROC and analyzed all 171 categories", flush=True)
    bins, summaries, associations = aggregate(records)
    for filename, rows in [("categories.csv", records), ("bins.csv", bins),
                           ("type_summary.csv", summaries), ("associations.csv", associations)]:
        write_csv(args.out / filename, rows)
    draw(bins, records, args.out, args.font)
    atomic_json(args.out / "manifest.json", dict(
        source=str(args.run.resolve()), script_sha256=sha256(Path(__file__)),
        representatives_sha256=sha256(args.run / "representatives.json"),
        selection_sha256=sha256(args.run / "representatives.csv"),
        primary_split="val2017", selection_split="train2017", counting_unit="category",
        auroc="P(original > masked) + 0.5 P(original = masked), independent draws from two pools",
        recomputation_check="All 342 cached AUROCs matched direct Mann-Whitney U within 1e-12",
        inference="Descriptive only; categories sharing features are not independent replicates",
    ))


if __name__ == "__main__":
    main()
