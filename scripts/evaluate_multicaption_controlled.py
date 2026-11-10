#!/usr/bin/env python3
"""Full controlled COCO multi-caption audit over frozen SAE mappings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from mm_sae.io import sha256
from mm_sae.sae.inference import FrozenSAEs


KS = (1, 2, 3, 5)
CONTROLS = ("same_image", "duplicate", "cross_image")
EXPECTED_METHODS = {
    "hungarian",
    "pls",
    "cca",
    "sparse_pls",
    "sparse_cca",
    "cca_cov",
    "cca_cov64",
    "sinkhorn",
    "procrustes",
}


def jsonlines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)


def derangement(rng: np.random.Generator, size: int) -> np.ndarray:
    original = np.arange(size)
    for _ in range(100):
        value = rng.permutation(size)
        if np.all(value != original):
            return value
    shift = int(rng.integers(1, size))
    return (original + shift) % size


def caption_sets(
    caption_groups: list[list[int]], seeds: int
) -> dict[tuple[int, int, str], np.ndarray]:
    count = len(caption_groups)
    result = {}
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        ordered = [rng.permutation(rows) for rows in caption_groups]
        for k in KS:
            same = np.stack([rows[:k] for rows in ordered])
            duplicate = np.repeat(same[:, :1], k, axis=1)
            cross = [same[:, :1]]
            for column in range(1, k):
                donors = derangement(rng, count)
                cross.append(np.asarray([ordered[donor][column % len(ordered[donor])] for donor in donors])[:, None])
            result[(seed, k, "same_image")] = same
            result[(seed, k, "duplicate")] = duplicate
            result[(seed, k, "cross_image")] = np.concatenate(cross, axis=1)
    return result


def mean_embeddings(embeddings: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return normalize(np.asarray(embeddings[indices].mean(axis=1), dtype=np.float32))


def mean_sparse_rows(matrix: sparse.csr_matrix, indices: np.ndarray) -> sparse.csr_matrix:
    result = matrix[indices[:, 0]].astype(np.float32)
    for column in range(1, indices.shape[1]):
        result = result + matrix[indices[:, column]]
    if indices.shape[1] > 1:
        result = result.multiply(1.0 / indices.shape[1])
    return result.tocsr()


def all_activations(saes: FrozenSAEs, side: str, embeddings: np.ndarray, batch: int) -> sparse.csr_matrix:
    return sparse.vstack(
        [saes.activations(side, embeddings[start : start + batch]) for start in range(0, len(embeddings), batch)],
        format="csr",
    )


def project(codes: sparse.csr_matrix, mapping: dict[str, np.ndarray], side: str) -> np.ndarray:
    weights = mapping[side]
    coeff = mapping[side + "_inv"][:, None] * weights
    offset = mapping[side + "_mean"] @ coeff
    return np.asarray(codes @ coeff - offset, dtype=np.float32)


def method_label(config: dict) -> str:
    method = config["method"]
    if method == "cca" and config.get("rotation") == "covariance":
        return "cca_cov64" if config.get("prune") == 64 else "cca_cov"
    return method


def load_mappings(root: Path) -> dict[str, tuple[Path, dict]]:
    found = {}
    for receipt_path in root.glob("*/fit.json"):
        receipt = json.loads(receipt_path.read_text())
        weights = receipt_path.parent / "mapping.npz"
        if sha256(weights) != receipt["sha256"]:
            raise ValueError(f"Mapping hash mismatch: {weights}")
        label = method_label(receipt["config"])
        if label in EXPECTED_METHODS:
            if label in found:
                raise ValueError(f"Duplicate mapping label: {label}")
            found[label] = (weights, receipt)
    missing = EXPECTED_METHODS - set(found)
    if missing:
        raise ValueError(f"Missing independent-SAE mappings: {sorted(missing)}")
    return found


def diagonal_direction(q: torch.Tensor, c: torch.Tensor, block: int = 128):
    size = q.shape[0]
    ranks, positives, margins = [], [], []
    columns = torch.arange(size, device=q.device)
    for start in range(0, size, block):
        stop = min(start + block, size)
        rows = torch.arange(start, stop, device=q.device)
        similarities = q[start:stop] @ c.T
        local = torch.arange(stop - start, device=q.device)
        positive = similarities[local, rows]
        rank = 1 + (similarities > positive[:, None]).sum(1)
        rank += ((similarities == positive[:, None]) & (columns[None, :] < rows[:, None])).sum(1)
        similarities[local, rows] = -torch.inf
        hardest = similarities.max(1).values
        ranks.append(rank.cpu().numpy().astype(np.int32))
        positives.append(positive.cpu().numpy().astype(np.float32))
        margins.append((positive - hardest).cpu().numpy().astype(np.float32))
    return np.concatenate(ranks), np.concatenate(positives), np.concatenate(margins)


def controlled_metrics(image: np.ndarray, text: np.ndarray, device: str):
    q = torch.from_numpy(normalize(image)).to(device)
    c = torch.from_numpy(normalize(text)).to(device)
    image_ranks, positive, margin = diagonal_direction(q, c)
    text_ranks, _, _ = diagonal_direction(c, q)
    summary = {}
    for direction, ranks in (("image_to_text", image_ranks), ("text_to_image", text_ranks)):
        for k in (1, 5, 10):
            summary[f"{direction}_R@{k}"] = float(np.mean(ranks <= k))
    summary["mean_R@5"] = (summary["image_to_text_R@5"] + summary["text_to_image_R@5"]) / 2
    summary["positive_cosine"] = float(positive.mean())
    summary["hardest_negative_margin"] = float(margin.mean())
    return summary, image_ranks, text_ranks, positive, margin


def original_anchor_metrics(
    image: np.ndarray, text: np.ndarray, parents: np.ndarray, device: str, block: int = 128
):
    q = torch.from_numpy(normalize(image)).to(device)
    c = torch.from_numpy(normalize(text)).to(device)
    image_count, caption_count = len(image), len(text)
    groups = [[] for _ in range(image_count)]
    for caption, parent in enumerate(parents):
        groups[int(parent)].append(caption)
    width = max(map(len, groups))
    positive_indices = np.full((image_count, width), -1, dtype=np.int64)
    positive_mask = np.zeros((image_count, width), dtype=bool)
    for row, values in enumerate(groups):
        positive_indices[row, : len(values)] = values
        positive_mask[row, : len(values)] = True
    positive_indices_t = torch.from_numpy(np.maximum(positive_indices, 0)).to(device)
    positive_mask_t = torch.from_numpy(positive_mask).to(device)
    caption_columns = torch.arange(caption_count, device=device)
    image_ranks = []
    for start in range(0, image_count, block):
        stop = min(start + block, image_count)
        similarities = q[start:stop] @ c.T
        indices = positive_indices_t[start:stop]
        values = similarities.gather(1, indices).masked_fill(~positive_mask_t[start:stop], -torch.inf)
        best_value, local_choice = values.max(1)
        best_index = indices.gather(1, local_choice[:, None]).squeeze(1)
        rank = 1 + (similarities > best_value[:, None]).sum(1)
        rank += ((similarities == best_value[:, None]) & (caption_columns[None, :] < best_index[:, None])).sum(1)
        image_ranks.append(rank.cpu().numpy().astype(np.int32))
    image_ranks = np.concatenate(image_ranks)
    image_columns = torch.arange(image_count, device=device)
    text_ranks = []
    for start in range(0, caption_count, block):
        stop = min(start + block, caption_count)
        similarities = c[start:stop] @ q.T
        target = torch.from_numpy(parents[start:stop]).to(device)
        local = torch.arange(stop - start, device=device)
        positive = similarities[local, target]
        rank = 1 + (similarities > positive[:, None]).sum(1)
        rank += ((similarities == positive[:, None]) & (image_columns[None, :] < target[:, None])).sum(1)
        text_ranks.append(rank.cpu().numpy().astype(np.int32))
    text_ranks = np.concatenate(text_ranks)
    result = {}
    for direction, ranks in (("image_to_text", image_ranks), ("text_to_image", text_ranks)):
        for k in (1, 5, 10):
            result[f"{direction}_R@{k}"] = float(np.mean(ranks <= k))
    result["mean_R@5"] = (result["image_to_text_R@5"] + result["text_to_image_R@5"]) / 2
    return result, image_ranks, text_ranks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--mapping-root", type=Path, required=True)
    parser.add_argument("--sae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    table = args.table.resolve()
    manifest = json.loads((table / "manifest.json").read_text())
    images_meta, captions_meta = jsonlines(table / "image.jsonl"), jsonlines(table / "text.jsonl")
    image_embeddings = np.load(table / "image.npy", mmap_mode="r")
    text_embeddings = np.load(table / "text.npy", mmap_mode="r")
    if (len(images_meta), len(captions_meta)) != (5_000, 25_014):
        raise ValueError("Expected the exact COCO val2017 retrieval population")
    image_groups = [row["group"] for row in images_meta]
    if len(set(image_groups)) != len(image_groups):
        raise ValueError("Expected one image per COCO group")
    lookup = {group: index for index, group in enumerate(image_groups)}
    parents = np.asarray([lookup[row["group"]] for row in captions_meta], dtype=np.int64)
    grouped = [[] for _ in image_groups]
    for caption, parent in enumerate(parents):
        grouped[int(parent)].append(caption)
    if min(map(len, grouped)) < 5:
        raise ValueError("Every controlled COCO group needs at least five captions")

    sets = caption_sets(grouped, args.seeds)
    np.savez_compressed(output / "caption_sets.npz", **{f"s{s}_k{k}_{c}": v for (s, k, c), v in sets.items()})
    selection_hash = hashlib.sha256((output / "caption_sets.npz").read_bytes()).hexdigest()
    (output / "caption_sets_manifest.json").write_text(
        json.dumps(
            {
                "seeds": args.seeds,
                "ks": KS,
                "controls": CONTROLS,
                "images": len(images_meta),
                "captions": len(captions_meta),
                "sha256": selection_hash,
            },
            indent=2,
        )
        + "\n"
    )

    saes = FrozenSAEs(args.sae.resolve(), args.device)
    image_codes = all_activations(saes, "image", image_embeddings, args.batch_size)
    caption_codes = all_activations(saes, "text", text_embeddings, args.batch_size)
    pre_codes = {}
    post_codes = {}
    activation_dir = output / "activation_sets"
    activation_dir.mkdir(exist_ok=True)
    for key, indices in sets.items():
        name = f"s{key[0]}_k{key[1]}_{key[2]}"
        pre_path, post_path = activation_dir / f"{name}_pre.npz", activation_dir / f"{name}_post.npz"
        if key[2] == "duplicate":
            # A repeated observation is the same measurement, not a fresh
            # floating-point reduction. Reuse k=1 exactly so ties cannot move.
            base = (key[0], 1, "same_image")
            pre_codes[key], post_codes[key] = pre_codes[base], post_codes[base]
            sparse.save_npz(pre_path, pre_codes[key], compressed=True)
            sparse.save_npz(post_path, post_codes[key], compressed=True)
            continue
        if pre_path.exists() and post_path.exists():
            pre_codes[key], post_codes[key] = sparse.load_npz(pre_path), sparse.load_npz(post_path)
            continue
        pre_codes[key] = all_activations(
            saes, "text", mean_embeddings(text_embeddings, indices), args.batch_size
        )
        post_codes[key] = mean_sparse_rows(caption_codes, indices)
        sparse.save_npz(pre_path, pre_codes[key], compressed=True)
        sparse.save_npz(post_path, post_codes[key], compressed=True)

    mappings = load_mappings(args.mapping_root.resolve())
    summaries = []
    for label in sorted(mappings):
        path, receipt = mappings[label]
        with np.load(path) as packed:
            mapping = dict(packed)
        image_scores = project(image_codes, mapping, "image")
        caption_scores = project(caption_codes, mapping, "text")
        anchor, anchor_i, anchor_t = original_anchor_metrics(image_scores, caption_scores, parents, args.device)
        summaries.append({"method": label, "benchmark": "original_anchor", "aggregation": "individual", "seed": -1, "k": 0, "control": "original", **anchor})
        ranks = {"anchor_image_ranks": anchor_i, "anchor_text_ranks": anchor_t}
        for key in sorted(sets):
            seed, k, control = key
            for aggregation, codes in (("clip_mean_pre_sae", pre_codes[key]), ("activation_mean_post_sae", post_codes[key])):
                text_scores = project(codes, mapping, "text")
                metrics, ir, tr, positive, margin = controlled_metrics(image_scores, text_scores, args.device)
                prefix = f"s{seed}_k{k}_{control}_{aggregation}"
                ranks[prefix + "_image_ranks"] = ir
                ranks[prefix + "_text_ranks"] = tr
                ranks[prefix + "_positive"] = positive
                ranks[prefix + "_margin"] = margin
                summaries.append(
                    {
                        "method": label,
                        "benchmark": "controlled_5000x5000",
                        "aggregation": aggregation,
                        "seed": seed,
                        "k": k,
                        "control": control,
                        **metrics,
                    }
                )
        np.savez_compressed(output / f"per_query_{label}.npz", **ranks)

    fields = [
        "method",
        "benchmark",
        "aggregation",
        "seed",
        "k",
        "control",
        "image_to_text_R@1",
        "image_to_text_R@5",
        "image_to_text_R@10",
        "text_to_image_R@1",
        "text_to_image_R@5",
        "text_to_image_R@10",
        "mean_R@5",
        "positive_cosine",
        "hardest_negative_margin",
    ]
    with (output / "summary_by_seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "state": "completed",
                "table_signature": manifest["signature"],
                "evaluator_sha256": sha256(Path(__file__).resolve()),
                "sae_hashes": saes.hashes,
                "mapping_methods": sorted(mappings),
                "seeds": args.seeds,
                "ks": KS,
                "controls": CONTROLS,
                "aggregations": ["clip_mean_pre_sae", "activation_mean_post_sae"],
                "selection_sha256": selection_hash,
                "sae_path": str(args.sae.resolve()),
                "mapping_receipts": {
                    label: {
                        "mapping_sha256": receipt["sha256"],
                        "fit_receipt": str(path.parent / "fit.json"),
                    }
                    for label, (path, receipt) in sorted(mappings.items())
                },
                "rows": len(summaries),
            },
            indent=2,
        )
        + "\n"
    )
    print(output / "run_manifest.json", flush=True)


if __name__ == "__main__":
    main()
