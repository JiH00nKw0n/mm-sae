"""Fit and serialize a coordinate-preserving cross-modal mapping."""

import numpy as np
from .direct import fit_in_image_coordinates, matched_features
from mm_sae.data.paired import ChunkedMatrix
from .projection import fit_common_projection
from .sparse_pls import fit_sparse_pls
from .sparse_cca import fit_sparse_cca
from .rotation import row_normalize, rotate


def project(active, weights, mean, inv):
    scaled = inv[:, None] * weights
    return np.asarray(active @ scaled - mean @ scaled, dtype=np.float32)


def prune(weights, k):
    if k >= len(weights):
        return weights.copy()
    indices = np.argsort(-np.abs(weights), axis=0, kind="stable")[:k]
    result = np.zeros_like(weights)
    np.put_along_axis(result, indices, np.take_along_axis(weights, indices, axis=0), axis=0)
    return result


def fit(stats, cfg, cache, saes, batch):
    if cfg.feature_scope == "all_coordinates":
        if cfg.method not in ("hungarian", "procrustes", "sinkhorn"):
            raise ValueError("Only direct coordinate mappings support all_coordinates")
        if stats["xy"].shape != (cfg.rank, cfg.rank):
            raise ValueError("Full coordinate alignment requires equal image, text and output widths")
        ix = iy = np.arange(cfg.rank)
    else:
        ix, iy = [np.flatnonzero(stats[k + "_inv"]) for k in ("image", "text")]
    xx = stats["xx"][np.ix_(ix, ix)]
    yy = stats["yy"][np.ix_(iy, iy)]
    cross = stats["xy"][np.ix_(ix, iy)]
    r = cfg.rank
    if r > min(cross.shape) and not (
        cfg.method in ("sinkhorn", "procrustes") and cfg.feature_scope == "all_active"
    ):
        raise ValueError(f"rank {r} exceeds active widths {cross.shape}")
    meta = {"method": cfg.method}
    if cfg.method == "hungarian":
        i, j = matched_features(cross, r)
        if cfg.feature_scope == "all_coordinates":
            order = np.argsort(i)
            i, j = i[order], j[order]
        a, b = np.zeros((len(ix), r)), np.zeros((len(iy), r))
        a[i, np.arange(r)] = 1
        b[j, np.arange(r)] = 1
        meta.update(
            image_ids=ix[i].tolist(),
            text_ids=iy[j].tolist(),
            selection=(
                "full permutation"
                if cfg.feature_scope == "all_coordinates"
                else "top correlations in full assignment"
            ),
        )
    elif cfg.method in ("sinkhorn", "procrustes"):
        a, b, meta["solver"] = fit_in_image_coordinates(cross, cfg)
    else:
        if cfg.method == "sparse_cca":
            result = fit_sparse_cca(
                xx,
                cross,
                yy,
                r,
                k=cfg.support,
                ridge=cfg.ridge,
                max_iter=cfg.max_iter,
                tolerance=cfg.tolerance,
            )
        else:
            result = fit_common_projection(xx, cross, yy, r, whiten=cfg.method == "cca", ridge=cfg.ridge)
            if cfg.method == "sparse_pls":
                result = fit_sparse_pls(
                    cross,
                    result.image,
                    result.text,
                    k=cfg.support,
                    max_iter=cfg.max_iter,
                    tolerance=cfg.tolerance,
                )
        a, b = result.image, result.text
        meta["solver"] = result.metadata
    rot = np.eye(r)
    if cfg.rotation != "none":
        left, right = row_normalize(xx @ a), row_normalize(yy @ b)
        rot, meta["covariance_rotation"] = rotate(left, right, "covariance", cfg.max_iter, cfg.tolerance)
        if cfg.rotation == "squared_score_covariance":
            scores = []
            rows = None
            for side, idx, coeff in (("image", ix, a), ("text", iy, b)):
                table = ChunkedMatrix(cache, side)
                try:
                    if rows is None:
                        rows = np.sort(
                            np.random.default_rng(cfg.seed).choice(
                                len(table), min(len(table), cfg.score_pairs), replace=False
                            )
                        )
                    chunks = []
                    for start in range(0, len(rows), batch):
                        act = saes.activations(side, table[rows[start : start + batch]])[:, idx]
                        chunks.append(
                            project(act, coeff @ rot, stats[side + "_mean"][idx], stats[side + "_inv"][idx])
                        )
                    scores.append(np.concatenate(chunks))
                finally:
                    table.close()
            delta, meta["score_rotation"] = rotate(
                *scores, "squared_score_covariance", cfg.max_iter, cfg.tolerance
            )
            rot = rot @ delta
    arrays = {"rotation": rot}
    for side, idx, coeff in (("image", ix, a), ("text", iy, b)):
        arrays[side + "_mean"] = stats[side + "_mean"]
        arrays[side + "_inv"] = stats[side + "_inv"]
        full = np.zeros((len(arrays[side + "_mean"]), r))
        full[idx] = coeff
        arrays[side + "_base"] = full
        rotated = full @ rot if cfg.rotation != "none" else full
        arrays[side] = prune(rotated, cfg.prune) if cfg.prune else rotated
    return arrays, meta


def scores(saes, mapping, side, embeddings, batch_size, path=None, *, feature_budget=None):
    shared = "shared_width" in mapping
    if shared:
        from scipy import sparse

        return sparse.vstack(
            [
                saes.activations(side, embeddings[i : i + batch_size])
                for i in range(0, len(embeddings), batch_size)
            ],
            format="csr",
        )
    width = mapping[side].shape[1]
    shape = (len(embeddings), width)
    result = (
        np.empty(shape, dtype=np.float32)
        if path is None
        else np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    )
    weights = prune(mapping[side], feature_budget) if feature_budget is not None else mapping[side]
    coeff = mapping[side + "_inv"][:, None] * weights
    offset = mapping[side + "_mean"] @ coeff
    for start in range(0, len(embeddings), batch_size):
        sl = slice(start, start + batch_size)
        codes = saes.activations(side, embeddings[sl])
        result[sl] = codes @ coeff - offset
    return result
