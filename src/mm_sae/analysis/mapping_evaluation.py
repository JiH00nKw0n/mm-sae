"""Evaluate a nonnegative correspondence without refitting representation models.

Mapping matrices always have image features in rows and text features in columns.
Regression coefficients instead have predictor features in rows and targets in
columns. Regression moments must use the training means and standard deviations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import linalg, sparse
from scipy.optimize import nnls
from scipy.sparse.csgraph import connected_components


@dataclass
class RidgeFit:
    coefficients: np.ndarray
    converged: bool
    iterations: int
    kkt_residual: float


def _matrix(value, name: str) -> np.ndarray:
    result = np.asarray(value.toarray() if sparse.issparse(value) else value, dtype=np.float64)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite two-dimensional matrix")
    return result


def _stationarity(coefficients, gradient, support) -> float:
    """Infinity norm of the projected gradient on the allowed coefficients."""
    residual = coefficients - np.maximum(coefficients - gradient, 0)
    return float(np.max(np.abs(residual[support]), initial=0))


def fit_nonnegative_ridge(
    xx, xy, support, penalty: float, *, max_iter: int = 4000,
    tolerance: float = 1e-7, device: str = "cpu", sparse_support_limit: int = 64,
) -> RidgeFit:
    """Minimize ``.5 * E[(Y - XB)^2] + .5 * penalty * ||B||²`` per target.

    Disallowed coefficients are exactly zero and allowed coefficients are
    nonnegative. Small supports use an exact nonnegative least-squares solve.
    Larger supports use batched accelerated projected gradients on ``device``.
    Both solve the identical constrained objective and are checked with the same
    projected-gradient optimality residual. No held-out observations are used.
    The penalty multiplies average squared error, independent of sample count.
    """
    xx, xy = _matrix(xx, "xx"), _matrix(xy, "xy")
    support = np.asarray(support, dtype=bool)
    if xx.shape != (xy.shape[0], xy.shape[0]) or support.shape != xy.shape:
        raise ValueError("xx, xy, and support dimensions do not agree")
    if not np.allclose(xx, xx.T, rtol=1e-9, atol=1e-10):
        raise ValueError("xx must be symmetric")
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be finite and nonnegative")
    if max_iter < 1 or not np.isfinite(tolerance) or tolerance <= 0 or sparse_support_limit < 0:
        raise ValueError("Iteration and tolerance settings must be positive")
    gram = (xx + xx.T) / 2 + penalty * np.eye(len(xx))
    coefficients = np.zeros_like(xy)
    counts = support.sum(axis=0)
    limit = tolerance * max(1.0, float(np.max(np.abs(xy[support]), initial=0)))
    for target in np.flatnonzero((counts > 0) & (counts <= sparse_support_limit)):
        indices = np.flatnonzero(support[:, target])
        local = gram[np.ix_(indices, indices)]
        cross = xy[indices, target]
        try:
            lower = linalg.cholesky(local, lower=True)
            design = lower.T
            response = linalg.solve_triangular(lower, cross, lower=True)
        except linalg.LinAlgError:
            values, vectors = linalg.eigh(local)
            if values.min() < -1e-9:
                raise ValueError("xx must be positive semidefinite") from None
            keep = values > np.finfo(float).eps * max(1.0, float(values.max())) * len(values)
            if not np.allclose(vectors[:, ~keep].T @ cross, 0, atol=limit):
                raise ValueError("Singular moments have an inconsistent cross moment") from None
            if not np.any(keep):
                continue
            design = np.sqrt(values[keep])[:, None] * vectors[:, keep].T
            response = (vectors[:, keep].T @ cross) / np.sqrt(values[keep])
        solution, _ = nnls(design, response, maxiter=max(max_iter, 3 * len(indices)), atol=limit)
        coefficients[indices, target] = solution

    large = np.flatnonzero(counts > sparse_support_limit)
    iterations = 0
    if len(large):
        import torch

        # Remove predictor rows absent from every target in this batch.
        used = np.flatnonzero(support[:, large].any(axis=1))
        reduced = gram[np.ix_(used, used)]
        values = linalg.eigvalsh(reduced)
        if values.min() < -1e-9:
            raise ValueError("xx must be positive semidefinite")
        lipschitz = float(values[-1])
        if lipschitz <= 0:
            if np.any(xy[np.ix_(used, large)] > limit):
                raise ValueError("Zero predictor moments have an inconsistent cross moment")
        else:
            with torch.no_grad():
                a = torch.as_tensor(reduced, dtype=torch.float64, device=device)
                c = torch.as_tensor(xy[np.ix_(used, large)], dtype=torch.float64, device=device)
                mask = torch.as_tensor(support[np.ix_(used, large)], device=device)
                current = torch.zeros_like(c)
                extrapolated = current.clone()
                acceleration = 1.0
                for iterations in range(1, max_iter + 1):
                    gradient = a @ extrapolated - c
                    candidate = torch.clamp(extrapolated - gradient / lipschitz, min=0) * mask
                    next_acceleration = (1 + np.sqrt(1 + 4 * acceleration**2)) / 2
                    momentum = (acceleration - 1) / next_acceleration
                    # Adaptive restart avoids oscillation on correlated features.
                    if torch.sum((extrapolated - candidate) * (candidate - current)).item() > 0:
                        next_acceleration, momentum = 1.0, 0.0
                    extrapolated = candidate + momentum * (candidate - current)
                    current, acceleration = candidate, next_acceleration
                    if iterations % 10 == 0 or iterations == max_iter:
                        gradient = a @ current - c
                        residual = (current - torch.clamp(current - gradient, min=0)) * mask
                        if torch.max(torch.abs(residual)).item() <= limit:
                            break
                coefficients[np.ix_(used, large)] = current.cpu().numpy()
    residual = _stationarity(coefficients, gram @ coefficients - xy, support)
    return RidgeFit(coefficients, residual <= limit, iterations, residual)


def prediction_r2(coefficients, held_xx, held_xy, held_yy, *, held_mean_y=None) -> np.ndarray:
    """Per-target held-out R², with training standardization and a fixed intercept.

    ``held_mean_y`` is the mean held-out target after subtracting the training
    mean and dividing by the training standard deviation. Supplying it retains
    the shift in the prediction error while centering the R² denominator at the
    actual held-out mean. Omit it only when this mean is exactly zero. Constant
    held-out targets return NaN. Unmatched targets retain a zero coefficient
    column, predicting the training target mean without discarding that target.
    ``held_yy`` accepts either a second-moment matrix or its diagonal.
    """
    coefficients = _matrix(coefficients, "coefficients")
    xx, xy = _matrix(held_xx, "held_xx"), _matrix(held_xy, "held_xy")
    yy = np.asarray(held_yy, dtype=float)
    if yy.ndim == 2:
        yy = np.diag(yy)
    n_predictors, n_targets = coefficients.shape
    if xx.shape != (n_predictors, n_predictors) or xy.shape != coefficients.shape or yy.shape != (n_targets,):
        raise ValueError("Held-out moment dimensions do not agree with coefficients")
    mean = np.zeros(n_targets) if held_mean_y is None else np.asarray(held_mean_y, dtype=float)
    if mean.shape != (n_targets,) or not np.all(np.isfinite(yy)) or not np.all(np.isfinite(mean)):
        raise ValueError("held_yy and held_mean_y must contain finite target moments")
    mse = yy - 2 * np.sum(coefficients * xy, axis=0) + np.sum(coefficients * (xx @ coefficients), axis=0)
    variance = yy - mean**2
    result = np.full(n_targets, np.nan)
    valid = variance > np.finfo(float).eps * np.maximum(1, np.abs(yy)) * 32
    result[valid] = 1 - np.maximum(mse[valid], 0) / variance[valid]
    return result


def _normalise_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(values, axis=1)
    nonzero = norms > 0
    return np.divide(values, norms[:, None], out=np.zeros_like(values), where=nonzero[:, None]), nonzero


def _retrieval_direction(query, candidates, relevant, ks, chunk_size, device) -> dict[str, Any]:
    query, query_valid = _normalise_rows(query)
    candidates, candidate_valid = _normalise_rows(candidates)
    n_query, n_candidates = len(query), len(candidates)
    ranks = np.full(n_query, n_candidates + 1, dtype=np.int64)
    candidate_indices = np.arange(n_candidates)
    if device != "cpu":
        import torch

        tensor_candidates = torch.as_tensor(candidates.T, dtype=torch.float32, device=device)
    for start in range(0, n_query, chunk_size):
        stop = min(start + chunk_size, n_query)
        if device == "cpu":
            # Explicit BLAS avoids spurious NumPy floating-point flag warnings
            # on macOS without suppressing any actual invalid arithmetic.
            gemm = linalg.get_blas_funcs(("gemm",), dtype=np.float64)[0]
            scores = gemm(1.0, query[start:stop], candidates.T)
        else:
            scores = (torch.as_tensor(query[start:stop], dtype=torch.float32, device=device)
                      @ tensor_candidates).cpu().numpy()
        scores[:, ~candidate_valid] = -np.inf
        # Rank the best relevant candidate directly instead of sorting all scores.
        # Equal scores use the smaller candidate index, identically for every method.
        for offset, row in enumerate(range(start, stop)):
            positives = relevant[row]
            positives = positives[candidate_valid[positives]]
            if not query_valid[row] or not len(positives):
                continue
            positive_scores = scores[offset, positives]
            best_score = np.max(positive_scores)
            best_index = int(np.min(positives[positive_scores == best_score]))
            rank = 1 + np.count_nonzero(scores[offset] > best_score)
            rank += np.count_nonzero((scores[offset] == best_score) & (candidate_indices < best_index))
            ranks[row] = rank
    hits = {int(k): (ranks <= min(k, n_candidates)) for k in ks}
    return {
        "query_count": n_query,
        "candidate_count": n_candidates,
        "zero_norm_query_count": int((~query_valid).sum()),
        "zero_norm_query_fraction": float((~query_valid).mean()),
        "zero_norm_candidate_count": int((~candidate_valid).sum()),
        "zero_norm_candidate_fraction": float((~candidate_valid).mean()),
        "recall": {int(k): float(hit.mean()) for k, hit in hits.items()},
        "hits": hits,
        "ranks": ranks,
    }


def paired_retrieval(
    image_features, text_features, caption_parents, *, image_ids=None,
    ks=(1, 5, 10), chunk_size: int = 128, device: str = "cpu",
) -> dict[str, Any]:
    """Bidirectional cosine retrieval for representations in a common space.

    ``caption_parents`` contains image row indices unless ``image_ids`` supplies
    the corresponding external identifiers. Every caption belonging to an image
    is relevant to that image. Both image and caption query denominators remain
    fixed when a mapping produces zero vectors. Zero-norm queries fail, and
    zero-norm candidates cannot be retrieved. Returned hit vectors retain input
    order for paired, image-cluster bootstrap comparisons.

    Only one query chunk of similarities is materialized at a time. Feature
    arrays may be dense or scipy sparse, and may have signed values.
    """
    images = _matrix(image_features, "image_features")
    texts = _matrix(text_features, "text_features")
    if images.shape[1] != texts.shape[1]:
        raise ValueError("Image and text representations must share a feature dimension")
    if not len(images) or not len(texts) or chunk_size < 1:
        raise ValueError("Retrieval requires nonempty observations and a positive chunk size")
    if not ks or any(int(k) != k or k < 1 for k in ks):
        raise ValueError("ks must contain positive integer ranks")
    parents = np.asarray(caption_parents)
    if parents.shape != (len(texts),):
        raise ValueError("caption_parents must have one entry per caption")
    if image_ids is not None:
        ids = np.asarray(image_ids)
        if ids.shape != (len(images),) or len(np.unique(ids)) != len(ids):
            raise ValueError("image_ids must contain one unique identifier per image")
        lookup = {identifier: index for index, identifier in enumerate(ids.tolist())}
        try:
            parents = np.asarray([lookup[parent] for parent in parents.tolist()], dtype=np.int64)
        except KeyError as exc:
            raise ValueError("A caption parent is absent from image_ids") from exc
    elif not np.issubdtype(parents.dtype, np.integer):
        raise ValueError("caption_parents must contain integer image row indices")
    if np.any((parents < 0) | (parents >= len(images))):
        raise ValueError("caption parent indices are outside the image array")
    image_positives = [np.flatnonzero(parents == index) for index in range(len(images))]
    text_positives = [np.asarray([parent], dtype=np.int64) for parent in parents]
    return {
        "image_to_text": _retrieval_direction(images, texts, image_positives, ks, chunk_size, device),
        "text_to_image": _retrieval_direction(texts, images, text_positives, ks, chunk_size, device),
    }


def mapping_retrieval(
    image_features, text_features, mapping, caption_parents, *, image_ids=None,
    ks=(1, 5, 10), chunk_size: int = 128, device: str = "cpu",
) -> dict[str, Any]:
    """Cosine retrieval in both feature spaces under an image-by-text mapping.

    Text projection is ``text @ rownormalize(mapping).T``. Image projection is
    ``image @ rownormalize(mapping.T).T``. Row normalization divides by the sum
    of mapping weights. Each space returns both retrieval directions under the
    fixed-denominator, all-positive-caption convention of ``paired_retrieval``.
    """
    images, texts, mapping = (_matrix(image_features, "image_features"),
                             _matrix(text_features, "text_features"), _matrix(mapping, "mapping"))
    if mapping.shape != (images.shape[1], texts.shape[1]) or np.any(mapping < 0):
        raise ValueError("mapping must be nonnegative with image-by-text feature dimensions")
    row_sum, column_sum = mapping.sum(axis=1), mapping.sum(axis=0)
    forward = np.divide(mapping, row_sum[:, None], out=np.zeros_like(mapping), where=row_sum[:, None] > 0)
    backward = np.divide(mapping, column_sum[None, :], out=np.zeros_like(mapping), where=column_sum[None, :] > 0)
    spaces = {
        "text_projected_to_image": (images, texts @ forward.T),
        "image_projected_to_text": (images @ backward, texts),
    }
    result = {}
    for name, (image_values, text_values) in spaces.items():
        result[name] = paired_retrieval(image_values, text_values, caption_parents, image_ids=image_ids,
                                        ks=ks, chunk_size=chunk_size, device=device)
    return result


def mapping_structure(mapping, *, threshold: float = 0.0) -> dict[str, Any]:
    """Support, weight concentration, and connected groups after an explicit cutoff.

    An edge is present exactly when its weight is greater than ``threshold``.
    Effective degree is ``sum(weights)**2 / sum(weights**2)`` and equals zero for
    unmatched features. Connected groups contain only features with an edge.
    An isolated feature is reported as unmatched, rather than a mapping group.
    """
    mapping = _matrix(mapping, "mapping")
    if np.any(mapping < 0) or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Mapping weights and threshold must be nonnegative")
    support = mapping > threshold
    weights = np.where(support, mapping, 0)
    ni, nt = mapping.shape
    image_degree, text_degree = support.sum(axis=1), support.sum(axis=0)

    def effective(axis):
        squared_sum = (weights**2).sum(axis=axis)
        return np.divide(weights.sum(axis=axis)**2, squared_sum,
                         out=np.zeros_like(squared_sum), where=squared_sum > 0)

    adjacency = sparse.bmat([[None, sparse.csr_matrix(support)],
                            [sparse.csr_matrix(support.T), None]], format="csr")
    _, labels = connected_components(adjacency, directed=False)
    matched = np.r_[image_degree > 0, text_degree > 0]
    labels[~matched] = -1
    groups = []
    for label in np.unique(labels[matched]):
        image_indices = np.flatnonzero(labels[:ni] == label)
        text_indices = np.flatnonzero(labels[ni:] == label)
        group_weights = weights[np.ix_(image_indices, text_indices)]
        groups.append({"image_features": int(len(image_indices)), "text_features": int(len(text_indices)),
                       "edges": int(np.count_nonzero(group_weights)), "weight": float(group_weights.sum())})
    edges = int(support.sum())
    return {
        "threshold": float(threshold), "image_feature_count": ni, "text_feature_count": nt,
        "edge_count": edges, "density": edges / mapping.size if mapping.size else 0.0,
        "image_coverage": float((image_degree > 0).mean()) if ni else 0.0,
        "text_coverage": float((text_degree > 0).mean()) if nt else 0.0,
        "image_degree": image_degree, "text_degree": text_degree,
        "image_effective_degree": effective(1), "text_effective_degree": effective(0),
        "image_group": labels[:ni], "text_group": labels[ni:],
        "group_count": len(groups), "groups": groups,
        "total_weight": float(weights.sum()),
    }
