"""Single-coordinate semantic transfer with selection restricted to source tuning data.

Score columns must already be paired by the fitted correspondence method. Each
concept chooses one coordinate and its polarity using source tuning labels only.
The identical coordinate and polarity then score source and target validation
observations. No classifier combines coordinates, and validation never refits a
polarity. Missing classes remain explicit entries with undefined AUROC values.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.linalg import get_blas_funcs
from scipy.stats import rankdata


_gemm = get_blas_funcs(("gemm",), dtype=np.float64)[0]


def _scores(value, name: str = "scores"):
    result = value.tocsc() if sparse.issparse(value) else np.asarray(value)
    if result.ndim != 2 or not np.issubdtype(result.dtype, np.number) or np.iscomplexobj(result):
        raise ValueError(f"{name} must be a real two-dimensional matrix")
    return result


def _labels(value, n_samples: int, name: str = "labels") -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 2 or result.shape[0] != n_samples:
        raise ValueError(f"{name} must have one row per score observation and one column per concept")
    if not np.all((result == 0) | (result == 1)):
        raise ValueError(f"{name} must contain only binary zero/one labels")
    return np.asarray(result, dtype=np.float64)


def _chunk_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError("chunk_size must be a positive integer")
    return int(value)


def _dense_columns(scores, columns) -> np.ndarray:
    block = scores[:, columns]
    block = np.asarray(block.toarray() if sparse.issparse(block) else block, dtype=np.float64)
    if not np.all(np.isfinite(block)):
        raise ValueError("scores must contain only finite values")
    return block


def auc_matrix(scores, labels, *, chunk_size: int = 256) -> np.ndarray:
    """Return exact AUROC with shape ``(n_coordinates, n_concepts)``.

    Inputs have shapes ``(n_samples, n_coordinates)`` and
    ``(n_samples, n_concepts)``. Signed scores and sparse matrices are supported.
    Each coordinate is ranked once, with tied observations receiving average
    ranks. Matrix multiplication sums ranks for every concept. Coordinate
    chunks bound temporary rank/sort memory. Constant coordinates yield 0.5
    when both classes exist, while missing classes yield NaN without removal.
    """
    scores = _scores(scores)
    labels = _labels(labels, scores.shape[0])
    chunk_size = _chunk_size(chunk_size)
    positives = labels.sum(axis=0)
    negatives = scores.shape[0] - positives
    valid = (positives > 0) & (negatives > 0)
    valid_labels = np.asfortranarray(labels[:, valid])
    result = np.full((scores.shape[1], labels.shape[1]), np.nan)
    for start in range(0, scores.shape[1], chunk_size):
        stop = min(start + chunk_size, scores.shape[1])
        block = _dense_columns(scores, slice(start, stop))
        if not np.any(valid):
            continue
        ranks = rankdata(block, method="average", axis=0)
        wins = _gemm(1.0, ranks, valid_labels, trans_a=True) - positives[valid] * (positives[valid] + 1) / 2
        result[start:stop, valid] = wins / (positives[valid] * negatives[valid])
    return result


def select_from_auc(source_tune_auc) -> list[dict[str, Any]]:
    """Choose one source coordinate and polarity per concept from tuning AUROC.

    The input follows ``auc_matrix``'s coordinate-by-concept layout. Ties prefer
    the lowest coordinate index. AUROC exactly 0.5 prefers positive polarity.
    An all-NaN concept retains its place with no chosen coordinate or polarity.
    This helper allows callers to reuse a previously computed tuning matrix.
    """
    values = np.asarray(source_tune_auc, dtype=np.float64)
    if values.ndim != 2 or np.any(np.isinf(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("source_tune_auc must be a coordinate-by-concept matrix in [0, 1] or NaN")
    result = []
    for concept in range(values.shape[1]):
        entry: dict[str, Any] = dict(concept_index=concept, coordinate=None, sign=None,
                                     source_tune_auc=float("nan"))
        valid = np.flatnonzero(np.isfinite(values[:, concept]))
        if not len(valid):
            entry["status"] = "no_coordinates" if not values.shape[0] else "missing_source_tune_class"
            result.append(entry)
            continue
        raw = values[valid, concept]
        oriented = np.maximum(raw, 1 - raw)
        # Opposite polarities can differ by one floating-point rounding unit.
        ties = np.isclose(oriented, oriented.max(), rtol=0, atol=4 * np.finfo(float).eps)
        index = int(valid[np.flatnonzero(ties)[0]])
        sign = 1 if values[index, concept] >= 0.5 else -1
        entry.update(coordinate=index, sign=sign,
                     source_tune_auc=float(values[index, concept] if sign == 1 else 1 - values[index, concept]),
                     status="ok")
        result.append(entry)
    return result


def select_source_coordinates(source_tune_scores, source_tune_labels, *, chunk_size: int = 256):
    """Select coordinates using source tuning observations and labels only."""
    return select_from_auc(auc_matrix(source_tune_scores, source_tune_labels, chunk_size=chunk_size))


def _selection_arrays(selection, n_coordinates: int, n_concepts: int):
    if len(selection) != n_concepts:
        raise ValueError("selection must retain exactly one entry per concept")
    coordinates = np.full(n_concepts, -1, dtype=int)
    signs = np.ones(n_concepts, dtype=int)
    for concept, entry in enumerate(selection):
        if entry["concept_index"] != concept:
            raise ValueError("selection concept order must match label columns")
        coordinate, sign = entry["coordinate"], entry["sign"]
        if coordinate is None and sign is None:
            continue
        if (isinstance(coordinate, bool) or not isinstance(coordinate, (int, np.integer))
                or not 0 <= coordinate < n_coordinates or sign not in (-1, 1)):
            raise ValueError("selected coordinate or sign is invalid for these scores")
        coordinates[concept], signs[concept] = coordinate, sign
    return coordinates, signs


def _fixed_coordinate_auc(scores, labels, coordinates, *, chunk_size: int) -> np.ndarray:
    """Evaluate only the chosen coordinate/concept pairs, deduplicating ranks."""
    result = np.full(labels.shape[1], np.nan)
    positives = labels.sum(axis=0)
    negatives = scores.shape[0] - positives
    active = np.flatnonzero((coordinates >= 0) & (positives > 0) & (negatives > 0))
    for start in range(0, len(active), chunk_size):
        concepts = active[start:start + chunk_size]
        unique, inverse = np.unique(coordinates[concepts], return_inverse=True)
        ranks = rankdata(_dense_columns(scores, unique), method="average", axis=0)
        sums = np.einsum("ij,ij->j", ranks[:, inverse], labels[:, concepts])
        wins = sums - positives[concepts] * (positives[concepts] + 1) / 2
        result[concepts] = wins / (positives[concepts] * negatives[concepts])
    return result


def calibrate_target_signs(selection, target_tune_scores, target_tune_labels, *, chunk_size: int = 256):
    """Diagnostic only, choose a target tuning polarity for the fixed coordinate.

    This never changes the source-selected coordinate or source polarity. Its
    separately calibrated polarity can measure target semantics on validation,
    but must not replace the source polarity in semantic transfer results.
    """
    scores = _scores(target_tune_scores)
    labels = _labels(target_tune_labels, scores.shape[0])
    coordinates, _ = _selection_arrays(selection, scores.shape[1], labels.shape[1])
    values = _fixed_coordinate_auc(scores, labels, coordinates, chunk_size=_chunk_size(chunk_size))
    result = []
    for concept, value in enumerate(values):
        sign = None if not np.isfinite(value) else (1 if value >= 0.5 else -1)
        result.append(dict(concept_index=concept, coordinate=selection[concept]["coordinate"],
                           target_tune_sign=sign,
                           target_tune_auc=float(value if sign == 1 else 1 - value)))
    return result


def evaluate_selected_coordinates(
    selection, source_val_scores, source_val_labels, target_val_scores, target_val_labels,
    *, target_calibration=None, chunk_size: int = 256,
) -> list[dict[str, Any]]:
    """Evaluate frozen coordinate/polarity choices on both validation modalities.

    Source and target may have different observation counts, but their paired
    coordinate counts and concept column ordering must agree. Every concept is
    retained. ``paired_minimum_auc`` is undefined if either modality lacks a
    class or if source tuning could not select a coordinate. Optional target
    calibration must come from ``calibrate_target_signs`` on separate tuning
    data, and adds a diagnostic without changing the transfer measurement.
    """
    source = _scores(source_val_scores, "source_val_scores")
    target = _scores(target_val_scores, "target_val_scores")
    source_labels = _labels(source_val_labels, source.shape[0], "source_val_labels")
    target_labels = _labels(target_val_labels, target.shape[0], "target_val_labels")
    if source.shape[1] != target.shape[1] or source_labels.shape[1] != target_labels.shape[1]:
        raise ValueError("source and target must have equal paired-coordinate and concept counts")
    coordinates, signs = _selection_arrays(selection, source.shape[1], source_labels.shape[1])
    chunk_size = _chunk_size(chunk_size)
    source_raw = _fixed_coordinate_auc(source, source_labels, coordinates, chunk_size=chunk_size)
    target_raw = _fixed_coordinate_auc(target, target_labels, coordinates, chunk_size=chunk_size)
    source_auc = np.where(signs == 1, source_raw, 1 - source_raw)
    target_auc = np.where(signs == 1, target_raw, 1 - target_raw)
    if target_calibration is not None and len(target_calibration) != len(selection):
        raise ValueError("target_calibration must retain exactly one entry per concept")
    result = []
    for concept, chosen in enumerate(selection):
        entry = dict(chosen)
        entry.update(source_auc=float(source_auc[concept]), target_transfer_auc=float(target_auc[concept]),
                     paired_minimum_auc=float(np.minimum(source_auc[concept], target_auc[concept])))
        for modality, labels in (("source", source_labels), ("target", target_labels)):
            n_positive = int(labels[:, concept].sum())
            n_negative = int(labels.shape[0] - n_positive)
            entry[f"{modality}_val_positive_count"] = n_positive
            entry[f"{modality}_val_negative_count"] = n_negative
            entry[f"{modality}_val_status"] = ("not_selected" if coordinates[concept] < 0 else
                                               "ok" if n_positive and n_negative else "missing_class")
        if target_calibration is not None:
            diagnostic = target_calibration[concept]
            if diagnostic["concept_index"] != concept or diagnostic["coordinate"] != chosen["coordinate"]:
                raise ValueError("target calibration must preserve source-selected coordinates and concept order")
            sign = diagnostic["target_tune_sign"]
            if sign not in (None, -1, 1):
                raise ValueError("target tuning signs must be +1, -1, or None")
            entry.update(diagnostic)
            entry["target_calibrated_auc"] = (float("nan") if sign is None else
                                              float(target_raw[concept] if sign == 1 else 1 - target_raw[concept]))
        result.append(entry)
    return result
