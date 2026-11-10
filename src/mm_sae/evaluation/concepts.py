"""Independent scalar logistic probes; validation selects an axis for each concept."""

from array import array
from collections import defaultdict

import numpy as np
from scipy.special import expit
from scipy import sparse


def losses(scores, y, w, b):
    logits = scores * w + b
    return np.mean(np.logaddexp(0.0, logits) - y[:, None] * logits, axis=0)


def detection_metrics(logits, positive_rows):
    """Compute held-out concept detection counts at probability >= 0.5."""
    predicted = logits >= 0.0
    tp = int(np.count_nonzero(predicted[positive_rows]))
    fp = int(np.count_nonzero(predicted)) - tp
    fn = len(positive_rows) - tp
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": len(logits) - tp - fp - fn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
    }


def macro_detection(probes):
    return {
        "macro_" + metric: float(np.mean([p[metric] for p in probes.values()]))
        for metric in ("precision", "recall", "f1")
    }


def fit_columns(x, y, iterations=80):
    # Every column has its own 2x2 Newton system and backtracking line search.
    y = np.asarray(y, np.float64)
    pos = y.mean()
    b = np.full(x.shape[1], np.log(pos / (1 - pos)))
    w = np.zeros_like(b)
    old = losses(x, y, w, b)
    for _ in range(iterations):
        p = expit(x * w + b)
        e = p - y[:, None]
        h = p * (1 - p)
        gw = (e * x).mean(0)
        gb = e.mean(0)
        hww = (h * x * x).mean(0) + 1e-10
        hbb = h.mean(0) + 1e-10
        hwb = (h * x).mean(0)
        det = hww * hbb - hwb * hwb
        dw = (hbb * gw - hwb * gb) / det
        db = (hww * gb - hwb * gw) / det
        step = np.ones_like(w)
        for _ in range(24):
            new = losses(x, y, w - step * dw, b - step * db)
            reject = new > old
            if not reject.any():
                break
            step[reject] *= 0.5
        step[new > old] = 0.0
        w -= step * dw
        b -= step * db
        if np.max(np.abs(old - new)) < 1e-9:
            break
        old = losses(x, y, w, b)
    return w, b


def evaluate(
    scores,
    records,
    min_count,
    diagnostics,
    allowed=None,
    use_cuda=False,
    min_test_images=0,
    label_prefix=None,
):
    row_ids = {key: array("q") for key in ("train", "val", "test")}
    positives = defaultdict(lambda: array("q"))
    train_counts = defaultdict(int)
    group_split = {}
    test_images_by_label = defaultdict(set)
    n_rows = 0
    for i, row in enumerate(records):
        group, split = row["group"], row["split"]
        if group_split.setdefault(group, split) != split:
            raise ValueError("An image group crosses probe splits")
        row_ids[split].append(i)
        for label in set(row["labels"]):
            positives[label].append(i)
            if min_test_images and split == "test":
                test_images_by_label[label].add(group)
            if split == "train":
                train_counts[label] += 1
        n_rows += 1
    if n_rows != scores.shape[0]:
        raise ValueError("Embedding/metadata row mismatch")
    ids = {key: np.asarray(values) for key, values in row_ids.items()}
    if any(not len(values) for values in ids.values()):
        raise ValueError("Concept probes require nonempty train, val and test splits")
    names = sorted(train_counts) if allowed is None else sorted(set(allowed))
    if min_test_images:
        test_groups = sum(split == "test" for split in group_split.values())
        candidates = sorted(test_images_by_label) if allowed is None else names
        names = [
            name
            for name in candidates
            if min(len(test_images_by_label[name]), test_groups - len(test_images_by_label[name]))
            >= min_test_images
        ]
    if label_prefix is not None:
        names = [name for name in names if name.startswith(label_prefix)]
    if use_cuda:
        from scipy.sparse import csc_matrix
        from . import concepts_cuda

        offsets = np.cumsum([0] + [len(positives[name]) for name in names])
        row_index = (
            np.concatenate([np.asarray(positives[name], dtype=np.int32) for name in names])
            if names
            else np.empty(0, np.int32)
        )
        labels = csc_matrix(
            (np.ones(len(row_index), dtype=bool), row_index, offsets), shape=(n_rows, len(names))
        )
        return concepts_cuda.evaluate(scores, labels, names, ids, min_count, diagnostics)

    def values(rows, cols):
        block = scores[rows, cols]
        return block.toarray() if sparse.issparse(block) else block

    it, iv, ie = ids["train"], ids["val"], ids["test"]
    total = np.zeros(scores.shape[1])
    squares = np.zeros_like(total)
    lowest = np.full_like(total, np.inf)
    highest = np.full_like(total, -np.inf)
    for start in range(0, len(it), 8192):
        block = np.asarray(values(it[start : start + 8192], slice(None)), dtype=np.float64)
        lowest = np.minimum(lowest, block.min(0))
        highest = np.maximum(highest, block.max(0))
        total += block.sum(0)
        squares += np.einsum("ij,ij->j", block, block)
    constant = lowest == highest
    mean = total / len(it)
    scale = np.sqrt(np.maximum(squares / len(it) - mean**2, 0))
    scale[scale < 1e-10] = 1.0
    # Bound the Newton working arrays independently of dataset size.
    width = max(1, min(32, (32 << 20) // (8 * len(it))))
    result = {}
    excluded = {}
    importance = []
    val_losses = []
    for name in names:
        if min(train_counts[name], len(it) - train_counts[name]) < min_count:
            excluded[name] = "insufficient train support"
            continue
        y = np.zeros(n_rows, dtype=np.float64)
        y[np.asarray(positives[name], dtype=int)] = 1
        if min(y[it].sum(), len(it) - y[it].sum()) < min_count or min(y[iv].sum(), len(iv) - y[iv].sum()) < 1:
            excluded[name] = "insufficient train or validation support"
            continue
        prior = y[it].mean()
        prior_bias = np.log(prior) - np.log1p(-prior)
        fraction = y[iv].mean()
        null = -fraction * np.log(prior) - (1 - fraction) * np.log1p(-prior)
        weights = []
        intercepts = []
        valid = []
        for start in range(0, scores.shape[1], width):
            block = slice(start, start + width)
            x = (values(it, block) - mean[block]) / scale[block]
            w, b = fit_columns(x, y[it])
            w = w / scale[block]
            b = b - w * mean[block]
            w[constant[block]] = 0
            b[constant[block]] = prior_bias
            validation_loss = losses(values(iv, block), y[iv], w, b)
            validation_loss[constant[block]] = null
            weights.extend(w)
            intercepts.extend(b)
            valid.extend(validation_loss)
        j = int(np.argmin(valid))
        w, b = weights[j], intercepts[j]
        logits = values(ie, slice(j, j + 1)).ravel().astype(float) * w + b
        test_loss = float(np.mean(np.logaddexp(0.0, np.where(y[ie], -logits, logits))))
        importance.append(np.maximum(0.0, null - np.asarray(valid)))
        val_losses.append(valid)
        result[name] = {
            "axis": j,
            "weight": w,
            "bias": b,
            "val_loss": valid[j],
            "test_loss": test_loss,
            **detection_metrics(logits, np.flatnonzero(y[ie])),
            "positives": {k: int(y[v].sum()) for k, v in ids.items()},
        }
    if not result:
        raise ValueError("No eligible concepts; check labels, split sizes and min_train_positives")
    g = np.array(importance).T
    summary = {
        "macro_test_loss": float(np.mean([x["test_loss"] for x in result.values()])),
        **macro_detection(result),
        "eligible": len(result),
        "excluded": excluded,
        "split_sizes": {k: len(v) for k, v in ids.items()},
        "concepts": result,
    }
    if diagnostics:
        summary["diagnostics"] = entropy_scores(g)
    return summary, {
        "concepts": np.array(list(result)),
        "validation_loss": np.array(val_losses).T,
        "importance": g,
    }


def entropy_scores(g):
    from scipy.special import xlogy

    def entropy(values):
        mass = values.sum(1)
        p = np.divide(values, mass[:, None], out=np.zeros_like(values), where=mass[:, None] > 0)
        h = -xlogy(p, p).sum(1) / np.log(values.shape[1]) if values.shape[1] > 1 else np.zeros(len(values))
        return h, mass

    mix, axis_mass = entropy(g)
    spread, concept_mass = entropy(g.T)
    valid = concept_mass > 0
    return {
        "mixing": float(np.average(mix, weights=axis_mass)) if valid.any() else None,
        "dispersion": float(spread[valid].mean()) if valid.any() else None,
        "zero_importance_concepts": int((~valid).sum()),
        "zero_importance_axes": int((axis_mass == 0).sum()),
    }
