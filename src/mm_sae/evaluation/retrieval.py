"""Exact full-candidate cosine retrieval with deterministic ties and multiple positives."""

import numpy as np
from scipy import sparse


def normalize(x):
    if sparse.issparse(x):
        norms = np.sqrt(np.asarray(x.multiply(x).sum(1)))
        scale = np.divide(1, norms, out=np.zeros_like(norms), where=norms > 0)
        return x.multiply(scale).tocsr()
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)


def direction(q, c, qgroups, cgroups, block=128):
    lookup = {}
    for i, k in enumerate(cgroups):
        lookup.setdefault(k, []).append(i)
    if set(qgroups) != set(cgroups):
        raise ValueError("Image and caption IDs do not cover the same groups")
    ranks = []
    indices = np.arange(c.shape[0])
    for start in range(0, q.shape[0], block):
        similarities = q[start : start + block] @ c.T
        if sparse.issparse(similarities):
            similarities = similarities.toarray()
        for j, row in enumerate(similarities):
            positives = lookup[qgroups[start + j]]
            best = positives[int(np.argmax(row[positives]))]
            rank = 1 + np.sum(row > row[best]) + np.sum((row == row[best]) & (indices < best))
            ranks.append(int(rank))
    return {f"R@{k}": float(np.mean(np.asarray(ranks) <= k)) for k in (1, 5, 10)}, ranks


def evaluate(image, text, images, captions):
    a, b = normalize(image), normalize(text)
    x, y = [r["group"] for r in images], [r["group"] for r in captions]
    if len(set(x)) != len(x):
        raise ValueError("Expected one candidate row per image")
    i, ir = direction(a, b, x, y)
    t, tr = direction(b, a, y, x)
    return {
        "image_to_text": i,
        "text_to_image": t,
        "images": len(x),
        "captions": len(y),
        "truncated_fraction": float(np.mean([row["truncated"] for row in captions])),
    }, {"image_ranks": ir, "caption_ranks": tr}
