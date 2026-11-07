"""Accumulate uncentered moments from sparse SAE outputs, then standardize."""

import numpy as np
from mm_sae.data.paired import ChunkedMatrix


def fit_moments(cache, saes, batch_size):
    xs, ys = (ChunkedMatrix(cache, k) for k in ("image", "text"))
    dx, dy = [saes.models[k].config.latent_size for k in ("image", "text")]
    xx, xy, yy = np.zeros((dx, dx)), np.zeros((dx, dy)), np.zeros((dy, dy))
    sums = [np.zeros(dx), np.zeros(dy)]
    try:
        for start in range(0, len(xs), batch_size):
            sl = slice(start, start + batch_size)
            a, b = [saes.activations(k, v[sl]).astype(np.float64) for k, v in (("image", xs), ("text", ys))]
            for total, block in zip(sums, (a, b)):
                total += np.asarray(block.sum(axis=0)).ravel()
            for total, product in ((xx, a.T @ a), (xy, a.T @ b), (yy, b.T @ b)):
                delta = product.tocoo()
                total[delta.row, delta.col] += delta.data
        n = len(xs)
    finally:
        xs.close()
        ys.close()
    means = [s / n for s in sums]
    for mat, left, right in ((xx, 0, 0), (xy, 0, 1), (yy, 1, 1)):
        mat /= n
        mat -= np.outer(means[left], means[right])
    inv = []
    for mat in (xx, yy):
        sd = np.sqrt(np.maximum(mat.diagonal(), 0))
        inv.append(np.divide(1.0, sd, out=np.zeros_like(sd), where=sd > 1e-10))
    for mat, left, right in ((xx, 0, 0), (xy, 0, 1), (yy, 1, 1)):
        mat *= inv[left][:, None]
        mat *= inv[right][None, :]
    return dict(
        xx=xx, xy=xy, yy=yy, image_mean=means[0], text_mean=means[1], image_inv=inv[0], text_inv=inv[1]
    )
