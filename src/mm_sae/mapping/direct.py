"""Match standardized features in the image coordinate system."""

import numpy as np
from scipy.linalg import svd
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp


def matched_features(cross, rank):
    """Take the strongest pairs in the global maximum-sum one-to-one assignment."""
    image, text = linear_sum_assignment(cross, maximize=True)
    order = np.argsort(-cross[image, text], kind="stable")[:rank]
    return image[order], text[order]


def sinkhorn(cross, epsilon, tolerance, max_iter):
    """Fit balanced entropy-regularized transport with uniform marginals in log space."""
    image_count, text_count = cross.shape
    log_kernel = (cross - cross.max()) / epsilon
    right = np.zeros(text_count)
    for step in range(1, max_iter + 1):
        left = -np.log(image_count) - logsumexp(log_kernel + right, axis=1)
        right = -np.log(text_count) - logsumexp(log_kernel + left[:, None], axis=0)
        if step == 1 or step % 10 == 0 or step == max_iter:
            coupling = np.exp(log_kernel + left[:, None] + right)
            residual = max(
                np.abs(coupling.sum(1) - 1 / image_count).max(),
                np.abs(coupling.sum(0) - 1 / text_count).max(),
            )
            if residual <= tolerance:
                break
    return coupling, {
        "iterations": step,
        "max_marginal_error": float(residual),
        "converged": bool(residual <= tolerance),
        "epsilon": epsilon,
    }


def fit_in_image_coordinates(cross, cfg):
    if cfg.feature_scope == "matched_pairs":
        image, text = matched_features(cross, cfg.rank)
    else:
        image, text = np.arange(cross.shape[0]), np.arange(cross.shape[1])
        if cfg.rank != len(image):
            raise ValueError(f"Image coordinates have width {len(image)}; the configured rank must match")
    selected = cross[np.ix_(image, text)]
    if cfg.method == "sinkhorn":
        coupling, details = sinkhorn(selected, cfg.transport_epsilon, cfg.tolerance, cfg.max_iter)
        # Divide by the uniform text marginal. Each text feature distributes unit mass.
        transport = coupling * len(text)
    else:
        u, _, vt = svd(selected, full_matrices=False)
        transport = u @ vt
        details = {"solver": "orthogonal_procrustes_svd"}
    a = np.zeros((cross.shape[0], len(image)))
    b = np.zeros((cross.shape[1], len(image)))
    a[image, np.arange(len(image))] = 1
    b[text] = transport.T
    return (
        a,
        b,
        details
        | {
            "feature_scope": cfg.feature_scope,
            "image_indices_active": image.tolist(),
            "text_indices_active": text.tolist(),
        },
    )
