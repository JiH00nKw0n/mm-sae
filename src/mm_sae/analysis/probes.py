"""Regularized linear probes with the same objective on CPU and CUDA.

Only matrix products move to CUDA. SciPy retains the deterministic L-BFGS-B
optimizer, tuning criterion, and convergence checks used by the cached studies.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.special import expit

from .linear import LinearScore, compress_zero_rows, logloss, minimize_logistic


def representation_statistics(x):
    if sparse.issparse(x):
        mean = np.asarray(x.mean(axis=0)).ravel()
        second = np.asarray(x.power(2).mean(axis=0)).ravel()
        std = np.sqrt(np.maximum(second - mean**2, 0))
    else:
        mean = np.mean(x, axis=0, dtype=np.float64)
        std = np.std(x, axis=0, dtype=np.float64)
    return mean, np.where(std > 1e-10, std, 1.0)


def available_columns(x):
    """Dropping always-zero fit columns is exactly equivalent under L2 penalty."""
    if sparse.issparse(x):
        return np.flatnonzero(np.asarray(x.getnnz(axis=0)).ravel())
    return np.flatnonzero(np.any(x != 0, axis=0))


def dense_columns(x, columns):
    selected = x[:, columns]
    return np.asarray(selected.toarray() if sparse.issparse(selected) else selected, dtype=np.float64)


def fit_probe(
    x,
    y,
    tune_x,
    tune_y,
    mean,
    scale,
    penalties,
    *,
    features=None,
    nonnegative=False,
    device="cpu",
    maxiter=1000,
    callback=None,
) -> tuple[LinearScore | None, dict[str, Any]]:
    """Return the selected score and every penalty's optimization diagnostics."""
    if np.unique(y).size < 2 or np.unique(tune_y).size < 2:
        return None, {"status": "missing_fit_or_tune_class"}
    features = available_columns(x) if features is None else np.asarray(features, int)
    raw = dense_columns(x, features)
    raw, labels, mass = compress_zero_rows(raw, np.asarray(y, float))
    mass /= mass.sum()
    z = (raw - mean[features]) / scale[features]
    del raw
    tz = (dense_columns(tune_x, features) - mean[features]) / scale[features]
    initial = np.zeros(len(features) + 1)
    rate = np.clip(np.dot(mass, labels), 1e-8, 1 - 1e-8)
    initial[-1] = np.log(rate / (1 - rate))
    cuda = device.startswith("cuda")
    if cuda:
        import torch

        zz = torch.as_tensor(z, device=device, dtype=torch.float64)
        yy = torch.as_tensor(labels, device=device, dtype=torch.float64)
        mm = torch.as_tensor(mass, device=device, dtype=torch.float64)
        tt = torch.as_tensor(tz, device=device, dtype=torch.float64)
        ty = torch.as_tensor(np.asarray(tune_y, float), device=device, dtype=torch.float64)
        del z, tz
    best, best_loss, trace = None, float("inf"), []
    for penalty in sorted(penalties, reverse=True):
        if callback is not None:
            callback(penalty)

        def objective(theta):
            w, b = theta[:-1], theta[-1]
            if cuda:
                ww = torch.as_tensor(w, device=device, dtype=torch.float64)
                score = zz @ ww + b
                residual = (torch.sigmoid(score) - yy) * mm
                loss = torch.dot(mm, torch.nn.functional.softplus(score) - yy * score)
                loss += penalty * torch.dot(ww, ww) / 2
                grad = torch.cat((zz.T @ residual + penalty * ww, residual.sum().reshape(1)))
                return float(loss.item()), grad.cpu().numpy()
            score = z @ w + b
            residual = (expit(score) - labels) * mass
            loss = np.dot(mass, np.logaddexp(0, score) - labels * score)
            loss += penalty * np.dot(w, w) / 2
            grad = np.r_[z.T @ residual + penalty * w, residual.sum()]
            return float(loss), grad

        result = minimize_logistic(
            objective,
            initial,
            bounds=[(0, None)] * len(features) + [(None, None)] if nonnegative else None,
            maxiter=maxiter,
        )
        if not result.success:
            raise RuntimeError(f"Probe did not converge at lambda={penalty}: {result.message}")
        initial = result.x
        if cuda:
            ww = torch.as_tensor(initial[:-1], device=device, dtype=torch.float64)
            ts = tt @ ww + initial[-1]
            loss = float((torch.nn.functional.softplus(ts) - ty * ts).mean().item())
        else:
            loss = logloss(tune_y, tz @ initial[:-1] + initial[-1])
        trace.append(
            dict(
                penalty=penalty,
                tune_logloss=loss,
                iterations=int(result.nit),
                success=bool(result.success),
                objective=float(result.fun),
                gradient_max=float(np.max(np.abs(result.jac))),
                message=str(result.message),
                line_search_retried=bool(result.line_search_retried),
            )
        )
        if loss < best_loss - 1e-12:
            best_loss = loss
            best = LinearScore(
                features.copy(),
                mean[features],
                scale[features],
                initial[:-1].copy(),
                float(initial[-1]),
                float(penalty),
            )
    return best, dict(
        status="ok",
        device=device,
        optimized_coordinates=len(features),
        input_coordinates=x.shape[1],
        penalties=trace,
    )
