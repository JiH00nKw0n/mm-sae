"""Fit weighted logistic concept scores on fixed, already-selected SAE features.

This module never selects features, changes input normalization, or reads
validation/test examples. Each concept's support must be supplied by the caller.
The fitted logits remain linear in standardized activations. For retrieval,
subtract ``fit_prediction_mean`` from logits before any variance normalization;
use sigmoid(logits), not centered logits, to evaluate label probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from scipy import sparse
from scipy.special import expit

from .linear import minimize_logistic


@dataclass(frozen=True)
class FixedSupportLogisticFit:
    """Linear-logit coefficients in the supplied standardized input coordinates.

    Rows of ``coefficients`` index SAE features; columns index concepts. Raw
    scores are ``((x - mean) / scale) @ coefficients + bias``. The provided
    ``fit_prediction_mean`` includes the intercept and uses the fit sample
    weights. Support and convergence information are recorded per concept.
    """

    coefficients: np.ndarray
    bias: np.ndarray
    fit_prediction_mean: np.ndarray
    penalty: float
    diagnostics: list[dict[str, Any]]


def _validated_inputs(x, y, mean, scale, supports, sample_weight, penalty, device, maxiter):
    if sparse.issparse(x):
        raw_values = x.data
    else:
        x = np.asarray(x)
        raw_values = x
    labels = np.asarray(y)
    mean, scale = np.asarray(mean, float), np.asarray(scale, float)
    if (x.ndim != 2 or labels.ndim != 2 or not len(labels)
            or x.shape[0] != labels.shape[0] or mean.shape != (x.shape[1],)
            or scale.shape != mean.shape or len(supports) != labels.shape[1]):
        raise ValueError("Predictor, label, normalization and support dimensions must agree")
    if (not np.isfinite(raw_values).all() or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("Predictors and normalization must be finite, with positive scales")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Concept labels must contain only binary values 0 and 1")
    weights = np.ones(len(labels)) if sample_weight is None else np.asarray(sample_weight, float)
    if (weights.shape != (len(labels),) or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.isfinite(weights.sum()) or weights.sum() <= 0):
        raise ValueError("Sample weights must be finite, nonnegative, and have positive total mass")
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("penalty must be finite and positive")
    if (isinstance(maxiter, (bool, np.bool_)) or not isinstance(maxiter, (int, np.integer))
            or maxiter < 1):
        raise ValueError("maxiter must be a positive integer")
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("device must be cpu or a CUDA device")
    selected = []
    for support in supports:
        indices = np.asarray(support)
        if (indices.ndim != 1 or (len(indices) and not np.issubdtype(indices.dtype, np.integer))
                or np.any(indices < 0) or np.any(indices >= x.shape[1])
                or len(np.unique(indices)) != len(indices)):
            raise ValueError("Each support must contain unique valid integer feature indices")
        selected.append(indices.astype(int))
    return x, labels, mean, scale, selected, weights / weights.sum()


def _compress_weighted_zero_rows(raw, labels, weights):
    """Collapse zero activation rows exactly, keeping separate label masses."""
    positive = weights > 0
    active = np.any(raw != 0, axis=1) & positive
    blocks, targets, masses = [raw[active]], [labels[active]], [weights[active]]
    for label in (0, 1):
        mass = float(np.sum(weights[positive & ~active & (labels == label)]))
        if mass > 0:
            blocks.append(np.zeros((1, raw.shape[1])))
            targets.append(np.array([label]))
            masses.append(np.array([mass]))
    return np.concatenate(blocks), np.concatenate(targets), np.concatenate(masses)


def _objective_cpu(z, labels, mass, penalty):
    def objective(theta):
        weight, bias = theta[:-1], theta[-1]
        score = np.einsum("ij,j->i", z, weight) + bias
        residual = (expit(score) - labels) * mass
        loss = np.dot(mass, np.logaddexp(0, score) - labels * score)
        loss += penalty * np.dot(weight, weight) / 2
        gradient = np.r_[np.einsum("ij,i->j", z, residual) + penalty * weight, residual.sum()]
        return float(loss), gradient

    return objective


def _objective_cuda(z, labels, mass, penalty, device):
    # Keep the same float64 objective and SciPy optimizer as the CPU path.
    import torch

    zz = torch.as_tensor(z, device=device, dtype=torch.float64)
    yy = torch.as_tensor(labels, device=device, dtype=torch.float64)
    mm = torch.as_tensor(mass, device=device, dtype=torch.float64)

    def objective(theta):
        weight = torch.as_tensor(theta[:-1], device=device, dtype=torch.float64)
        score = zz @ weight + theta[-1]
        residual = (torch.sigmoid(score) - yy) * mm
        loss = torch.dot(mm, torch.nn.functional.softplus(score) - yy * score)
        loss += penalty * torch.dot(weight, weight) / 2
        gradient = torch.cat((zz.T @ residual + penalty * weight, residual.sum().reshape(1)))
        return float(loss.item()), gradient.cpu().numpy()

    return objective


def fit_fixed_support_logistic(
    x,
    y: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    supports: Sequence[Sequence[int] | np.ndarray],
    *,
    sample_weight: np.ndarray | None = None,
    penalty: float = 0.01,
    device: str = "cpu",
    maxiter: int = 1000,
    callback: Callable[[int, dict[str, Any]], None] | None = None,
) -> FixedSupportLogisticFit:
    """Fit one signed logistic readout per concept without changing its support.

    Minimize weighted-mean binary cross entropy plus ``penalty / 2 * ||w||²``.
    Weights are normalized to total mass one; the bias is unpenalized. Repeating
    an image according to its caption count is equivalent to passing that count
    as its sample weight. The feature supports and normalization must be derived
    from training data before calling this function.

    Only one selected feature block is densified at a time. Exact compression of
    all-zero SAE rows reduces work while preserving label masses. CPU and CUDA
    use the same deterministic L-BFGS-B routine. Optimization failure raises an
    error rather than returning unconverged weights. A constant label receives
    zero coefficients and a finite clipped-prevalence intercept, explicitly
    marked in diagnostics; its centered retrieval coordinate is zero.
    """
    x, labels, mean, scale, selected, weights = _validated_inputs(
        x, y, mean, scale, supports, sample_weight, penalty, device, maxiter,
    )
    coefficients = np.zeros((x.shape[1], labels.shape[1]), dtype=np.float64)
    bias = np.zeros(labels.shape[1], dtype=np.float64)
    prediction_mean = np.zeros_like(bias)
    diagnostics = []
    for concept, features in enumerate(selected):
        target = labels[:, concept].astype(np.float64)
        rate = float(np.dot(weights, target))
        prevalence = float(np.clip(rate, 1e-8, 1 - 1e-8))
        initial = np.zeros(len(features) + 1)
        initial[-1] = np.log(prevalence / (1 - prevalence))
        constant = not np.any(target[weights > 0] != target[weights > 0][0])
        detail: dict[str, Any] = dict(
            concept=concept, selected_features=features.tolist(), selected_count=len(features),
            fit_rows=len(labels), positive_label_mass=rate, penalty=float(penalty), device=device,
        )
        if constant or not len(features):
            bias[concept] = prediction_mean[concept] = initial[-1]
            loss = float(np.logaddexp(0, initial[-1]) - rate * initial[-1])
            detail.update(
                status="constant_label" if constant else "intercept_only", success=True,
                iterations=0, objective=loss, fit_logloss=loss, optimized_rows=0,
                gradient_max=None if constant else 0.0, line_search_retried=False,
                message="Clipped prevalence for constant label" if constant else "Exact intercept solution",
            )
        else:
            block: Any = x[:, features]
            raw = np.asarray(block.toarray() if sparse.issparse(block) else block, dtype=np.float64)
            raw, target, mass = _compress_weighted_zero_rows(raw, target, weights)
            z = (raw - mean[features]) / scale[features]
            objective = (_objective_cpu(z, target, mass, penalty) if device == "cpu"
                         else _objective_cuda(z, target, mass, penalty, device))
            result = minimize_logistic(objective, initial, maxiter=maxiter)
            if (not result.success or not np.isfinite(result.x).all()
                    or not np.isfinite(result.fun) or not np.isfinite(result.jac).all()):
                raise RuntimeError(f"Concept {concept} logistic optimizer failed: {result.message}")
            fitted = result.x[:-1]
            coefficients[features, concept] = fitted
            bias[concept] = result.x[-1]
            prediction_mean[concept] = np.dot(np.einsum("i,ij->j", mass, z), fitted) + bias[concept]
            detail.update(
                status="ok", success=True, iterations=int(result.nit), objective=float(result.fun),
                fit_logloss=float(result.fun - penalty * np.dot(fitted, fitted) / 2),
                optimized_rows=len(target), gradient_max=float(np.max(np.abs(result.jac))),
                line_search_retried=bool(result.line_search_retried), message=str(result.message),
            )
        diagnostics.append(detail)
        if callback is not None:
            callback(concept, detail)
    return FixedSupportLogisticFit(coefficients, bias, prediction_mean, float(penalty), diagnostics)
