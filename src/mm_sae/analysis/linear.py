"""Small regularized linear models; no encoder or SAE training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


@dataclass
class LinearScore:
    features: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    bias: float
    penalty: float

    def predict(self, raw):
        x = raw[:, self.features]
        if hasattr(x, "toarray"):
            x = x.toarray()
        return ((np.asarray(x) - self.mean) / self.scale) @ self.weight + self.bias

    def record(self):
        return dict(features=self.features, mean=self.mean, scale=self.scale,
                    weight=self.weight, bias=self.bias, penalty=self.penalty)

    @classmethod
    def from_record(cls, r):
        return cls(np.asarray(r["features"], int), np.asarray(r["mean"]), np.asarray(r["scale"]),
                   np.asarray(r["weight"]), r["bias"], r["penalty"])


def logloss(y, score, weights=None):
    return float(np.average(np.logaddexp(0, score) - y * score, weights=weights))


def compress_zero_rows(raw, y):
    """Collapse identical zero rows exactly, preserving both label masses."""
    active = np.any(raw != 0, axis=1)
    x, labels, mass = [raw[active]], [y[active]], [np.ones(active.sum())]
    for label in [0, 1]:
        count = np.sum(~active & (y == label))
        if count:
            x.append(np.zeros((1, raw.shape[1])))
            labels.append(np.array([label]))
            mass.append(np.array([count]))
    return np.concatenate(x), np.concatenate(labels), np.concatenate(mass)


def minimize_logistic(objective, initial, bounds=None, maxiter=1000):
    """Retry a failed line search without changing the objective or tolerances."""
    options = {"maxiter": maxiter, "ftol": 1e-11, "gtol": 1e-7}
    result = minimize(objective, initial, jac=True, bounds=bounds, method="L-BFGS-B",
                      options=options)
    retried = not result.success and "ABNORMAL" in str(result.message)
    if retried:
        result = minimize(objective, initial, jac=True, bounds=bounds, method="L-BFGS-B",
                          options={**options, "maxls": 100})
    result["line_search_retried"] = retried
    return result


def fit_logistic(x, y, tune_x, tune_y, features, mean, scale, penalties, nonnegative=False):
    """Weighted-mean logistic loss + lambda/2 ||w||²; intercept is unpenalized."""
    if np.unique(y).size < 2 or np.unique(tune_y).size < 2:
        return None
    raw = np.asarray(x[:, features].toarray() if hasattr(x, "toarray") else x[:, features], float)
    tx = np.asarray(tune_x[:, features].toarray() if hasattr(tune_x, "toarray")
                    else tune_x[:, features], float)
    raw, labels, mass = compress_zero_rows(raw, np.asarray(y))
    mass = mass / mass.sum()
    m, s = mean[features], scale[features]
    z, tz = (raw-m)/s, (tx-m)/s
    initial = np.zeros(len(features)+1)
    rate = np.clip(np.dot(mass, labels), 1e-8, 1-1e-8)
    initial[-1] = np.log(rate/(1-rate))
    best, best_loss = None, float("inf")
    for penalty in sorted(penalties, reverse=True):
        def objective(theta):
            w, b = theta[:-1], theta[-1]
            score = z @ w + b
            residual = (expit(score)-labels)*mass
            loss = np.dot(mass, np.logaddexp(0, score)-labels*score) + penalty*np.dot(w, w)/2
            grad = np.r_[z.T @ residual + penalty*w, residual.sum()]
            return float(loss), grad
        bounds = [(0, None)]*len(features)+[(None, None)] if nonnegative else None
        result = minimize_logistic(objective, initial, bounds)
        if not result.success:
            raise RuntimeError(f"Logistic optimizer failed: {result.message}")
        initial = result.x
        loss = logloss(tune_y, tz @ initial[:-1]+initial[-1])
        if loss < best_loss-1e-12:
            best_loss = loss
            best = LinearScore(np.asarray(features), m, s, initial[:-1].copy(),
                               float(initial[-1]), float(penalty))
    return best


def scale_statistics(x):
    mean = np.asarray(x.mean(0)).ravel()
    second = np.asarray(x.power(2).mean(0)).ravel()
    sd = np.sqrt(np.maximum(second-mean**2, 0))
    return mean, np.where(sd > 1e-10, sd, 1.)


def identity_score(feature, mean, scale):
    return LinearScore(np.array([feature]), mean[[feature]], scale[[feature]], np.ones(1), 0., 0.)
