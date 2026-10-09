"""Column-budget gradient hard thresholding pursuit for concept probes.

This adapts GraHTP to a product of per-column cardinality constraints.
The SAE is fixed. All coordinates are eligible again at every outer step.
No claim of a global optimum is made. Computation uses float64 throughout.
"""

import time
import torch
import torch.nn.functional as F


class ProbeLoss:
    def __init__(self, x, y, weights, kind, penalty, batch=2048):
        self.x, self.y, self.weights = x, y, weights
        self.kind, self.penalty, self.batch = kind, penalty, batch
        self.used = weights * ((y.sum(1) > 0) if kind == "ce" else 1)
        self.mass = self.used.sum()
        if self.mass <= 0:
            raise ValueError("Empty training population")
        self.mean = (weights[:, None] * y).sum(0) / weights.sum()
        self.variable = (self.mean > 0) & (self.mean < 1)
        self.cov = self.cross = None
        if kind == "mse":
            self.cov = x.T @ (x * (weights / weights.sum())[:, None])
            self.cross = x.T @ ((y - self.mean) * (weights / weights.sum())[:, None])
            self.variance = (self.mean * (1 - self.mean)).sum()

    def full(self, w, b, gradient=True):
        if self.kind == "mse":
            cw = self.cov @ w
            value = (
                0.5
                * (
                    self.variance
                    - 2 * (w * self.cross).sum()
                    + (w * cw).sum()
                    + (b - self.mean).square().sum()
                )
                + 0.5 * self.penalty * w.square().sum()
            )
            return float(value), cw - self.cross + self.penalty * w, b - self.mean
        value = torch.zeros((), dtype=w.dtype, device=w.device)
        gw, gb = torch.zeros_like(w), torch.zeros_like(b)
        for start in range(0, len(self.x), self.batch):
            x, y = self.x[start : start + self.batch], self.y[start : start + self.batch]
            q = self.used[start : start + self.batch] / self.mass
            logits = x @ w + b
            if self.kind == "binary_ce":
                losses = F.binary_cross_entropy_with_logits(logits, y, reduction="none").sum(1)
                delta = torch.sigmoid(logits) - y
            elif self.kind == "squared_hinge":
                target = 2 * y - 1
                margin = (1 - target * logits).clamp_min(0)
                losses = margin.square().sum(1)
                delta = -2 * target * margin
            else:
                target = y / y.sum(1, keepdim=True).clamp_min(1)
                losses = -(target * F.log_softmax(logits, dim=1)).sum(1)
                delta = torch.softmax(logits, dim=1) - target
            value += (q * losses).sum()
            if gradient:
                delta *= q[:, None]
                gw += x.T @ delta
                gb += delta.sum(0)
        value += 0.5 * self.penalty * w.square().sum()
        if gradient:
            gw += self.penalty * w
        return float(value), gw, gb

    def refit(self, support, w, b, max_iter=300, tolerance=1e-6):
        d, c = w.shape
        k = support.shape[0]
        columns = torch.arange(c, device=w.device)[None, :]
        if self.kind == "mse":
            assert self.cov is not None and self.cross is not None
            idx = support.T
            matrices = self.cov[idx[:, :, None], idx[:, None, :]]
            matrices = matrices + self.penalty * torch.eye(k, device=w.device, dtype=w.dtype)[None]
            rhs = self.cross[support, columns].T
            # Pseudoinverse also defines the unregularized singular-support solution.
            local = (torch.linalg.pinv(matrices, hermitian=True, rtol=1e-12) @ rhs[:, :, None]).squeeze(-1).T
            result = torch.zeros_like(w).scatter(0, support, local)
            result[:, ~self.variable] = 0
            val, grad, _ = self.full(result, self.mean)
            residual = float(grad[support, columns].abs().max())
            return (
                result,
                self.mean.clone(),
                dict(
                    inner_iterations=1,
                    inner_evaluations=1,
                    inner_gradient=residual,
                    inner_converged=residual <= tolerance,
                    objective=val,
                ),
            )
        local = w[support, columns].clone().detach().requires_grad_(True)
        bias = b.clone().detach().requires_grad_(True)
        active = self.variable[None, :].to(w.dtype)
        optimizer = torch.optim.LBFGS(
            [local, bias],
            lr=1,
            max_iter=max_iter,
            history_size=20,
            tolerance_grad=tolerance * 0.1,
            tolerance_change=1e-13,
            line_search_fn="strong_wolfe",
        )
        evaluations = 0
        loss_trace = []
        # Keep the selected design on GPU across all inner optimizer evaluations.
        selected_design = None
        estimated_bytes = len(self.x) * support.numel() * self.x.element_size()
        if self.x.is_cuda and torch.cuda.mem_get_info(self.x.device)[0] > estimated_bytes + 12 * 2**30:
            selected_design = self.x[:, support]
        started = time.monotonic()

        def closure():
            nonlocal evaluations
            optimizer.zero_grad()
            total = torch.zeros((), dtype=w.dtype, device=w.device)
            for start in range(0, len(self.x), self.batch):
                x, y = self.x[start : start + self.batch], self.y[start : start + self.batch]
                q = self.used[start : start + self.batch] / self.mass
                selected = (
                    x[:, support] if selected_design is None else selected_design[start : start + self.batch]
                )
                logits = torch.einsum("nkc,kc->nc", selected, local * active) + bias
                if self.kind == "binary_ce":
                    values = F.binary_cross_entropy_with_logits(logits, y, reduction="none").sum(1)
                elif self.kind == "squared_hinge":
                    target = 2 * y - 1
                    values = (1 - target * logits).clamp_min(0).square().sum(1)
                else:
                    target = y / y.sum(1, keepdim=True).clamp_min(1)
                    values = -(target * F.log_softmax(logits, dim=1)).sum(1)
                loss = (q * values).sum()
                loss.backward()
                total += loss.detach()
            reg = 0.5 * self.penalty * (local * active).square().sum()
            reg.backward()
            if self.kind in ("binary_ce", "squared_hinge"):
                bias.grad[~self.variable] = 0
            total += reg.detach()
            evaluations += 1
            if evaluations == 1 or evaluations % 10 == 0:
                loss_trace.append(
                    dict(evaluation=evaluations, objective=float(total), seconds=time.monotonic() - started)
                )
            if not torch.isfinite(total):
                raise FloatingPointError("Nonfinite objective")
            return total

        optimizer.step(closure)
        final = float(closure())
        residual = max(float(local.grad.abs().max()), float(bias.grad.abs().max()))
        result = torch.zeros_like(w).scatter(0, support, (local * active).detach())
        return (
            result,
            bias.detach(),
            dict(
                inner_iterations=int(optimizer.state[local]["n_iter"]),
                inner_evaluations=evaluations,
                inner_gradient=residual,
                inner_converged=residual <= tolerance,
                objective=final,
                inner_seconds=time.monotonic() - started,
                inner_loss_trace=loss_trace,
            ),
        )


def fit_pursuit(
    x,
    y,
    weights,
    *,
    kind,
    penalty,
    budget=16,
    max_outer=50,
    inner_iter=300,
    tolerance=1e-6,
    progress=None,
    initial=None,
):
    """Backtracked GraHTP with restricted refits and an explicit column budget."""
    problem = ProbeLoss(x, y, weights, kind, penalty)
    w = torch.zeros((x.shape[1], y.shape[1]), device=x.device, dtype=x.dtype)
    p = problem.mean.clamp(1e-8, 1 - 1e-8)
    b = problem.mean.clone() if kind == "mse" else torch.logit(p) if kind == "binary_ce" else p.log()
    if initial is not None:
        w = torch.as_tensor(initial[0], device=x.device, dtype=x.dtype).clone()
        b = torch.as_tensor(initial[1], device=x.device, dtype=x.dtype).clone()
    # Power iteration estimates a useful starting step; monotone backtracking checks it.
    v = torch.ones(x.shape[1], device=x.device, dtype=x.dtype)
    v /= v.norm()
    q = problem.used / problem.mass
    for _ in range(30):
        u = x.T @ ((x @ v) * q)
        v = u / u.norm().clamp_min(1e-15)
    spectral = float((v * (x.T @ ((x @ v) * q))).sum())
    factor = {"mse": 1.0, "binary_ce": 0.25, "ce": 0.5, "squared_hinge": 2.0}[kind]
    eta = 1 / max(factor * spectral + penalty, 1e-8)
    initial_objective = problem.full(w, b)[0]
    history = []
    old_support = None
    stop = "outer_iteration_limit"
    start = time.monotonic()
    for outer in range(max_outer):
        value, gw, gb = problem.full(w, b)
        gw[:, ~problem.variable] = 0
        record = dict(outer_iteration=outer, objective=value, elapsed_seconds=time.monotonic() - start)
        if progress:
            progress(dict(record, state="selecting_support"))
        accepted = False
        support = None
        nw, nb = w, b
        audit = {}
        for attempt in range(12):
            proposal = w - eta * gw
            support = torch.topk(proposal.abs(), min(budget, len(proposal)), dim=0).indices
            nw, nb, audit = problem.refit(support, w, b, inner_iter, tolerance)
            if audit["objective"] <= value + 1e-11 * max(1, abs(value)):
                accepted = True
                break
            eta *= 0.5
        if not accepted:
            stop = "line_search_failed"
            record.update(state=stop, step_size=eta)
            history.append(record)
            if progress:
                progress(record)
            break
        assert support is not None
        selected = torch.zeros_like(w, dtype=torch.bool).scatter(0, support, True)
        changes = int(selected.sum()) if old_support is None else int((selected != old_support).sum())
        record.update(
            audit,
            step_size=eta,
            support_changes=changes,
            max_coefficient=float(nw.abs().max()),
            state="outer_completed",
        )
        history.append(record)
        if progress:
            progress(record)
        w, b = nw, nb
        if changes == 0 and audit["inner_converged"]:
            stop = "support_fixed_and_inner_stationary"
            break
        if float(w.abs().max()) > 1e4:
            stop = "coefficient_growth_limit"
            break
        old_support = selected
    value, gw, gb = problem.full(w, b)
    support = torch.topk(w.abs(), min(budget, len(w)), dim=0).indices
    restricted = gw.gather(0, support)
    grad = max(float(restricted.abs().max()), float(gb[problem.variable].abs().max()))
    return (
        w.cpu().numpy(),
        b.cpu().numpy(),
        dict(
            kind=kind,
            penalty=penalty,
            stop_reason=stop,
            objective=value,
            initial_objective=initial_objective,
            restricted_gradient=grad,
            max_nonzeros=int((w != 0).sum(0).max()),
            seconds=time.monotonic() - start,
            history=history,
            global_optimum_certified=False,
        ),
    )
