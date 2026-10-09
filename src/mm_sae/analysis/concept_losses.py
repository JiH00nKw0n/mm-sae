"""Fixed-support annotation probes with binary or soft-label cross entropy."""

import numpy as np
import torch
import torch.nn.functional as F


def supervised_loss(logits, labels, weights, kind):
    if kind == 'binary_ce':
        per_sample = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').sum(1)
        used = weights
    elif kind == 'ce':
        count = labels.sum(1)
        target = labels / count.clamp_min(1)[:, None]
        per_sample = -(target * F.log_softmax(logits, dim=1)).sum(1)
        used = weights * (count > 0)
    else:
        raise ValueError(f'Unknown loss: {kind}')
    if used.sum() <= 0:
        raise ValueError('No supervised samples')
    return (used * per_sample).sum() / used.sum()


def fit_fixed_support(raw, labels, sample_weights, mean, scale, support_weights, *,
                      kind, penalty=.01, max_iter=250, device='cuda', progress=None):
    """Fit weights and intercepts; selection stays fixed to the given nonzero support.

    Binary CE sums independent class losses, then averages over weighted samples.
    CE averages softmax cross entropy for uniform mass over positive concepts.
    Empty-label samples have zero weight only in CE. Ridge penalty is half lambda
    times the sum of squared weights, with unpenalized intercepts.
    """
    supports = [np.flatnonzero(support_weights[:, c]) for c in range(support_weights.shape[1])]
    width = max(map(len, supports))
    indices = np.zeros((len(supports), width), dtype=np.int64)
    mask = np.zeros_like(indices, dtype=bool)
    for c, chosen in enumerate(supports):
        indices[c, :len(chosen)] = chosen
        mask[c, :len(chosen)] = True
    n = raw.shape[0]
    design = torch.empty((n, *indices.shape), dtype=torch.float32, device="cpu")
    for start in range(0, n, 4096):
        block = raw[start:start+4096, indices.ravel()].toarray()
        block = (block - mean[indices.ravel()]) / scale[indices.ravel()]
        design[start:start+len(block)] = torch.as_tensor(
            block.reshape(len(block), *indices.shape), dtype=torch.float32)
    active = torch.as_tensor(mask, dtype=torch.float32, device=device)
    target = torch.as_tensor(labels, dtype=torch.float32, device=device)
    weight = torch.as_tensor(sample_weights, dtype=torch.float32, device=device)
    if not all(torch.isfinite(block).all() for block in design.split(4096)):
        raise ValueError('Nonfinite standardized activations')
    coef = torch.zeros(indices.shape, device=device, requires_grad=True)
    prevalence = ((target * weight[:, None]).sum(0)/weight.sum()).clamp(1e-5, 1-1e-5)
    intercept = (torch.logit(prevalence) if kind == 'binary_ce' else prevalence.log()).detach()
    intercept.requires_grad_(True)
    optimizer = torch.optim.LBFGS([coef, intercept], lr=1., max_iter=max_iter,
                                  history_size=10, tolerance_grad=1e-5,
                                  tolerance_change=1e-9, line_search_fn='strong_wolfe')
    evaluations = 0
    first_loss = None
    last_loss = None

    def closure():
        nonlocal evaluations, first_loss, last_loss
        optimizer.zero_grad()
        total_weight = (weight * (target.sum(1) > 0)).sum() if kind == 'ce' else weight.sum()
        loss_value = torch.zeros((), device=device)
        for start in range(0, n, 4096):
            chunk = design[start:start+4096].to(device)
            yt = target[start:start+4096]
            wt = weight[start:start+4096]
            mass = (wt * (yt.sum(1) > 0)).sum() if kind == 'ce' else wt.sum()
            if mass == 0:
                continue
            logits = (chunk * (coef * active)[None]).sum(2) + intercept
            part = supervised_loss(logits, yt, wt, kind) * (mass / total_weight)
            part.backward()
            loss_value += part.detach()
        regularizer = .5 * penalty * (coef * active).square().sum()
        regularizer.backward()
        loss = loss_value + regularizer.detach()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        evaluations += 1
        last_loss = float(loss.detach())
        if first_loss is None:
            first_loss = last_loss
        if progress and evaluations % 10 == 0:
            progress(dict(function_evaluations=evaluations, training_loss=last_loss))
        return loss

    optimizer.step(closure)
    closure()
    gradients = [p.grad for p in (coef, intercept)]
    assert all(g is not None for g in gradients)
    gradient = max(float(g.abs().max()) for g in gradients if g is not None)
    local = (coef * active).detach().cpu().numpy()
    result = np.zeros_like(support_weights)
    for c, chosen in enumerate(supports):
        result[chosen, c] = local[c, :len(chosen)]
    audit = dict(kind=kind, penalty=penalty, max_iter=max_iter,
                 iterations=optimizer.state[coef].get('n_iter', 0), function_evaluations=evaluations,
                 initial_loss=first_loss, final_loss=last_loss, gradient_inf=gradient,
                 stationary_at_1e_4=gradient <= 1e-4,
                 zero_label_samples=int((target.sum(1) == 0).sum()),
                 support_selection='fixed Ridge lambda=0.01 support',
                 intercept=intercept.detach().cpu().numpy().tolist())
    return result, audit
