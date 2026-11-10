"""Fuse the six weighted one-dimensional sums without changing quadrature or the Newton solver."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _moments(X, M, W, B, OUT, n, k, nb, hessian: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    g = tl.program_id(0)
    xs = g * BN + tl.arange(0, BN)
    labels = tl.program_id(1) * BK + tl.arange(0, BK)
    x = tl.load(X + xs, mask=xs < n, other=0.0).to(tl.float64)[:, None]
    mass = tl.load(M + xs, mask=xs < n, other=0.0).to(tl.float64)[:, None]
    w = tl.load(W + labels, mask=labels < k, other=0.0)[None, :]
    b = tl.load(B + labels, mask=labels < k, other=0.0)[None, :]
    z = x * w + b
    e = tl.exp(-tl.abs(z))
    p = tl.where(z >= 0, 1.0, e) / (1.0 + e)
    f = mass * (tl.maximum(0.0, z) + libdevice.log1p(e))
    off = g * k + labels
    tl.store(OUT + off, tl.sum(f, 0), mask=labels < k)
    if hessian:
        a = mass * p
        h = mass * e / ((1.0 + e) * (1.0 + e))
        tl.store(OUT + nb * k + off, tl.sum(a * x, 0), mask=labels < k)
        tl.store(OUT + 2 * nb * k + off, tl.sum(a, 0), mask=labels < k)
        tl.store(OUT + 3 * nb * k + off, tl.sum(h * x * x, 0), mask=labels < k)
        tl.store(OUT + 4 * nb * k + off, tl.sum(h * x, 0), mask=labels < k)
        tl.store(OUT + 5 * nb * k + off, tl.sum(h, 0), mask=labels < k)


def calc_nodes(x, mass, w, b, hessian):
    n = x.numel()
    k = w.numel()
    bn = 128
    bk = 8
    nb = triton.cdiv(n, bn)
    out = torch.empty((6 if hessian else 1, nb, k), dtype=torch.float64, device=x.device)
    _moments[(nb, triton.cdiv(k, bk))](
        x, mass, w.contiguous(), b.contiguous(), out, n, k, nb, hessian, bn, bk, enable_fp_fusion=False
    )
    return out[:, 0, :] if nb == 1 else out.sum(1)
