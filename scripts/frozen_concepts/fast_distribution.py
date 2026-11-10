"""Degree-12 empirical quadrature on sorted GPU scores; refine until each relevant logit interval is <=1.
Same logistic objective, Newton solver, and validation/test populations as the original scalar evaluator.
"""

import numpy as np
import torch
import triton
import triton.language as tl
from fused import calc_nodes


@triton.jit
def _chunks(X, START, END, IDS, MID, RAD, OUT, N: tl.constexpr):
    g = tl.program_id(0)
    start = tl.load(START + g)
    end = tl.load(END + g)
    i = start + tl.arange(0, N)
    ok = i < end
    x = tl.load(X + i, mask=ok, other=0.0)
    c = tl.load(IDS + g)
    r = tl.load(RAD + c)
    mid = tl.load(MID + c)
    v = tl.where(ok, (x - mid) / tl.where(r > 0, r, 1.0), 0.0)
    old = tl.full((N,), 1.0, tl.float64)
    t = v
    tl.store(OUT + g * 13, tl.sum(ok.to(tl.float64), 0))
    tl.store(OUT + g * 13 + 1, tl.sum(v, 0))
    for k in tl.static_range(2, 13):
        nxt = 2.0 * v * t - old
        old = t
        t = nxt
        tl.store(OUT + g * 13 + k, tl.sum(tl.where(ok, t, 0.0), 0))


class Distribution:
    def __init__(self, x, zero):
        self.x = x.contiguous().sort().values
        self.n = len(x)
        self.device = x.device
        self.theta = torch.arange(13, device=x.device, dtype=x.dtype).add_(0.5).mul_(np.pi / 13)
        self.T = torch.cos(torch.arange(13, device=x.device, dtype=x.dtype)[:, None] * self.theta) / 13
        self.T[1:] *= 2
        lo, hi = self.x[0].item(), self.x[-1].item()
        edges = (
            zero
            + np.sinh(np.linspace(np.arcsinh((lo - zero) / 0.15), np.arcsinh((hi - zero) / 0.15), 129)) * 0.15
        )
        self.boundaries = np.unique(
            np.r_[
                0, torch.searchsorted(self.x, torch.as_tensor(edges, device=x.device)).cpu().numpy(), self.n
            ]
        )
        self.pack()

    def pack(self):
        st = self.boundaries[:-1]
        en = self.boundaries[1:]
        device = self.device

        def ints(a):
            return torch.as_tensor(a, dtype=torch.int64, device=device)

        self.lo = self.x[ints(st)]
        self.hi = self.x[ints(en - 1)]
        mi = (self.lo + self.hi) / 2
        ra = (self.hi - self.lo) / 2
        counts = (en - st + 511) // 512
        ids = np.repeat(np.arange(len(st)), counts)
        prev = np.repeat(np.cumsum(counts) - counts, counts)
        starts = st[ids] + (np.arange(len(ids)) - prev) * 512
        ends = np.minimum(starts + 512, en[ids])
        ix = ints(ids)
        out = torch.empty((len(ids), 13), dtype=self.x.dtype, device=device)
        _chunks[(len(ids),)](self.x, ints(starts), ints(ends), ix, mi, ra, out, 512, enable_fp_fusion=False)
        mom = torch.zeros((len(st), 13), dtype=self.x.dtype, device=device).index_add_(0, ix, out)
        nodes = mi[:, None] + ra[:, None] * self.theta.cos()
        mass = (mom @ self.T) / self.n
        self.nodes = nodes.flatten()
        self.mass = mass.flatten()

    def calc(self, w, b, hessian):
        for _ in range(80):
            a = self.lo[:, None] * w + b
            c = self.hi[:, None] * w + b
            bad = ((abs(a - c) > 1) & (torch.minimum(a, c) < 38) & (torch.maximum(a, c) > -38)).any(1)
            if not bad.any():
                return calc_nodes(self.nodes, self.mass, w, b, hessian)
            # Split by observed value, never separate equal scores.
            mids = (self.lo[bad] + self.hi[bad]) / 2
            cuts = torch.searchsorted(self.x, mids, right=True).cpu().numpy()
            new = np.unique(np.r_[self.boundaries, cuts])
            assert len(new) > len(self.boundaries)
            self.boundaries = new
            self.pack()
        raise RuntimeError("quadrature refinement did not converge")
