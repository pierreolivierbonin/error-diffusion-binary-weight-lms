import os

import numpy as np
import torch
import torch.nn.functional as F

_HCACHE = {}
QCLIP = float(os.environ.get("QCLIP", "3"))


def _hadamard(n):
    if n not in _HCACHE:
        H = torch.ones(1, 1)
        while H.shape[0] < n:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        _HCACHE[n] = H / np.sqrt(n)
    return _HCACHE[n]


def get_rht(dim, gen, device="cpu", block=32):
    H = _hadamard(block)
    d = torch.randint(0, 2, (dim,), generator=gen).float() * 2 - 1
    return H.to(device), d.to(device)


def crand(shape, gen, device):
    return torch.rand(shape, generator=gen).to(device)


def q1_sr(g, gen):
    s = g.abs().amax(dim=1, keepdim=True) + 1e-12
    p = (g / s + 1) / 2
    return (2 * (crand(g.shape, gen, g.device) < p).float() - 1) * s


def qgrad_fn(g, mode, rht, gen):
    if mode == "fp":
        return g
    if mode == "det1":
        s = g.abs().mean(dim=1, keepdim=True) + 1e-12
        return torch.sign(g) * s
    if mode == "sr1":
        return q1_sr(g, gen)
    if mode == "sr1_clip":
        c = QCLIP * g.abs().mean(dim=1, keepdim=True) + 1e-12
        return q1_sr(g.clamp(-c, c), gen)
    if mode in ("sr1_rht", "sr1_rht_clip"):
        H, d = rht
        blk = H.shape[0]
        B, D = g.shape
        gt = ((g * d).view(B, D // blk, blk) @ H).view(B, D)
        if mode == "sr1_rht_clip":
            c = QCLIP * gt.abs().mean(dim=1, keepdim=True) + 1e-12
            gt = gt.clamp(-c, c)
        q = q1_sr(gt, gen)
        return ((q.view(B, D // blk, blk) @ H.T).view(B, D)) * d
    raise ValueError(mode)


class STEsign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h):
        ctx.save_for_backward(h)
        return torch.sign(h)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (g,) = grad_outputs
        (h,) = ctx.saved_tensors
        return g * (h.abs() <= 1.0).float()


def fwd(x, Wproj, mids, Wout, bout, binact=False, qgrad="fp", rhts=None, gen=None):
    h = F.relu(x @ Wproj.T)
    for li, W in enumerate(mids):
        z = h @ W.T
        if qgrad != "fp" and z.requires_grad:
            rht = None
            if qgrad == "sr1_rht":
                assert rhts is not None
                rht = rhts[li]
            z.register_hook(lambda g, r=rht: qgrad_fn(g, qgrad, r, gen))
        if binact:
            s = z.abs().mean().detach() + 1e-8
            h = STEsign.apply(z / s) * s
        else:
            h = F.relu(z)
    return h @ Wout.T + bout
