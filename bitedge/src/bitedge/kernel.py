import torch
import torch.nn.functional as F

MASK32 = 0xFFFFFFFF


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


def fwd(x, Wproj, mids, Wout, bout, binact=True):
    h = F.relu(x @ Wproj.T)
    for W in mids:
        z = h @ W.T
        if binact:
            s = z.abs().mean().detach() + 1e-8
            h = STEsign.apply(z / s) * s
        else:
            h = F.relu(z)
    return h @ Wout.T + bout


def srq_float(e, clamp, bits, gen):
    import os
    qstep = 2 * clamp / (2**bits - (2 if os.environ.get('ONGRID') == '1' else 1))
    q = e / qstep
    fl = q.floor()
    r = torch.rand(q.shape, generator=gen).to(q.device)
    return ((fl + (r < (q - fl)).float()) * qstep).clamp_(-clamp, clamp)


def ed_step_float(wb, ebuf, grad, scale, lr, bits, gen, clamp=1.0):
    u = -lr * grad / scale
    t = ebuf + u
    flip = (t.sign() != wb.sign()) & (t.abs() >= 1.0)
    realized = torch.where(flip, 2.0 * t.sign(), torch.zeros_like(t))
    wb = torch.where(flip, -wb, wb)
    e = srq_float((t - realized).clamp_(-clamp, clamp), clamp, bits, gen)
    return wb, e, int(flip.sum())


def xorshift_seed(shape, seed, device):
    idx = torch.arange(torch.tensor(shape).prod().item(), device=device, dtype=torch.int64)
    s = (idx * 2654435761 + seed * 40503 + 1) & MASK32
    s = torch.where(s == 0, torch.tensor(88172645463325252 & MASK32, device=device), s)
    return s.reshape(shape)


def xorshift_step(state):
    x = state
    x = (x ^ ((x << 13) & MASK32)) & MASK32
    x = (x ^ (x >> 17)) & MASK32
    x = (x ^ ((x << 5) & MASK32)) & MASK32
    return x


DEVSPREAD = float(__import__("os").environ.get("DEVSPREAD", "0"))


def dev_offsets(numel, shape, device):
    idx = torch.arange(numel, device=device, dtype=torch.int64)
    x = (idx * 2654435761 + 12345) & MASK32
    x = (x ^ (x >> 16)) & MASK32
    x = (x * 73244475) & MASK32
    x = (x ^ (x >> 16)) & MASK32
    u = (x >> 8).to(torch.float32) / float(1 << 24)
    return ((u - 0.5) * 2 * DEVSPREAD).reshape(shape)


def ed_step_int(w8, k8, rng, grad, scale, lr, bits, rail=0):
    rail = rail or 2 ** (bits - 1) - 1
    uq = (-lr / scale) * grad * rail
    ui = uq.floor()
    frac = uq - ui
    rng = xorshift_step(rng)
    r = (rng >> 8).to(torch.float32) / float(1 << 24)
    if DEVSPREAD:
        r = (r + dev_offsets(r.numel(), r.shape, r.device)).clamp(0.0, 1.0)
    ui = (ui + (r < frac).float()).to(torch.int32)
    t = k8.to(torch.int32) + ui
    ws = w8.to(torch.int32)
    ts = torch.sign(t)
    push = torch.sign(uq).to(torch.int32)
    flip = (ts != ws) & ((t.abs() > rail) | ((t.abs() == rail) & (push == ts)))
    t = torch.where(flip, t - 2 * rail * ts.to(torch.int32), t)
    w8 = torch.where(flip, -w8, w8)
    k8 = t.clamp(-rail, rail).to(torch.int8)
    return w8, k8, rng, int(flip.sum())


def stateless_r(numel, shape, soff, step, device):
    idx = torch.arange(numel, device=device, dtype=torch.int64)
    x = (idx * 2654435761 + (step * 40503 + soff * 69069 + 1)) & MASK32
    x = (x ^ (x >> 16)) & MASK32
    x = (x * 73244475) & MASK32
    x = (x ^ (x >> 16)) & MASK32
    x = (x * 73244475) & MASK32
    x = (x ^ (x >> 16)) & MASK32
    return ((x >> 8).to(torch.float32) / float(1 << 24)).reshape(shape)
