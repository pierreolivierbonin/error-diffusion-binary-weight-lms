import torch

from .layers import crand


def _srq(e, clamp, buf_bits, gen):
    qstep = 2 * clamp / (2**buf_bits - 1)
    q = e / qstep
    fl = q.floor()
    return ((fl + (crand(q.shape, gen, q.device) < (q - fl)).float()) * qstep).clamp_(-clamp, clamp)


def ed_step(wb, ebuf, grad, scale, lr, clamp, buf_bits, mode, gen, uclip=0.0):
    u = -lr * grad / scale
    if uclip:
        u = u.clamp(-uclip, uclip)
    if mode == "sr":
        opposes = u.sign() != wb.sign()
        p = (u.abs() / 2).clamp(max=1.0)
        flip = opposes & (crand(u.shape, gen, u.device) < p)
        return torch.where(flip, -wb, wb), ebuf, int(flip.sum())
    t = ebuf + u
    flip = (t.sign() != wb.sign()) & (t.abs() >= 1.0)
    realized = torch.where(flip, 2.0 * t.sign(), torch.zeros_like(t))
    wb = torch.where(flip, -wb, wb)
    e = (t - realized).clamp_(-clamp, clamp)
    if mode == "edB":
        e = _srq(e, clamp, buf_bits, gen)
    elif mode == "edB_det":
        qstep = 2 * clamp / (2**buf_bits - 1)
        e = ((e / qstep).round() * qstep).clamp_(-clamp, clamp)
    return wb, e, int(flip.sum())


def ed2_step(wb, e1, e2, grad, scale, lr, clamp, buf_bits, gen):
    u = -lr * grad / scale
    t1 = e1 + u
    t2 = e2 + t1
    flip = (t2.sign() != wb.sign()) & (t2.abs() >= 1.0)
    realized = torch.where(flip, 2.0 * t2.sign(), torch.zeros_like(t2))
    wb = torch.where(flip, -wb, wb)
    e1 = _srq((t1 - realized).clamp_(-clamp, clamp), clamp, buf_bits, gen)
    e2 = _srq((t2 - realized).clamp_(-clamp, clamp), clamp, buf_bits, gen)
    return wb, e1, e2, int(flip.sum())


def intlat_step(x, grad, scale, lr, bits, gen):
    u = -lr * grad / scale
    t = (x + u).clamp_(-2.0, 2.0)
    qstep = 4.0 / (2**bits - 1)
    q = t / qstep
    fl = q.floor()
    x2 = ((fl + (crand(q.shape, gen, q.device) < (q - fl)).float()) * qstep).clamp_(-2.0, 2.0)
    flips = int((x2.sign() != x.sign()).sum())
    return x2, flips


def _half_split(r):
    return 0.5 * (torch.roll(r, 1, dims=1) + torch.roll(r, -1, dims=1))


def ed_spatial_step(wb, ebuf, grad, scale, lr, clamp, buf_bits, alpha, gen):
    u = -lr * grad / scale
    even = torch.arange(u.shape[1], device=u.device) % 2 == 0
    zero = torch.zeros_like(u)

    t_e = torch.where(even, ebuf + u, zero)
    flip_e = (t_e.sign() != wb.sign()) & (t_e.abs() >= 1.0) & even
    realized_e = torch.where(flip_e, 2.0 * t_e.sign(), zero)
    r_e = torch.where(even, t_e - realized_e, zero)
    send_e = _half_split(r_e * (1 - alpha))

    t_o = torch.where(~even, ebuf + u + send_e, zero)
    flip_o = (t_o.sign() != wb.sign()) & (t_o.abs() >= 1.0) & ~even
    realized_o = torch.where(flip_o, 2.0 * t_o.sign(), zero)
    r_o = torch.where(~even, t_o - realized_o, zero)
    recv_e = _half_split(r_o * (1 - alpha))

    flip = flip_e | flip_o
    wb = torch.where(flip, -wb, wb)
    e = (alpha * (r_e + r_o) + recv_e).clamp_(-clamp, clamp)
    e = _srq(e, clamp, buf_bits, gen)
    return wb, e, int(flip.sum())


def ed_hyst_step(wb, ebuf, refract, grad, scale, lr, clamp, buf_bits, delta, gen):
    u = -lr * grad / scale
    t = ebuf + u
    flip = (t.sign() != wb.sign()) & (t.abs() >= 1.0 + delta * refract)
    realized = torch.where(flip, 2.0 * t.sign(), torch.zeros_like(t))
    wb = torch.where(flip, -wb, wb)
    e = (t - realized).clamp_(-clamp, clamp)
    e = _srq(e, clamp, buf_bits, gen)
    return wb, e, flip.float(), int(flip.sum())


def edm_step(wb, ebuf, mom, grad, scale, lr, clamp, buf_bits, mom_bits, beta, gen):
    u = -lr * grad / scale
    m = beta * mom + u
    t = ebuf + m
    flip = (t.sign() != wb.sign()) & (t.abs() >= 1.0)
    realized = torch.where(flip, 2.0 * t.sign(), torch.zeros_like(t))
    wb = torch.where(flip, -wb, wb)
    e = (t - realized).clamp_(-clamp, clamp)
    e = _srq(e, clamp, buf_bits, gen)
    s = m.abs().amax() + 1e-12
    qstep = 2 * s / (2**mom_bits - 1)
    q = m / qstep
    fl = q.floor()
    m = (fl + (crand(q.shape, gen, q.device) < (q - fl)).float()) * qstep
    return wb, e, m, int(flip.sum())


def ede_step(wb, m, grad, scale, lr, clamp, mom_bits, beta, gen):
    u = -lr * grad / scale
    t = beta * m + u
    flip = (t.sign() != wb.sign()) & (t.abs() >= 1.0)
    realized = torch.where(flip, 2.0 * t.sign(), torch.zeros_like(t))
    wb = torch.where(flip, -wb, wb)
    r = (t - realized).clamp_(-clamp, clamp)
    m = r if mom_bits == 0 else _srq(r, clamp, mom_bits, gen)
    return wb, m, int(flip.sum())


def cnt_step(wb, c, grad, cbits):
    rail = float(2 ** (cbits - 1) - 1)
    c = (c - grad.sign()).clamp_(-rail, rail)
    flip = (c.abs() >= rail) & (c.sign() != wb.sign())
    wb = torch.where(flip, -wb, wb)
    c = torch.where(flip, torch.zeros_like(c), c)
    return wb, c, int(flip.sum())


def bop_step(wb, m, grad, gamma, tau, bits=None, gen=None, align=False):
    m = (1 - gamma) * m + gamma * grad
    if align:
        flip = (m.abs() >= tau) & (m.sign() == wb.sign())
    else:
        flip = (m.abs() > tau) & (m.sign() == wb.sign())
    wb = torch.where(flip, -wb, wb)
    m = torch.where(flip, torch.zeros_like(m), m)
    if bits is not None:
        if align:
            qstep = 2 * tau / (2**bits - 1)
            q = m.clamp(-tau, tau) / qstep
        else:
            s = m.abs().amax() + 1e-12
            qstep = 2 * s / (2**bits - 1)
            q = m / qstep
        fl = q.floor()
        m = (fl + (crand(q.shape, gen, q.device) < (q - fl)).float()) * qstep
    return wb, m, int(flip.sum())
