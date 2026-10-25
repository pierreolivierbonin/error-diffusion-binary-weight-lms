import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import os

from .kernel import stateless_r, xorshift_seed, xorshift_step


def pack_bits(w8):
    bits = (w8.flatten() > 0).to(torch.uint8)
    pad = (-bits.numel()) % 8
    if pad:
        bits = torch.cat([bits, torch.zeros(pad, dtype=torch.uint8, device=bits.device)])
    bits = bits.reshape(-1, 8)
    weights = (2 ** torch.arange(8, dtype=torch.uint8, device=bits.device)).unsqueeze(0)
    return (bits * weights).sum(1).to(torch.uint8)


def unpack_bits(packed, shape, device):
    n = int(np.prod(shape))
    exp = torch.arange(8, dtype=torch.uint8, device=device).unsqueeze(0)
    bits = (packed.unsqueeze(1) >> exp) & 1
    return (bits.flatten()[:n].to(torch.float32) * 2 - 1).reshape(shape)


class STEsignPacked(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h):
        mask = h.abs() <= 1.0
        ctx.save_for_backward(pack_bits(mask.to(torch.int8) * 2 - 1))
        ctx.shape = h.shape
        ctx.dev = h.device
        return torch.sign(h)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (g,) = grad_outputs
        (packed,) = ctx.saved_tensors
        mask = (unpack_bits(packed, ctx.shape, ctx.dev) + 1) / 2
        return g * mask


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


class BitLinear(nn.Module):
    def __init__(self, d_in, d_out, mode, gen, device, seed_off=0):
        super().__init__()
        W0 = torch.randn(d_out, d_in, generator=gen) / np.sqrt(d_in)
        self.mode = mode
        self.shape = (d_out, d_in)
        self.register_buffer("scale", W0.abs().mean().to(device))
        s = torch.sign(W0) + (torch.sign(W0) == 0).float()
        if mode == "bitnet":
            self.weight = nn.Parameter(W0.to(device))
        elif mode == "ed_naive":
            self.register_buffer("wb", s.to(device))
            self.register_buffer("ebuf", torch.zeros_like(W0).to(device))
        elif mode == "ed_compact":
            self.register_buffer("wb", s.to(torch.int8).to(device))
            self.register_buffer("kbuf", torch.zeros(self.shape, dtype=torch.int8, device=device))
            self.register_buffer("rng", xorshift_seed(self.shape, 17 + seed_off, device))
        elif mode == "ed_packed":
            self.register_buffer("wpk", pack_bits(s.to(torch.int8).to(device)))
            self.register_buffer("kbuf", torch.zeros(self.shape, dtype=torch.int8, device=device))
            self.soff = seed_off
            if os.environ.get("STATELESS") != "1":
                self.register_buffer("rng", xorshift_seed(self.shape, 17 + seed_off, device))
        self.proxy = None

    def material_w(self):
        if self.mode == "bitnet":
            return (self.weight + (torch.sign(self.weight) - self.weight).detach()) * self.scale
        if self.mode == "ed_naive":
            w = self.wb
        elif self.mode == "ed_compact":
            w = self.wb.to(torch.float32)
        else:
            w = unpack_bits(self.wpk, self.shape, self.kbuf.device)
        self.proxy = w.clone().requires_grad_(True)
        return self.proxy * self.scale

    def forward(self, x):
        if self.mode == "ed_packed" and getattr(self, "fused", None):
            return _FusedEDLinear.apply(x, self, self.fused[0], self.fused[1])
        return x @ self.material_w().T


class Block(nn.Module):
    def __init__(self, cfg, mode, gen, device, packact, li):
        super().__init__()
        self.n_head = cfg["heads"]
        self.packact = packact
        c = cfg["embd"]
        self.ln1 = nn.LayerNorm(c)
        self.ln2 = nn.LayerNorm(c)
        self.qkv = BitLinear(c, 3 * c, mode, gen, device, li * 4)
        self.attn_proj = BitLinear(c, c, mode, gen, device, li * 4 + 1)
        self.mlp_fc = BitLinear(c, 4 * c, mode, gen, device, li * 4 + 2)
        self.mlp_proj = BitLinear(4 * c, c, mode, gen, device, li * 4 + 3)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.attn_proj(y.transpose(1, 2).contiguous().view(B, T, C))
        z = self.mlp_fc(self.ln2(x))
        s = z.abs().mean().detach() + 1e-8
        fn = STEsignPacked if self.packact else STEsign
        h = fn.apply(z / s) * s
        return x + self.mlp_proj(h)


class GPT(nn.Module):
    def __init__(self, cfg, mode, gen, device, packact=False):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg["vocab"], cfg["embd"])
        self.wpe = nn.Embedding(cfg["block"], cfg["embd"])
        self.blocks = nn.ModuleList(
            Block(cfg, mode, gen, device, packact, i) for i in range(cfg["layers"])
        )
        self.ln_f = nn.LayerNorm(cfg["embd"])
        self.head = nn.Linear(cfg["embd"], cfg["vocab"], bias=False)
        self.to(device)

    def forward(self, idx, targets):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


def ed_update(model, lr, bits=8, step=0):
    rail2 = 2**bits - 1
    flips = 0
    for m in model.modules():
        if isinstance(m, BitLinear) and m.mode.startswith("ed") and m.proxy is not None:
            g = m.proxy.grad
            assert g is not None
            if m.mode == "ed_naive":
                u = -lr * g / m.scale
                t = m.ebuf + u
                flip = (t.sign() != m.wb.sign()) & (t.abs() >= 1.0)
                realized = torch.where(flip, 2.0 * t.sign(), torch.zeros_like(t))
                m.wb.copy_(torch.where(flip, -m.wb, m.wb))
                e = (t - realized).clamp_(-1, 1)
                qstep = 2 / rail2
                q = e / qstep
                fl = q.floor()
                r = torch.rand(q.shape, device=q.device)
                m.ebuf.copy_(((fl + (r < (q - fl)).float()) * qstep).clamp_(-1, 1))
                flips += int(flip.sum())
            else:
                w = m.wb.to(torch.int32) if m.mode == "ed_compact" else None
                if w is None:
                    wf = unpack_bits(m.wpk, m.shape, g.device)
                    w = wf.to(torch.int32)
                rail = 2 ** (bits - 1) - 1
                uq = (-lr / m.scale.item()) * g * rail
                ui = uq.floor()
                frac = uq - ui
                if hasattr(m, "rng"):
                    m.rng.copy_(xorshift_step(m.rng))
                    r = (m.rng >> 8).to(torch.float32) / float(1 << 24)
                else:
                    r = stateless_r(g.numel(), m.shape, m.soff, step, g.device)
                ui = (ui + (r < frac).float()).to(torch.int32)
                t = m.kbuf.to(torch.int32) + ui
                ts = torch.sign(t)
                flip = (ts != w) & (t.abs() >= rail)
                t = torch.where(flip, t - 2 * rail * ts.to(torch.int32), t)
                w = torch.where(flip, -w, w)
                m.kbuf.copy_(t.clamp(-rail, rail).to(torch.int8))
                if m.mode == "ed_compact":
                    m.wb.copy_(w.to(torch.int8))
                else:
                    m.wpk.copy_(pack_bits(w.to(torch.int8)))
                flips += int(flip.sum())
            m.proxy = None
    return flips


def state_bytes(model, opt):
    out = {"binary_w": 0, "buffer": 0, "latent": 0, "adam": 0, "fp_remnant": 0, "rng": 0}
    bl_params = set()
    for m in model.modules():
        if isinstance(m, BitLinear):
            if m.mode == "bitnet":
                out["latent"] += m.weight.numel() * m.weight.element_size()
                bl_params.add(id(m.weight))
            elif m.mode == "ed_naive":
                out["binary_w"] += m.wb.numel() * m.wb.element_size()
                out["buffer"] += m.ebuf.numel() * m.ebuf.element_size()
            elif m.mode == "ed_compact":
                out["binary_w"] += m.wb.numel()
                out["buffer"] += m.kbuf.numel()
                out["rng"] += m.rng.numel() * m.rng.element_size()
            else:
                out["binary_w"] += m.wpk.numel()
                out["buffer"] += m.kbuf.numel()
                out["rng"] += m.rng.numel() * m.rng.element_size()
    for p in model.parameters():
        if id(p) not in bl_params:
            out["fp_remnant"] += p.numel() * p.element_size()
    out["adam_latent"] = 0
    for prm, st in opt.state.items():
        key = "adam_latent" if id(prm) in bl_params else "adam"
        for v in st.values():
            if torch.is_tensor(v):
                out[key] += v.numel() * v.element_size()
    return out


class _FusedEDLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mod, lr, step):
        w = unpack_bits(mod.wpk, mod.shape, x.device)
        ctx.save_for_backward(x)
        ctx.mod = mod
        ctx.lr = lr
        ctx.step = step
        return x @ (w * mod.scale).T

    @staticmethod
    def backward(ctx, *gouts):
        (g,) = gouts
        (x,) = ctx.saved_tensors
        mod = ctx.mod
        w = unpack_bits(mod.wpk, mod.shape, g.device)
        grad_x = (g * mod.scale) @ w
        gw = g.reshape(-1, g.shape[-1]).T @ x.reshape(-1, x.shape[-1]) * mod.scale
        fused_int_update(mod, gw, ctx.lr, ctx.step)
        return grad_x, None, None, None


def fused_int_update(mod, g, lr, step, bits=8):
    rail = 2 ** (bits - 1) - 1
    w = unpack_bits(mod.wpk, mod.shape, g.device).to(torch.int32)
    uq = (-lr / mod.scale.item()) * g * rail
    ui = uq.floor()
    frac = uq - ui
    r = stateless_r(g.numel(), mod.shape, mod.soff, step, g.device)
    ui = (ui + (r < frac).float()).to(torch.int32)
    t = mod.kbuf.to(torch.int32) + ui
    ts = torch.sign(t)
    flip = (ts != w) & (t.abs() >= rail)
    t = torch.where(flip, t - 2 * rail * ts.to(torch.int32), t)
    w = torch.where(flip, -w, w)
    mod.kbuf.copy_(t.clamp(-rail, rail).to(torch.int8))
    mod.wpk.copy_(pack_bits(w.to(torch.int8)))
