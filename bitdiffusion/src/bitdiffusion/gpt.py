import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .layers import STEsign, get_rht, qgrad_fn
from .optim import ed_step


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    binact: bool = False
    qgrad: str = "fp"
    dropout: float = 0.0
    act: str = "gelu"


class _ScopedQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, mod):
        ctx.save_for_backward(x, w)
        ctx.mod = mod
        return x @ w.T

    @staticmethod
    def backward(ctx, *gouts):
        (g,) = gouts
        x, w = ctx.saved_tensors
        mod = ctx.mod
        d = g.shape[-1]
        gq = qgrad_fn(g.reshape(-1, d), mod.qgrad, mod.rht, mod.qgen).reshape(g.shape)
        grad_x = g @ w
        grad_w = gq.reshape(-1, d).T @ x.reshape(-1, x.shape[-1])
        return grad_x, grad_w, None


UCLIP = float(os.environ.get("UCLIP", "0"))


def _ebuf_init(W0, gen):
    a = float(os.environ.get("EDINIT", "0"))
    if a == 0:
        return torch.zeros_like(W0)
    return (torch.rand(W0.shape, generator=gen) * 2 - 1) * a


class BitLinear(nn.Module):
    wb: torch.Tensor
    ebuf: torch.Tensor
    scale: torch.Tensor

    def __init__(self, d_in, d_out, mode, gen, device, qgrad="fp"):
        super().__init__()
        W0 = torch.randn(d_out, d_in, generator=gen) / np.sqrt(d_in)
        self.mode = mode
        self.qgrad = qgrad
        self.qgen = gen
        self.qscope = os.environ.get("QSCOPE", "all")
        qblock = int(os.environ.get("QBLOCK", "32"))
        self.rht = get_rht(d_out, gen, device, qblock) if qgrad.startswith("sr1_rht") else None
        if mode == "fp32":
            self.weight = nn.Parameter(W0.to(device))
        elif mode == "bitnet":
            self.weight = nn.Parameter(W0.to(device))
            self.register_buffer("scale", W0.abs().mean().to(device))
        else:
            wb0 = torch.sign(W0) + (torch.sign(W0) == 0).float()
            self.register_buffer("wb", wb0.to(device))
            self.register_buffer("ebuf", _ebuf_init(W0, gen).to(device))
            self.register_buffer("scale", W0.abs().mean().to(device))
            self.proxy = None

    def _qhook(self, g):
        d = g.shape[-1]
        return qgrad_fn(g.reshape(-1, d), self.qgrad, self.rht, self.qgen).reshape(g.shape)

    def forward(self, x):
        if self.mode == "fp32":
            w = self.weight
        elif self.mode == "bitnet":
            w = (self.weight + (torch.sign(self.weight) - self.weight).detach()) * self.scale
        elif self.training:
            self.proxy = self.wb.clone().requires_grad_(True)
            w = self.proxy * self.scale
        else:
            w = self.wb * self.scale
        if self.training and self.qgrad != "fp" and self.qscope == "w" and w.requires_grad:
            return _ScopedQuant.apply(x, w, self)
        out = x @ w.T
        if self.training and self.qgrad != "fp" and out.requires_grad:
            out.register_hook(self._qhook)
        return out


class Block(nn.Module):
    def __init__(self, cfg, mode, gen, device):
        super().__init__()
        self.n_head = cfg.n_head
        self.binact = cfg.binact
        self.act = cfg.act
        self.dropout = cfg.dropout
        self.diag = os.environ.get("DIAG") == "1"
        self.drop = nn.Dropout(cfg.dropout)
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.qkv = BitLinear(cfg.n_embd, 3 * cfg.n_embd, mode, gen, device, cfg.qgrad)
        self.attn_proj = BitLinear(cfg.n_embd, cfg.n_embd, mode, gen, device, cfg.qgrad)
        self.mlp_fc = BitLinear(cfg.n_embd, 4 * cfg.n_embd, mode, gen, device, cfg.qgrad)
        self.mlp_proj = BitLinear(4 * cfg.n_embd, cfg.n_embd, mode, gen, device, cfg.qgrad)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.drop(self.attn_proj(y))
        z = self.mlp_fc(self.ln2(x))
        if self.binact:
            s = z.abs().mean().detach() + 1e-8
            if self.diag:
                self.last_s = float(s)
            h = STEsign.apply(z / s) * s
        elif self.act == "hardtanh":
            s = z.abs().mean().detach() + 1e-8
            h = (z / s).clamp(-1.0, 1.0) * s
        elif self.act == "tanh":
            s = z.abs().mean().detach() + 1e-8
            h = torch.tanh(z / s) * s
        else:
            h = F.gelu(z)
        x = x + self.drop(self.mlp_proj(h))
        return x


class GPT(nn.Module):
    def __init__(self, cfg, mode, gen, device):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        with torch.no_grad():
            self.wte.weight.copy_(torch.randn(cfg.vocab_size, cfg.n_embd, generator=gen) * 0.02)
            self.wpe.weight.copy_(torch.randn(cfg.block_size, cfg.n_embd, generator=gen) * 0.02)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg, mode, gen, device) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.randn(cfg.vocab_size, cfg.n_embd, generator=gen) * 0.02)
        self.to(device)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


def ed_update(model, lr, clamp, buf_bits, gen, adapt=0.0):
    flips = 0
    for m in model.modules():
        if isinstance(m, BitLinear) and m.mode == "ed" and m.proxy is not None:
            g = m.proxy.grad
            assert g is not None
            if os.environ.get("CNT") == "1":
                c = (m.ebuf - g.sign()).clamp_(-127.0, 127.0)
                flip = (c.abs() >= 127.0) & (c.sign() != m.wb.sign())
                m.wb.copy_(torch.where(flip, -m.wb, m.wb))
                m.ebuf.copy_(torch.where(flip, torch.zeros_like(c), c))
                m.proxy = None
                flips += int(flip.sum())
                continue
            if adapt:
                r = g.pow(2).mean().sqrt()
                prev = getattr(m, "gnorm", r)
                m.gnorm = (1 - adapt) * prev + adapt * r
                drive_scale = m.gnorm + 1e-12
            else:
                drive_scale = m.scale
            wb, ebuf, nf = ed_step(m.wb, m.ebuf, g, drive_scale, lr, clamp, buf_bits, "edB", gen, UCLIP)
            m.wb.copy_(wb)
            m.ebuf.copy_(ebuf)
            m.proxy = None
            flips += nf
    return flips


def diag_stats(model):
    ln = [p.detach() for n, p in model.named_parameters() if ".ln" in n and n.endswith("weight")]
    gfp = [p.grad.detach().norm() ** 2 for p in model.parameters() if p.grad is not None]
    gbin = [
        m.proxy.grad.detach().norm() ** 2
        for m in model.modules()
        if isinstance(m, BitLinear) and m.mode == "ed" and m.proxy is not None and m.proxy.grad is not None
    ]
    return {
        "ln_gain_mean": round(float(torch.stack([w.abs().mean() for w in ln]).mean()), 4),
        "ln_gain_max": round(float(max(w.abs().max() for w in ln)), 3),
        "head_norm": round(float(model.head.weight.detach().norm()), 3),
        "wte_norm": round(float(model.wte.weight.detach().norm()), 3),
        "grad_fp": round(float(sum(gfp).sqrt()), 5) if gfp else 0.0,
        "grad_bin": round(float(sum(gbin).sqrt()), 5) if gbin else 0.0,
        "act_scale": [round(b.last_s, 5) for b in model.blocks if hasattr(b, "last_s")],
    }


def buf_absmean(model):
    mods = [m for m in model.modules() if isinstance(m, BitLinear) and m.mode == "ed"]
    bufs = [m.ebuf.abs().mean().item() for m in mods]
    return float(np.mean(bufs)) if bufs else 0.0


def accountant(model):
    binary = fp = 0
    for m in model.modules():
        if isinstance(m, BitLinear) and m.mode != "fp32":
            if m.mode == "ed":
                binary += m.wb.numel()
            else:
                binary += m.weight.numel()
    for p in model.parameters():
        fp += p.numel()
    return binary, fp
