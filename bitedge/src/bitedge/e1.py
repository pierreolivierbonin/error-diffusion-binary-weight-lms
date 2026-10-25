import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .adam8 import SGDM, Adam8
from .gpt2e import GPT, BitLinear, ed_update, pack_bits, state_bytes, unpack_bits
from .kernel import xorshift_seed, xorshift_step

ROOT = Path(__file__).resolve().parents[2]
PARENT = Path(os.environ.get("PARENT", "../bitdiffusion"))
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
LOG = Path(os.environ.get("LOG", ROOT / "results" / "e1.jsonl"))
CFG = {"vocab": 256, "block": 256, "layers": 4, "heads": 4, "embd": 256}
BATCH = 64
STEPS = int(os.environ.get("STEPS", "2000"))
EDLR = float(os.environ.get("EDLR", "60"))
EDLR_EMB = float(os.environ.get("EDLR_EMB", "60"))


class BinEmbedding(torch.nn.Module):
    def __init__(self, n, d, gen, device, seed_off):
        super().__init__()
        W0 = torch.randn(n, d, generator=gen) * 0.02
        s = torch.sign(W0) + (torch.sign(W0) == 0).float()
        self.shape = (n, d)
        self.register_buffer("scale", W0.abs().mean().to(device))
        self.register_buffer("wpk", pack_bits(s.to(torch.int8).to(device)))
        self.register_buffer("kbuf", torch.zeros(self.shape, dtype=torch.int8, device=device))
        self.register_buffer("rng", xorshift_seed(self.shape, 91 + seed_off, device))
        self.proxy = None

    def forward(self, idx):
        w = unpack_bits(self.wpk, self.shape, self.kbuf.device)
        self.proxy = w.clone().requires_grad_(True)
        return F.embedding(idx, self.proxy) * self.scale

    def update(self, lr, bits=8):
        g = self.proxy.grad
        if g is None:
            return 0
        rail = 2 ** (bits - 1) - 1
        w = unpack_bits(self.wpk, self.shape, g.device).to(torch.int32)
        uq = (-lr / self.scale.item()) * g * rail
        ui = uq.floor()
        frac = uq - ui
        self.rng.copy_(xorshift_step(self.rng))
        r = (self.rng >> 8).to(torch.float32) / float(1 << 24)
        ui = (ui + (r < frac).float()).to(torch.int32)
        t = self.kbuf.to(torch.int32) + ui
        ts = torch.sign(t)
        flip = (ts != w) & (t.abs() >= rail)
        t = torch.where(flip, t - 2 * rail * ts.to(torch.int32), t)
        w = torch.where(flip, -w, w)
        self.kbuf.copy_(t.clamp(-rail, rail).to(torch.int8))
        self.wpk.copy_(pack_bits(w.to(torch.int8)))
        self.proxy = None
        return int(flip.sum())


def load_data():
    raw = np.frombuffer((PARENT / "data" / "shakespeare.txt").read_bytes(), dtype=np.uint8)
    data = torch.from_numpy(raw.astype(np.int64))
    n = int(0.9 * len(data))
    return data[:n], data[n:]


def get_batch(data, gen):
    ix = torch.randint(len(data) - CFG["block"] - 1, (BATCH,), generator=gen)
    x = torch.stack([data[i : i + CFG["block"]] for i in ix]).to(DEVICE)
    y = torch.stack([data[i + 1 : i + CFG["block"] + 1] for i in ix]).to(DEVICE)
    return x, y


@torch.no_grad()
def val_loss(model, vdata):
    model.eval()
    g = torch.Generator().manual_seed(4242)
    losses = [model(*get_batch(vdata, g)).item() for _ in range(25)]
    model.train()
    return float(np.mean(losses))


def run(arm):
    gen = torch.Generator().manual_seed(0)
    model = GPT(CFG, "ed_packed", gen, DEVICE, packact=True)
    binemb = arm == "binemb"
    if binemb:
        model.wte = BinEmbedding(CFG["vocab"], CFG["embd"], gen, DEVICE, 0)
        head = BitLinear(CFG["embd"], CFG["vocab"], "ed_packed", gen, DEVICE, 99)
        model.head = head
    model.train()
    ln_params = [p for n, p in model.named_parameters() if "ln" in n]
    big_params = [p for n, p in model.named_parameters() if "ln" not in n]
    if arm == "fp32adam":
        opts = [torch.optim.AdamW(list(model.parameters()), lr=3e-4, betas=(0.9, 0.95))]
    elif arm == "int8adam":
        opts = [Adam8(big_params), SGDM(ln_params)]
    else:
        opts = [Adam8([model.wpe.weight]), SGDM(ln_params)]
    tdata, vdata = load_data()
    best = float("inf")
    t0 = time.time()
    for step in range(1, STEPS + 1):
        x, y = get_batch(tdata, gen)
        for o in opts:
            o.zero_grad()
        loss = model(x, y)
        loss.backward()
        for o in opts:
            o.step()
        ed_update(model, EDLR)
        if binemb:
            model.wte.update(EDLR_EMB)
        if step % 500 == 0:
            vl = val_loss(model, vdata)
            best = min(best, vl)
            print(f"  {arm} step {step} val {vl:.4f}", flush=True)
    sb = state_bytes(model, opts[0] if arm == "fp32adam" else type("O", (), {"state": {}})())
    opt_bytes = sum(o.state_bytes() for o in opts if hasattr(o, "state_bytes"))
    if binemb:
        for m in [model.wte]:
            sb["binary_w"] += m.wpk.numel()
            sb["buffer"] += m.kbuf.numel()
            sb["rng"] += m.rng.numel() * m.rng.element_size()
    total = sum(v for k, v in sb.items() if k != "rng") + opt_bytes
    rec = {
        "exp": "e1",
        "arm": arm,
        "edlr_emb": EDLR_EMB,
        "steps": STEPS,
        "best_val": round(best, 4),
        "state_bytes": sb,
        "opt_bytes_quant": opt_bytes,
        "train_state_mb_no_rng": round(total / 1e6, 3),
        "wall_s": round(time.time() - t0),
    }
    print(
        f"e1 {arm:9s} | best val {best:.4f} | train-state {rec['train_state_mb_no_rng']:.3f} MB "
        f"(rng excl: {sb['rng'] / 1e6:.1f}) | {rec['wall_s']}s",
        flush=True,
    )
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    for arm in ["fp32adam", "int8adam", "binemb"]:
        run(arm)


if __name__ == "__main__":
    main()
