import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .kernel import ed_step_float, ed_step_int, fwd, xorshift_seed
from .tasks import teacher_student

D_IN, D_OUT, N_BIN = 64, 10, 2
D_H = int(os.environ.get("DH", "192"))
EPOCHS = int(os.environ.get("EPOCHS", "25"))
BATCH = int(os.environ.get("BATCH", "128"))
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
LOG = Path(os.environ.get("LOG", Path(__file__).resolve().parents[2] / "results" / "e0.jsonl"))

Xtr, Xte, ytr, yte = teacher_student(device=DEVICE)


def run(seed, mode, bits, lr=1.0):
    gen = torch.Generator().manual_seed(seed)
    Wproj = (torch.randn(D_H, D_IN, generator=gen) / np.sqrt(D_IN)).to(DEVICE)
    Wmids0 = [(torch.randn(D_H, D_H, generator=gen) / np.sqrt(D_H)).to(DEVICE) for _ in range(N_BIN)]
    Wout = (torch.randn(D_OUT, D_H, generator=gen) / np.sqrt(D_H)).to(DEVICE).requires_grad_(True)
    bout = torch.zeros(D_OUT, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([Wout, bout], lr=3e-3)
    scales = [W.abs().mean().item() for W in Wmids0]
    wb = [torch.sign(W) + (torch.sign(W) == 0).float() for W in Wmids0]
    if mode == "int":
        w8 = [w.to(torch.int8) for w in wb]
        k8 = [torch.zeros_like(W, dtype=torch.int8) for W in Wmids0]
        rng = [xorshift_seed(W.shape, seed * 7 + i + 1, DEVICE) for i, W in enumerate(Wmids0)]
    else:
        ebuf = [torch.zeros_like(W) for W in Wmids0]
    flips = 0
    for _ in range(EPOCHS):
        idx = torch.randperm(len(Xtr), generator=gen)
        for i in range(0, len(Xtr), BATCH):
            j = idx[i : i + BATCH].to(DEVICE)
            opt.zero_grad()
            if mode == "int":
                proxies = [w.to(torch.float32).requires_grad_(True) for w in w8]
            else:
                proxies = [w.clone().requires_grad_(True) for w in wb]
            mids = [p * s for p, s in zip(proxies, scales, strict=False)]
            torch.nn.functional.cross_entropy(fwd(Xtr[j], Wproj, mids, Wout, bout), ytr[j]).backward()
            opt.step()
            for kk in range(N_BIN):
                g = proxies[kk].grad
                assert g is not None
                if mode == "int":
                    w8[kk], k8[kk], rng[kk], nf = ed_step_int(
                        w8[kk], k8[kk], rng[kk], g, scales[kk], lr, bits, int(os.environ.get("RAIL", "0"))
                    )
                else:
                    wb[kk], ebuf[kk], nf = ed_step_float(wb[kk], ebuf[kk], g, scales[kk], lr, bits, gen)
                flips += nf
    final = [w8[i].to(torch.float32) * scales[i] for i in range(N_BIN)] if mode == "int" else [
        wb[i] * scales[i] for i in range(N_BIN)
    ]
    with torch.no_grad():
        acc = (fwd(Xte, Wproj, final, Wout, bout).argmax(1) == yte).float().mean().item()
    return acc, flips


def main():
    bits = int(sys.argv[sys.argv.index("bits") + 1]) if "bits" in sys.argv else 8
    lr = float(sys.argv[sys.argv.index("lr") + 1]) if "lr" in sys.argv else 1.0
    for mode in ["float", "int"]:
        t0 = time.time()
        accs = []
        fl = []
        for s in SEEDS:
            a, f = run(s, mode, bits, lr)
            accs.append(a)
            fl.append(f)
        rec = {
            "exp": "e0",
            "mode": mode,
            "bits": bits,
            "lr": lr,
            "width": D_H,
            "seeds": SEEDS,
            "acc_mean": round(float(np.mean(accs)) * 100, 2),
            "acc_std": round(float(np.std(accs)) * 100, 2),
            "flips_total_mean": round(float(np.mean(fl))),
            "wall_s": round(time.time() - t0),
        }
        print(
            f"e0 {mode:5s} b{bits} lr={lr} acc {rec['acc_mean']:5.2f} ± {rec['acc_std']:4.2f} "
            f"| {rec['wall_s']}s",
            flush=True,
        )
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
