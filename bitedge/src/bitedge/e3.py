import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from .adam8 import SGDM, Adam8
from .gpt2e import GPT, BitLinear, ed_update, pack_bits

FUSED = os.environ.get("FUSED") == "1"

ROOT = Path(__file__).resolve().parents[2]
PARENT = Path(os.environ.get("PARENT", "../bitdiffusion"))
CK = Path(os.environ.get("CK", PARENT / "results" / "ck_enwik8_bitnet_512.pt"))
ART = Path(os.environ.get("ART", ROOT / "results" / "artifact_512.pt"))
LOG = Path(os.environ.get("LOG", ROOT / "results" / "e3.jsonl"))
CFG = {
    "vocab": 256,
    "block": int(os.environ.get("BLOCK", "128")),
    "layers": int(os.environ.get("LAYERS", "6")),
    "heads": int(os.environ.get("HEADS", "8")),
    "embd": int(os.environ.get("EMBD", "512")),
}
BATCH = int(os.environ.get("BATCH", "8"))
STEPS = int(os.environ.get("STEPS", "2000"))
EDLR = float(os.environ.get("EDLR", "120"))
CAP_MB = float(os.environ.get("CAP_MB", "250"))


def rss_mb():
    return psutil.Process().memory_info().rss / 1e6


def load_ck(model, arm):
    ck = torch.load(ART if arm == "edge" else CK, map_location="cpu")
    with torch.no_grad():
        for n, m in model.named_modules():
            if isinstance(m, BitLinear):
                if arm == "latent" and n in ck["latent"]:
                    m.weight.copy_(ck["latent"][n])
                    m.scale.copy_(ck["scale"][n])
                elif arm == "edge" and n in ck["wpk"]:
                    m.wpk.copy_(ck["wpk"][n])
                    m.scale.copy_(ck["scale"][n])
        for n, p in model.named_parameters():
            if n in ck["fp"]:
                src = ck["fp"][n]
                if src.shape != p.shape:
                    src = src[: p.shape[0]]
                p.copy_(src)


def load_data():
    raw = np.frombuffer((PARENT / "data" / "shakespeare.txt").read_bytes(), dtype=np.uint8)
    data = torch.from_numpy(raw.astype(np.int64))
    n = int(0.9 * len(data))
    return data[:n], data[n:]


def get_batch(data, gen):
    ix = torch.randint(len(data) - CFG["block"] - 1, (BATCH,), generator=gen)
    x = torch.stack([data[i : i + CFG["block"]] for i in ix])
    y = torch.stack([data[i + 1 : i + CFG["block"] + 1] for i in ix])
    return x, y


@torch.no_grad()
def val_loss(model, vdata):
    model.eval()
    g = torch.Generator().manual_seed(4242)
    losses = [model(*get_batch(vdata, g)).item() for _ in range(50)]
    model.train()
    return float(np.mean(losses))


def run(arm):
    base_rss = rss_mb()
    peak = base_rss
    gen = torch.Generator().manual_seed(0)
    rec = {"exp": "e3", "arm": arm, "fused": FUSED, "cap_mb": CAP_MB, "base_rss_mb": round(base_rss, 1), "edlr": EDLR}
    try:
        mode = "bitnet" if arm == "latent" else "ed_packed"
        model = GPT(CFG, mode, gen, "cpu", packact=(arm == "edge"))
        load_ck(model, arm)
        model.train()
        if arm == "latent":
            opts = [torch.optim.AdamW(list(model.parameters()), lr=3e-4, betas=(0.9, 0.95))]
        else:
            ln = [p for n, p in model.named_parameters() if "ln" in n]
            big = [p for n, p in model.named_parameters() if "ln" not in n]
            opts = [Adam8(big), SGDM(ln, lr=float(os.environ.get("LNLR", "1e-2")))]
        tdata, vdata = load_data()
        zero = val_loss(model, vdata)
        rec["zero_shot"] = round(zero, 4)
        setup_rss = rss_mb()
        peak = setup_rss
        rec["setup_rss_mb"] = round(setup_rss, 1)
        print(f"e3 {arm}: zero-shot {zero:.4f} | setup rss {setup_rss:.0f} MB", flush=True)
        best = zero
        t0 = time.time()
        bls = [m for m in model.modules() if isinstance(m, BitLinear)]
        for step in range(1, STEPS + 1):
            if arm == "edge" and FUSED:
                for m in bls:
                    m.fused = (EDLR, step)
            x, y = get_batch(tdata, gen)
            for o in opts:
                o.zero_grad()
            loss = model(x, y)
            loss.backward()
            for o in opts:
                o.step()
            if arm == "edge" and not FUSED:
                ed_update(model, EDLR, step=step)
            if step % 200 == 0:
                gc.collect()
            peak = max(peak, rss_mb())
            if peak - setup_rss > CAP_MB:
                raise MemoryError(
                    f"training RSS {peak - setup_rss:.0f} MB over setup exceeds cap {CAP_MB:.0f} MB at step {step}"
                )
            if step == 250 and os.environ.get("PROBE") == "1":
                from .gpt2e import BitLinear as BL
                rails = 127.0
                bufs = [
                    m.kbuf.abs().to(torch.float32).mean().item() / rails
                    for m in model.modules()
                    if isinstance(m, BL) and hasattr(m, "kbuf")
                ]
                drive = float(np.mean(bufs)) / 250
                print(f"PROBE lr={EDLR} drive={drive:.2e}", flush=True)
                rec |= {"probe_drive": drive}
                break
            if step % 500 == 0:
                vl = val_loss(model, vdata)
                best = min(best, vl)
                print(f"  {arm} step {step} val {vl:.4f} | rss {rss_mb():.0f} MB", flush=True)
        rec |= {
            "status": "completed",
            "best_val": round(best, 4),
            "peak_rss_mb": round(peak, 1),
            "train_delta_mb": round(peak - rec.get("setup_rss_mb", base_rss), 1),
            "steps_per_s": round(STEPS / (time.time() - t0), 2),
        }
    except MemoryError as e:
        rec |= {"status": "CAP_EXCEEDED", "detail": str(e), "peak_rss_mb": round(peak, 1), "train_delta_mb": round(peak - rec.get("setup_rss_mb", base_rss), 1)}
    print(f"e3 {arm}: {rec['status']} | peak rss {rec['peak_rss_mb']} MB | {rec.get('best_val', '-')}", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    run(os.environ.get("ARM", "edge"))


if __name__ == "__main__":
    main()
