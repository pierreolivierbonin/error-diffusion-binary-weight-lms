import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .gpt2e import GPT, ed_update, state_bytes

ROOT = Path(__file__).resolve().parents[2]
PARENT = Path(os.environ.get("PARENT", "../bitdiffusion"))
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
LOG = Path(os.environ.get("LOG", ROOT / "results" / "e2.jsonl"))
CFG = {
    "vocab": 256,
    "block": int(os.environ.get("BLOCK", "256")),
    "layers": int(os.environ.get("LAYERS", "4")),
    "heads": int(os.environ.get("HEADS", "4")),
    "embd": int(os.environ.get("EMBD", "256")),
}
BATCH = int(os.environ.get("BATCH", "64"))
STEPS = int(os.environ.get("STEPS", "200"))
EDLR = float(os.environ.get("EDLR", "60"))


def load_data():
    raw = np.frombuffer((PARENT / "data" / "shakespeare.txt").read_bytes(), dtype=np.uint8)
    return torch.from_numpy(raw.astype(np.int64))


def get_batch(data, gen):
    ix = torch.randint(len(data) - CFG["block"] - 1, (BATCH,), generator=gen)
    x = torch.stack([data[i : i + CFG["block"]] for i in ix]).to(DEVICE)
    y = torch.stack([data[i + 1 : i + CFG["block"] + 1] for i in ix]).to(DEVICE)
    return x, y


def run(arm, packact):
    gen = torch.Generator().manual_seed(0)
    mode = "bitnet" if arm == "latent" else arm
    model = GPT(CFG, mode, gen, DEVICE, packact=packact)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
    data = load_data()
    losses = []
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for step in range(STEPS):
        x, y = get_batch(data, gen)
        opt.zero_grad()
        loss = model(x, y)
        loss.backward()
        opt.step()
        if mode != "bitnet":
            ed_update(model, EDLR)
        losses.append(loss.item())
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() if DEVICE == "cuda" else 0
    sb = state_bytes(model, opt)
    nbin = sum(
        m.shape[0] * m.shape[1]
        for m in model.modules()
        if hasattr(m, "shape") and hasattr(m, "material_w")
    )
    persist = sum(sb.values())
    rec = {
        "exp": "e2",
        "arm": arm,
        "packact": packact,
        "cfg": CFG,
        "steps": STEPS,
        "state_bytes": sb,
        "persistent_state_mb": round(persist / 1e6, 3),
        "binary_layer_bits_per_wt": round((sb["binary_w"] + sb["buffer"] + sb["latent"] + sb["adam_latent"]) * 8 / nbin, 2),
        "peak_cuda_mb": round(peak / 1e6, 1),
        "steps_per_s": round(STEPS / wall, 2),
        "loss_first10": round(float(np.mean(losses[:10])), 3),
        "loss_last10": round(float(np.mean(losses[-10:])), 3),
    }
    print(
        f"e2 {arm:11s} packact={int(packact)} | state {rec['persistent_state_mb']:7.2f} MB "
        f"({rec['binary_layer_bits_per_wt']:6.2f} b/wt binary-layer) | peak {rec['peak_cuda_mb']:7.1f} MB "
        f"| {rec['steps_per_s']:5.2f} st/s | loss {rec['loss_first10']} -> {rec['loss_last10']}",
        flush=True,
    )
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    for arm, packact in [
        ("latent", False),
        ("ed_naive", False),
        ("ed_compact", False),
        ("ed_packed", False),
        ("ed_compact", True),
    ]:
        run(arm, packact)


if __name__ == "__main__":
    main()
