import json
import os
import sys
import time
from pathlib import Path

import torch

from .gpt import GPT, BitLinear, GPTConfig, accountant, bop_update, buf_absmean, ed_update

ROOT = Path(__file__).resolve().parents[2]
LOG = Path(os.environ.get("LOG", ROOT / "results" / "gpt.jsonl"))


def log_rec(rec):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


torch.set_num_threads(int(os.environ.get("THREADS", "4")))

DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
SEED = int(os.environ.get("SEED", "0"))
STEPS = int(os.environ.get("STEPS", "2000"))
BATCH = int(os.environ.get("BATCH", "64"))
EDLR = float(os.environ.get("EDLR", "1.0"))
BITS = int(os.environ.get("BITS", "8"))
CLAMP = float(os.environ.get("CLAMP", "1.0"))
ADAMLR = float(os.environ.get("ADAMLR", "3e-4"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "250"))
EVAL_ITERS = int(os.environ.get("EVAL_ITERS", "50"))
PATIENCE = int(os.environ.get("PATIENCE", "0"))
MIN_DELTA = float(os.environ.get("MIN_DELTA", "0.005"))
ADAPT = float(os.environ.get("ADAPT", "0"))
EDSCHED = os.environ.get("EDSCHED", "const")
DIVERGE = float(os.environ.get("DIVERGE", "0"))
GAMMA = float(os.environ.get("GAMMA", "1e-2"))
TAU = float(os.environ.get("TAU", "1e-4"))
BOPBITS = int(os.environ["BOPBITS"]) if "BOPBITS" in os.environ else None
BOPALIGN = os.environ.get("BOPALIGN", "0") == "1"


def sched_lr(step):
    frac = step / STEPS
    if EDSCHED == "cosine":
        import math

        base = EDLR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
    elif EDSCHED == "linear":
        base = EDLR * (1 - 0.9 * frac)
    else:
        base = EDLR
    warm = int(os.environ.get("EDWARM", "0"))
    if warm and step < warm:
        base *= 0.1 + 0.9 * step / warm
    return base


def save_ck(model, path):
    ck = {"wb": {}, "scale": {}, "latent": {}, "fp": {}}
    blmods = {n for n, m in model.named_modules() if isinstance(m, BitLinear)}
    for n, m in model.named_modules():
        if isinstance(m, BitLinear):
            if m.mode == "bitnet":
                s = torch.sign(m.weight.detach())
                ck["wb"][n] = (s + (s == 0).float()).cpu()
                ck["latent"][n] = m.weight.detach().cpu()
                ck["scale"][n] = m.scale.detach().cpu()
            elif m.mode == "ed":
                ck["wb"][n] = m.wb.cpu()
                ck["scale"][n] = m.scale.cpu()
    for n, p in model.named_parameters():
        if n.rsplit(".", 1)[0] not in blmods:
            ck["fp"][n] = p.detach().cpu()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, path)


def load_ck(model, path):
    ck = torch.load(path, map_location=DEVICE)
    with torch.no_grad():
        for n, m in model.named_modules():
            if isinstance(m, BitLinear):
                if m.mode == "bitnet" and n in ck["latent"]:
                    m.weight.copy_(ck["latent"][n])
                    m.scale.copy_(ck["scale"][n])
                elif m.mode == "ed" and n in ck["wb"]:
                    m.wb.copy_(ck["wb"][n])
                    m.scale.copy_(ck["scale"][n])
        for n, p in model.named_parameters():
            if n in ck["fp"]:
                p.copy_(ck["fp"][n])


def save_full(model, opt, gen, step, best, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "gen": gen.get_state(),
            "trng": torch.get_rng_state(),
            "crng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "step": step,
            "best": best,
        },
        path,
    )


def load_full(model, opt, gen, path):
    ck = torch.load(path, map_location="cpu")
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    gen.set_state(ck["gen"])
    torch.set_rng_state(ck["trng"])
    if torch.cuda.is_available() and ck["crng"]:
        torch.cuda.set_rng_state_all(ck["crng"])
    return ck["step"], ck["best"]


def load_data():
    which = os.environ.get("DATA", "shakespeare")
    if which == "enwik8":
        import numpy as np

        raw = np.frombuffer((ROOT / "data" / "enwik8").read_bytes(), dtype=np.uint8)
        data = torch.from_numpy(raw.astype(np.int64))
        n = int(0.9 * len(data))
        v = int(0.95 * len(data))
        return data[:n], data[n:v], 256
    if which == "shakespeare_bytes":
        import numpy as np

        raw = np.frombuffer((ROOT / "data" / "shakespeare.txt").read_bytes(), dtype=np.uint8)
        data = torch.from_numpy(raw.astype(np.int64))
        n = int(0.9 * len(data))
        return data[:n], data[n:], 256
    if which == "wikitext103":
        import numpy as np

        raw = np.fromfile(ROOT / "data" / "wt103_gpt2.bin", dtype=np.uint16)
        data = torch.from_numpy(raw.astype(np.int64))
        n = int(0.9 * len(data))
        v = int(0.95 * len(data))
        return data[:n], data[n:v], 50257
    text = (ROOT / "data" / "shakespeare.txt").read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], len(chars)


def get_batch(data, block, batch, gen):
    ix = torch.randint(len(data) - block - 1, (batch,), generator=gen)
    x = torch.stack([data[i : i + block] for i in ix]).to(DEVICE)
    y = torch.stack([data[i + 1 : i + block + 1] for i in ix]).to(DEVICE)
    return x, y


@torch.no_grad()
def val_loss(model, vdata, block):
    model.eval()
    g = torch.Generator().manual_seed(4242)
    losses = []
    for _ in range(EVAL_ITERS):
        x, y = get_batch(vdata, block, BATCH, g)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    mode = sys.argv[1]
    cfg = GPTConfig(
        vocab_size=0,
        block_size=int(os.environ.get("BLOCK", "256")),
        n_layer=int(os.environ.get("LAYERS", "4")),
        n_head=int(os.environ.get("HEADS", "4")),
        n_embd=int(os.environ.get("EMBD", "256")),
        binact=os.environ.get("BINACT", "0") == "1",
        qgrad=os.environ.get("QGRAD", "fp"),
        dropout=float(os.environ.get("DROPOUT", "0")),
        act=os.environ.get("ACT", "gelu"),
    )
    tdata, vdata, vocab = load_data()
    cfg.vocab_size = vocab
    if cfg.dropout > 0:
        torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    model = GPT(cfg, mode, gen, DEVICE)
    if os.environ.get("LOADCK"):
        load_ck(model, os.environ["LOADCK"])
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=ADAMLR, betas=(0.9, 0.95))
    binary, fp = accountant(model)
    base = {
        "mode": mode,
        "embd": cfg.n_embd,
        "layers": cfg.n_layer,
        "block": cfg.block_size,
        "seed": SEED,
        "steps": STEPS,
        "edlr": EDLR,
        "bits": BITS,
        "clamp": CLAMP,
        "binact": cfg.binact,
        "qgrad": cfg.qgrad,
        "dropout": cfg.dropout,
        "act": cfg.act,
        "qblock": int(os.environ.get("QBLOCK", "32")),
        "qclip": float(os.environ.get("QCLIP", "3")),
        "adapt": ADAPT,
        "edsched": EDSCHED,
        "binary_params": binary,
        "fp_params": fp,
        "device": DEVICE,
        "data": os.environ.get("DATA", "shakespeare"),
        "loadck": os.environ.get("LOADCK", ""),
        "gamma": GAMMA,
        "tau": TAU,
        "bopbits": BOPBITS,
        "bopalign": BOPALIGN,
    }
    print(
        f"[gpt] mode={mode} embd={cfg.n_embd} L={cfg.n_layer} vocab={vocab} "
        f"binary={binary / 1e6:.2f}M fp={fp / 1e6:.2f}M seed={SEED} steps={STEPS} "
        f"edlr={EDLR} bits={BITS} C={CLAMP} binact={cfg.binact} qgrad={cfg.qgrad} "
        f"drop={cfg.dropout} dev={DEVICE}",
        flush=True,
    )
    best = float("inf")
    flips = 0
    stale = 0
    start = 1
    early = {}
    if os.environ.get("RESUME"):
        rstep, best = load_full(model, opt, gen, os.environ["RESUME"])
        start = rstep + 1
        print(f"resumed {os.environ['RESUME']} step {rstep} best {best:.4f}", flush=True)
        log_rec(base | {"type": "resume", "step": rstep, "best_val": round(best, 4)})
    t0 = time.time()
    if os.environ.get("EVAL0") == "1" and start == 1:
        vl0 = val_loss(model, vdata, cfg.block_size)
        best = vl0
        print(f"step     0 | val {vl0:.4f} (zero-shot)", flush=True)
        log_rec(base | {"type": "eval0", "step": 0, "val_loss": round(vl0, 4)})
    for step in range(start, STEPS + 1):
        x, y = get_batch(tdata, cfg.block_size, BATCH, gen)
        opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        if mode == "ed":
            flips += ed_update(model, sched_lr(step), CLAMP, BITS, gen, ADAPT)
        elif mode == "bop":
            flips += bop_update(model, GAMMA, TAU, BOPBITS, BOPALIGN, gen)
        if step % EVAL_EVERY == 0 or step == STEPS:
            vl = val_loss(model, vdata, cfg.block_size)
            stale = 0 if vl < best - MIN_DELTA else stale + 1
            best = min(best, vl)
            print(
                f"step {step:5d} | train {loss.item():.4f} | val {vl:.4f} | "
                f"flips/step {flips / EVAL_EVERY:.0f} | buf {buf_absmean(model):.3f} | "
                f"{(time.time() - t0):.0f}s",
                flush=True,
            )
            rec = {
                "type": "eval",
                "step": step,
                "train_loss": round(loss.item(), 4),
                "val_loss": round(vl, 4),
                "flips_per_step": round(flips / EVAL_EVERY, 1),
                "buf_absmean": round(buf_absmean(model), 4),
                "elapsed_s": round(time.time() - t0),
            }
            if mode == "bop":
                rec["ema_absmean"] = buf_absmean(model)
            log_rec(base | rec)
            flips = 0
            if DIVERGE and step >= 2000 and vl > best + DIVERGE:
                print(f"DIVERGED step {step} (val {vl:.4f} > best {best:.4f} + {DIVERGE})", flush=True)
                log_rec(base | {"type": "stall", "step": step, "val_loss": round(vl, 4)})
                sys.exit(3)
            if os.environ.get("STALLCHECK") == "1" and step in (1000, 2000, 3000):
                early[step] = vl
                prev = early.get(step - 1000)
                if prev is not None and vl > prev:
                    print(f"STALL step {step} (val {vl:.4f} > {prev:.4f})", flush=True)
                    log_rec(base | {"type": "stall", "step": step, "val_loss": round(vl, 4)})
                    sys.exit(3)
            if os.environ.get("FULLCK"):
                save_full(model, opt, gen, step, best, os.environ["FULLCK"])
            if PATIENCE and stale >= PATIENCE:
                print(f"EARLY-STOP step {step} (no val gain > {MIN_DELTA} for {PATIENCE} evals)", flush=True)
                log_rec(base | {"type": "early_stop", "step": step})
                break
    qb, qc = os.environ.get("QBLOCK", "32"), os.environ.get("QCLIP", "3")
    print(
        f"FINAL mode={mode} embd={cfg.n_embd} seed={SEED} best_val={best:.4f} "
        f"edlr={EDLR} bits={BITS} C={CLAMP} binact={cfg.binact} qgrad={cfg.qgrad} "
        f"qblock={qb} qclip={qc}",
        flush=True,
    )
    log_rec(base | {"type": "final", "best_val": round(best, 4)})
    if os.environ.get("SAVECK"):
        save_ck(model, os.environ["SAVECK"])
        print(f"saved deploy checkpoint -> {os.environ['SAVECK']}", flush=True)


if __name__ == "__main__":
    main()
