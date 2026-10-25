import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .layers import fwd, get_rht
from .optim import (
    bop_step,
    cnt_step,
    ed2_step,
    ed_hyst_step,
    ed_spatial_step,
    ed_step,
    ede_step,
    edm_step,
    intlat_step,
)
from .tasks import teacher_student

torch.set_num_threads(int(os.environ.get("THREADS", "4")))

D_IN, D_OUT, N_BIN = 64, 10, 2
D_H = int(os.environ.get("DH", "96"))
EPOCHS = int(os.environ.get("EPOCHS", "25"))
BATCH = int(os.environ.get("BATCH", "128"))
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1").split(",")]
TRACE = os.environ.get("TRACE")
TRACE_N = int(os.environ.get("TRACE_N", "32"))
TRACE_SEED = int(os.environ.get("TRACE_SEED", "2026"))

Xtr, Xte, ytr, yte = teacher_student(device=DEVICE)


def _g9(t):
    return [float(f"{v:.9g}") for v in t.tolist()]


def _trace_open(seed, mode, lr, clamp, buf_bits, binact):
    idx = np.random.default_rng(TRACE_SEED).choice(D_H * D_H, TRACE_N, replace=False)
    head = {
        "type": "header",
        "seed": seed,
        "mode": mode,
        "width": D_H,
        "batch": BATCH,
        "epochs": EPOCHS,
        "device": DEVICE,
        "lr": lr,
        "clamp": clamp,
        "buf_bits": buf_bits,
        "binact": binact,
        "layer": 0,
        "trace_seed": TRACE_SEED,
        "idx": [int(i) for i in idx],
    }
    return torch.from_numpy(idx).to(DEVICE), [head]


def _trace_close(rows):
    with open(f"{TRACE}.seed{rows[0]['seed']}.jsonl", "x", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def setup(seed):
    gen = torch.Generator().manual_seed(seed)
    rhts = [get_rht(D_H, gen, DEVICE) for _ in range(N_BIN)]
    Wproj = (torch.randn(D_H, D_IN, generator=gen) / np.sqrt(D_IN)).to(DEVICE)
    Wmids = [(torch.randn(D_H, D_H, generator=gen) / np.sqrt(D_H)).to(DEVICE) for _ in range(N_BIN)]
    Wout = (torch.randn(D_OUT, D_H, generator=gen) / np.sqrt(D_H)).to(DEVICE).requires_grad_(True)
    bout = torch.zeros(D_OUT, device=DEVICE, requires_grad=True)
    return gen, rhts, Wproj, Wmids, Wout, bout


def acc(Wproj, mids, Wout, bout, binact):
    with torch.no_grad():
        return (fwd(Xte, Wproj, mids, Wout, bout, binact=binact).argmax(1) == yte).float().mean().item()


def batches(gen):
    idx = torch.randperm(len(Xtr), generator=gen)
    for i in range(0, len(Xtr), BATCH):
        j = idx[i : i + BATCH].to(DEVICE)
        yield Xtr[j], ytr[j]


def dead_units(Wproj, mids, binact):
    with torch.no_grad():
        h = F.relu(Xte @ Wproj.T)
        out = []
        for W in mids:
            z = h @ W.T
            if binact:
                s = z.abs().mean() + 1e-8
                h = torch.sign(z / s) * s
            else:
                h = F.relu(z)
            out.append(float((h.var(0) < 1e-10).float().mean()))
        return out


def state_bits(mode, buf_bits, bop_bits=None, mom_bits=0):
    if mode == "bop" and bop_bits is not None:
        return (1 + bop_bits, f"wb 1 + EMA {bop_bits}")
    return {
        "fp32_mid": (96, "FP32 wt 32 + Adam 64"),
        "bitnet": (97, "wb 1 + latent 32 + Adam 64"),
        "bop": (33, "wb 1 + FP EMA 32"),
        "edA": (33, "wb 1 + FP buf 32"),
        "edB": (1 + buf_bits, f"wb 1 + buf {buf_bits}"),
        "edB_det": (1 + buf_bits, f"wb 1 + buf {buf_bits}"),
        "edS": (1 + buf_bits, f"wb 1 + buf {buf_bits}, spatial"),
        "edH": (2 + buf_bits, f"wb 1 + buf {buf_bits} + refract 1"),
        "ed2": (1 + 2 * buf_bits, f"wb 1 + 2 cascaded bufs {buf_bits} each"),
        "intlat": (buf_bits, f"integer latent {buf_bits}b, sign readout"),
        "edE": (1 + mom_bits if mom_bits else 33, f"wb 1 + leaky resid-in-mom {mom_bits or 'FP32'}"),
        "cnt": (1 + buf_bits, f"wb 1 + sign counter {buf_bits}b"),
        "edM": (1 + buf_bits + mom_bits, f"wb 1 + buf {buf_bits} + mom {mom_bits}"),
    }.get(mode, (1, "wb only"))


def _grad(t):
    g = t.grad
    assert g is not None
    return g


def _diag(ff, fl, buf, gv, gvn, Wproj, mids, binact):
    d = {}
    if ff is not None:
        d["flips_ep0"], d["flips_epN"] = ff, fl
    if buf is not None:
        d["buf"] = float(torch.cat([e.flatten() for e in buf]).abs().mean())
    if gvn:
        d["gradvar"] = float(np.mean([g / gvn for g in gv]))
    d["dead"] = float(np.mean(dead_units(Wproj, mids, binact)))
    return d


def run(seed, mode, lr=1.0, clamp=1.5, buf_bits=8, binact=False, qgrad="fp", **kwargs):
    gen, rhts, Wproj, Wmids0, Wout, bout = setup(seed)
    ggen = gen if qgrad != "fp" else None

    def fw(x, mids):
        return fwd(x, Wproj, mids, Wout, bout, binact=binact, qgrad=qgrad, rhts=rhts, gen=ggen)

    opt = torch.optim.Adam([Wout, bout], lr=3e-3)
    gv = [0.0] * N_BIN
    gvn = 0

    if mode == "fp32_mid":
        mids = [W.clone().requires_grad_(True) for W in Wmids0]
        opt = torch.optim.Adam([Wout, bout] + mids, lr=3e-3)
        for ep in range(EPOCHS):
            for xb, yb in batches(gen):
                opt.zero_grad()
                F.cross_entropy(fw(xb, mids), yb).backward()
                if ep == EPOCHS - 1:
                    for k in range(N_BIN):
                        gv[k] += float(_grad(mids[k]).var())
                    gvn += 1
                opt.step()
        mids = [m.detach() for m in mids]
        return acc(Wproj, mids, Wout, bout, binact), _diag(None, None, None, gv, gvn, Wproj, mids, binact)

    scales = [W.abs().mean() for W in Wmids0]

    if mode == "bitnet":
        lat = [W.clone().requires_grad_(True) for W in Wmids0]
        opt = torch.optim.Adam([Wout, bout] + lat, lr=3e-3)
        for ep in range(EPOCHS):
            for xb, yb in batches(gen):
                opt.zero_grad()
                mids = [
                    (loc + (torch.sign(loc) - loc).detach()) * s for loc, s in zip(lat, scales, strict=False)
                ]
                F.cross_entropy(fw(xb, mids), yb).backward()
                if ep == EPOCHS - 1:
                    for k in range(N_BIN):
                        gv[k] += float(_grad(lat[k]).var())
                    gvn += 1
                opt.step()
        mids = [torch.sign(loc.detach()) * s for loc, s in zip(lat, scales, strict=False)]
        return acc(Wproj, mids, Wout, bout, binact), _diag(None, None, None, gv, gvn, Wproj, mids, binact)

    wb = [torch.sign(W) + (torch.sign(W) == 0).float() for W in Wmids0]
    ebuf = [torch.zeros_like(W) for W in Wmids0]
    ebuf1 = [torch.zeros_like(W) for W in Wmids0]
    xlat = [w.clone() for w in wb]
    m_ema = [torch.zeros_like(W) for W in Wmids0]
    refract = [torch.zeros_like(W) for W in Wmids0]
    mom = [torch.zeros_like(W) for W in Wmids0]
    gamma, tau = kwargs.get("gamma", 1e-2), kwargs.get("tau", 1e-4)
    alpha, delta = kwargs.get("alpha", 0.5), kwargs.get("delta", 0.5)
    mom_bits, beta = kwargs.get("mom_bits", 4), kwargs.get("beta", 0.9)
    flips_first = flips_last = 0
    tracing = TRACE is not None and mode == "edB"
    tidx, trows = torch.zeros(0, dtype=torch.long, device=DEVICE), []
    if tracing:
        tidx, trows = _trace_open(seed, mode, lr, clamp, buf_bits, binact)
        trows[0]["w0"] = _g9(wb[0].flatten()[tidx])
    step = 0
    for ep in range(EPOCHS):
        ep_flips = 0
        for xb, yb in batches(gen):
            opt.zero_grad()
            proxies = [w.clone().requires_grad_(True) for w in wb]
            mids = [p * s for p, s in zip(proxies, scales, strict=False)]
            F.cross_entropy(fw(xb, mids), yb).backward()
            opt.step()
            if mode == "frozen":
                continue
            if ep == EPOCHS - 1:
                for k in range(N_BIN):
                    gv[k] += float(_grad(proxies[k]).var())
                gvn += 1
            for k in range(N_BIN):
                if mode == "bop":
                    wb[k], m_ema[k], nf = bop_step(
                        wb[k],
                        m_ema[k],
                        _grad(proxies[k]),
                        gamma,
                        tau,
                        kwargs.get("bop_bits"),
                        gen,
                        kwargs.get("align", False),
                    )
                elif mode == "edM":
                    wb[k], ebuf[k], mom[k], nf = edm_step(
                        wb[k],
                        ebuf[k],
                        mom[k],
                        _grad(proxies[k]),
                        scales[k],
                        lr,
                        clamp,
                        buf_bits,
                        mom_bits,
                        beta,
                        gen,
                    )
                elif mode == "intlat":
                    xlat[k], nf = intlat_step(xlat[k], _grad(proxies[k]), scales[k], lr, buf_bits, gen)
                    wb[k] = torch.sign(xlat[k]) + (torch.sign(xlat[k]) == 0).float()
                elif mode == "edE":
                    wb[k], mom[k], nf = ede_step(
                        wb[k], mom[k], _grad(proxies[k]), scales[k], lr, clamp, mom_bits, beta, gen
                    )
                elif mode == "cnt":
                    wb[k], ebuf[k], nf = cnt_step(wb[k], ebuf[k], _grad(proxies[k]), buf_bits)
                elif mode == "ed2":
                    wb[k], ebuf1[k], ebuf[k], nf = ed2_step(
                        wb[k], ebuf1[k], ebuf[k], _grad(proxies[k]), scales[k], lr, clamp, buf_bits, gen
                    )
                elif mode == "edS":
                    wb[k], ebuf[k], nf = ed_spatial_step(
                        wb[k], ebuf[k], _grad(proxies[k]), scales[k], lr, clamp, buf_bits, alpha, gen
                    )
                elif mode == "edH":
                    wb[k], ebuf[k], refract[k], nf = ed_hyst_step(
                        wb[k],
                        ebuf[k],
                        refract[k],
                        _grad(proxies[k]),
                        scales[k],
                        lr,
                        clamp,
                        buf_bits,
                        delta,
                        gen,
                    )
                else:
                    tu = -lr * _grad(proxies[k]).flatten()[tidx] / scales[k]
                    wb[k], ebuf[k], nf = ed_step(
                        wb[k], ebuf[k], _grad(proxies[k]), scales[k], lr, clamp, buf_bits, mode, gen
                    )
                    if tracing and k == 0:
                        trows.append(
                            {
                                "step": step,
                                "u": _g9(tu),
                                "e": _g9(ebuf[0].flatten()[tidx]),
                                "w": _g9(wb[0].flatten()[tidx]),
                            }
                        )
                ep_flips += nf
            step += 1
        if ep == 0:
            flips_first = ep_flips
        flips_last = ep_flips
    if tracing:
        _trace_close(trows)
    mids = [w * s for w, s in zip(wb, scales, strict=False)]
    buf = m_ema if mode == "bop" else (xlat if mode == "intlat" else ebuf)
    if mode == "frozen":
        d = _diag(None, None, None, None, 0, Wproj, mids, binact)
    else:
        d = _diag(flips_first, flips_last, buf, gv, gvn, Wproj, mids, binact)
    return acc(Wproj, mids, Wout, bout, binact), d


LOG = Path(os.environ.get("LOG", Path(__file__).resolve().parents[2] / "results" / "mlp.jsonl"))


def report(name, mode, **kw):
    t0 = time.time()
    accs, ds = zip(*[run(s, mode, **kw) for s in SEEDS], strict=False)
    extra = " | " + ", ".join(f"{k}={np.mean([d[k] for d in ds]):.3g}" for k in ds[0]) if ds[0] else ""
    bits, detail = state_bits(mode, kw.get("buf_bits", 8), kw.get("bop_bits"), kw.get("mom_bits", 0))
    print(
        f"{name:26s} acc {np.mean(accs) * 100:5.2f}% ± {np.std(accs) * 100:4.2f}{extra}"
        f" | state {bits}b/wt ({detail}) | {len(SEEDS)}s {time.time() - t0:.0f}s",
        flush=True,
    )
    LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "name": name,
        "mode": mode,
        "width": D_H,
        "batch": BATCH,
        "epochs": EPOCHS,
        "seeds": SEEDS,
        "device": DEVICE,
        "acc_mean": round(float(np.mean(accs)) * 100, 2),
        "acc_std": round(float(np.std(accs)) * 100, 2),
        "state_bits": bits,
        "state_detail": detail,
        "diag": {k: round(float(np.mean([d[k] for d in ds])), 6) for k in (ds[0] or {})},
        "wall_s": round(time.time() - t0),
    } | {k: v for k, v in kw.items() if isinstance(v, (int, float, str, bool))}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    which = sys.argv[1]
    print(f"[bitdiffusion] w={D_H} batch={BATCH} ep={EPOCHS} seeds={SEEDS} dev={DEVICE}", flush=True)
    if which == "base":
        print(
            f"teacher-student 20k/4k | frozen FP proj 64->{D_H}, {N_BIN} binary {D_H}x{D_H} mids, "
            f"FP readout | {EPOCHS} ep",
            flush=True,
        )
        report("frozen random binary", "frozen")
        report("fp32 middles (upper bnd)", "fp32_mid")
        report("bitnet STE (latent ref)", "bitnet")
    elif which == "edA":
        for lr in [float(x) for x in sys.argv[2:]] or [1.0, 3.0]:
            report(f"ed_tierA lr={lr}", "edA", lr=lr)
    elif which == "edB":
        for lr in [float(x) for x in sys.argv[2:]] or [1.0, 3.0]:
            report(f"ed_tierB int8 lr={lr}", "edB", lr=lr)
    elif which == "edBbits":
        for b in [int(x) for x in sys.argv[2:]] or [4, 2]:
            report(f"ed_tierB {b}-bit buf lr=1.0", "edB", lr=1.0, buf_bits=b)
    elif which == "edB_det":
        for b in [int(x) for x in sys.argv[2:]] or [2]:
            report(f"edB DET-round {b}-bit b={BATCH}", "edB_det", lr=1.0, buf_bits=b)
    elif which == "batchsweep":
        b = int(sys.argv[2])
        report(f"edB SR {b}-bit batch={BATCH}", "edB", lr=1.0, buf_bits=b)
    elif which == "bop":
        gamma = float(os.environ.get("GAMMA", "1e-2"))
        bop_bits = int(os.environ["BOPBITS"]) if "BOPBITS" in os.environ else None
        align = os.environ.get("BOPALIGN", "0") == "1"
        binact = "binact" in sys.argv
        for tau in [float(x) for x in sys.argv[2:] if x != "binact"] or [1e-4]:
            name = f"bop g={gamma} tau={tau} emabits={bop_bits} align={align} binact={binact}"
            report(name, "bop", gamma=gamma, tau=tau, bop_bits=bop_bits, binact=binact, align=align)
    elif which == "binact":
        print("(binary activations ON)", flush=True)
        report(f"frozen +binact w={D_H}", "frozen", binact=True)
        report(f"bitnet +binact w={D_H}", "bitnet", binact=True)
        report(f"edB int8 +binact w={D_H}", "edB", lr=1.0, buf_bits=8, binact=True)
    elif which == "qgrad":
        qg, mode = sys.argv[2], sys.argv[3]
        binact = "binact" in sys.argv
        bb = int(sys.argv[sys.argv.index("bits") + 1]) if "bits" in sys.argv else 8
        clamp = float(sys.argv[sys.argv.index("clamp") + 1]) if "clamp" in sys.argv else 1.5
        report(
            f"qg={qg} {mode} b{bb} binact={binact} C={clamp} w={D_H}",
            mode,
            lr=1.0,
            clamp=clamp,
            buf_bits=bb,
            binact=binact,
            qgrad=qg,
        )
    elif which == "one":
        mode = sys.argv[2]
        binact = "binact" in sys.argv
        bb = int(sys.argv[sys.argv.index("bits") + 1]) if "bits" in sys.argv else 8
        lr = float(sys.argv[sys.argv.index("lr") + 1]) if "lr" in sys.argv else 1.0
        alpha = float(sys.argv[sys.argv.index("alpha") + 1]) if "alpha" in sys.argv else 0.5
        delta = float(sys.argv[sys.argv.index("delta") + 1]) if "delta" in sys.argv else 0.5
        clamp = float(sys.argv[sys.argv.index("clamp") + 1]) if "clamp" in sys.argv else 1.5
        mbits = int(sys.argv[sys.argv.index("mbits") + 1]) if "mbits" in sys.argv else 4
        beta = float(sys.argv[sys.argv.index("beta") + 1]) if "beta" in sys.argv else 0.9
        extra = f" a={alpha}" if mode == "edS" else (f" d={delta}" if mode == "edH" else "")
        if mode == "edM":
            extra = f" mb={mbits} beta={beta}"
        report(
            f"{mode} b{bb} binact={binact} lr={lr} C={clamp}{extra} w={D_H}",
            mode,
            lr=lr,
            clamp=clamp,
            buf_bits=bb,
            binact=binact,
            alpha=alpha,
            delta=delta,
            mom_bits=mbits,
            beta=beta,
        )
    elif which == "sr":
        for lr in [float(x) for x in sys.argv[2:]] or [1.0, 3.0]:
            report(f"sr_only lr={lr}", "sr", lr=lr)
    else:
        sys.exit(f"unknown command {which!r}")


if __name__ == "__main__":
    main()
