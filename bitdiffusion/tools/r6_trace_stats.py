import json
import sys
from pathlib import Path

import numpy as np

TRACE = Path(sys.argv[1])
OUT = Path(sys.argv[2])


def load(path):
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    head = rows[0]
    u = np.array([r["u"] for r in rows[1:]])
    e = np.array([r["e"] for r in rows[1:]])
    w = np.array([r["w"] for r in rows[1:]])
    return head, u, e, w


def replay(head, u, e, w):
    clamp = head["clamp"]
    qstep = 2 * clamp / (2 ** head["buf_bits"] - 1)
    w_prev = np.array(head["w0"])
    e_prev = np.zeros(u.shape[1])
    bad_flip = bad_e = 0
    flips = np.zeros_like(u, dtype=bool)
    clip_loss = np.zeros_like(u)
    sr_noise = np.zeros_like(u)
    for t in range(u.shape[0]):
        tt = e_prev + u[t]
        flip = (np.sign(tt) != np.sign(w_prev)) & (np.abs(tt) >= 1.0)
        bad_flip += int((np.where(flip, -w_prev, w_prev) != w[t]).sum())
        r = tt - np.where(flip, 2.0 * np.sign(tt), 0.0)
        c = np.clip(r, -clamp, clamp)
        lo = np.clip(np.floor(c / qstep + 1e-6) * qstep, -clamp, clamp)
        hi = np.clip((np.floor(c / qstep + 1e-6) + 1) * qstep, -clamp, clamp)
        ok = (np.abs(e[t] - lo) < 1e-5) | (np.abs(e[t] - hi) < 1e-5) | (np.abs(e[t] - c) < 1e-5)
        bad_e += int((~ok).sum())
        flips[t] = w[t] != w_prev
        clip_loss[t] = r - c
        sr_noise[t] = e[t] - c
        w_prev, e_prev = w[t], e[t]
    return flips, clip_loss, sr_noise, bad_flip, bad_e


def main():
    head, u, e, w = load(TRACE)
    flips, clip_loss, sr_noise, bad_flip, bad_e = replay(head, u, e, w)
    clamp = head["clamp"]
    demanded = np.cumsum(u, axis=0)
    realized = np.cumsum(np.where(flips, 2.0 * w, 0.0), axis=0)
    gap = demanded - realized
    n_flips = flips.sum(axis=0)
    eligible = np.flatnonzero(n_flips >= 3)
    shown = int(eligible[0]) if len(eligible) else int(np.argmax(n_flips))
    outside = np.abs(gap) > clamp
    stats = {
        "trace": TRACE.name,
        "steps": int(u.shape[0]),
        "weights": int(u.shape[1]),
        "replay_flip_mismatches": bad_flip,
        "replay_buffer_mismatches": bad_e,
        "buffer_values": sorted({float(v) for v in np.unique(e)}),
        "flips_per_weight": [int(n) for n in n_flips],
        "weights_with_3plus_flips": int(len(eligible)),
        "shown_weight": shown,
        "shown_index": head["idx"][shown],
        "shown_flips": int(n_flips[shown]),
        "shown_flip_steps": [int(t) for t in np.flatnonzero(flips[:, shown])],
        "share_steps_outside_band_all": round(float(outside.mean()), 4),
        "share_steps_outside_band_shown": round(float(outside[:, shown].mean()), 4),
        "max_abs_gap_all": round(float(np.abs(gap).max()), 3),
        "max_abs_gap_shown": round(float(np.abs(gap[:, shown]).max()), 3),
        "median_final_abs_gap_all": round(float(np.median(np.abs(gap[-1]))), 3),
        "clip_loss_total_abs_mean": round(float(np.abs(clip_loss.sum(axis=0)).mean()), 3),
        "clip_loss_total_shown": round(float(clip_loss[:, shown].sum()), 3),
        "sr_noise_total_shown": round(float(sr_noise[:, shown].sum()), 3),
        "abs_u_median": float(f"{np.median(np.abs(u)):.3g}"),
        "abs_u_max": float(f"{np.abs(u).max():.3g}"),
        "identity_check_shown": round(
            float(gap[-1, shown] - e[-1, shown] - clip_loss[:, shown].sum() + sr_noise[:, shown].sum()), 6
        ),
    }
    with OUT.open("x", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
