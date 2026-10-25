from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INK = "#333333"
MUTED = "#8a8a8a"
BLUE = "#1f77b4"
RED = "#d62728"


def make_grayscale(n=72):
    y, x = np.mgrid[0:n, 0:n] / n
    img = 0.15 + 0.7 * x
    img += 0.35 * np.exp(-(((x - 0.35) ** 2 + (y - 0.35) ** 2) / 0.03))
    img -= 0.25 * np.exp(-(((x - 0.7) ** 2 + (y - 0.7) ** 2) / 0.05))
    return np.clip(img, 0, 1)


def floyd_steinberg(img):
    a = img.copy()
    h, w = a.shape
    out = np.zeros_like(a)
    for i in range(h):
        for j in range(w):
            old = a[i, j]
            new = 1.0 if old >= 0.5 else 0.0
            out[i, j] = new
            err = old - new
            if j + 1 < w:
                a[i, j + 1] += err * 7 / 16
            if i + 1 < h:
                if j > 0:
                    a[i + 1, j - 1] += err * 3 / 16
                a[i + 1, j] += err * 5 / 16
                if j + 1 < w:
                    a[i + 1, j + 1] += err * 1 / 16
    return out


def simulate_ed(steps=240, seed=0):
    rng = np.random.default_rng(seed)
    grid = np.array([-1.0, -1 / 3, 1 / 3, 1.0])
    e, w = 0.0, 1.0
    drive = 0.05 * np.sin(2 * np.pi * (np.arange(steps) / 150.0 + 0.35)) + rng.normal(0, 0.012, steps)
    es, ws, flips = [], [], []
    for t in range(steps):
        u = drive[t]
        tt = e + u
        if np.sign(tt) != np.sign(w) and abs(tt) >= 1.0:
            w = -w
            flips.append(t)
            tt -= 2 * np.sign(tt)
        tt = np.clip(tt, -1.0, 1.0)
        lo = grid[grid <= tt + 1e-12].max()
        hi = grid[grid >= tt - 1e-12].min()
        if hi > lo:
            p = (tt - lo) / (hi - lo)
            e = hi if rng.random() < p else lo
        else:
            e = lo
        es.append(e)
        ws.append(w)
    return np.array(drive), np.array(es), np.array(ws), flips, grid


def main():
    gray = make_grayscale()
    half = floyd_steinberg(gray)
    drive, es, ws, flips, grid = simulate_ed()
    steps = len(es)

    fig = plt.figure(figsize=(10.6, 4.6), dpi=200, layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.9], wspace=0.06, hspace=0.14)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(gray, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax_a.set_title("(a) grayscale", fontsize=8, color=INK, loc="left")
    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.imshow(half, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax_b.set_title("(b) halftone: residual → later pixels", fontsize=8, color=INK, loc="left")
    for ax in (ax_a, ax_b):
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(MUTED)

    ax_c = fig.add_subplot(gs[0, 1])
    for gl in grid:
        ax_c.axhline(gl, lw=0.7, color=MUTED, alpha=0.5, ls=":" if abs(gl) < 1 else "--")
    ax_c.step(range(steps), es, where="post", lw=1.2, color=BLUE)
    ax_c.plot(flips, [es[t] for t in flips], "v", ms=6, color=RED, zorder=5)
    ax_c.text(
        6,
        1.08,
        "top grid level = flip threshold (C = 1)",
        fontsize=6.8,
        color=INK,
        ha="left",
        va="bottom",
    )
    ax_c.set_ylabel("buffer $e_t$ (flip units)", fontsize=7.5, color=INK)
    ax_c.set_ylim(-1.5, 1.62)
    ax_c.set_title(
        "(c) ED buffer: residual → next step; flips fire at the threshold", fontsize=8, color=INK, loc="left"
    )
    ax_c.tick_params(labelsize=7, colors=MUTED)

    ax_d = fig.add_subplot(gs[1, 1], sharex=ax_c)
    demanded = np.cumsum(drive)
    realized = np.zeros(steps)
    move = 0.0
    fl = set(flips)
    w = 1.0
    for t in range(steps):
        if t in fl:
            w = -w
            move += 2 * np.sign(w)
        realized[t] = move
    ax_d.plot(range(steps), demanded, lw=1.4, color=BLUE)
    ax_d.step(range(steps), realized, where="post", lw=1.4, color=INK)
    ax_d.fill_between(range(steps), realized - 1.0, realized + 1.0, color=MUTED, alpha=0.15, lw=0)
    for t in flips:
        ax_d.axvline(t, lw=0.5, color=RED, alpha=0.3)
    ax_d.set_ylabel("cumulative movement\n(flip units)", fontsize=7.5, color=INK)
    ax_d.set_title(
        "(d) EF invariant: flips track demanded movement within $\\pm C$", fontsize=8, color=INK, loc="left"
    )
    ax_d.tick_params(labelsize=7, colors=MUTED)
    ax_d.set_xlabel(
        "training step\nblue: cumulative demanded $\\sum u_t$    black: realized by flips    "
        "grey band: $\\pm C$    red: flip times",
        fontsize=6.8,
        color=INK,
    )

    for ax in (ax_c, ax_d):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)

    out = Path(__file__).resolve().parent.parent / "results" / "halftone_concept.png"
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
