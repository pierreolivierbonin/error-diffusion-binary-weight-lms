import runpy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = runpy.run_path(str(Path(__file__).resolve().with_name("plot_halftone_concept.py")))
BLUE, INK, MUTED, RED = BASE["BLUE"], BASE["INK"], BASE["MUTED"], BASE["RED"]
floyd_steinberg, make_grayscale, simulate_ed = (
    BASE["floyd_steinberg"],
    BASE["make_grayscale"],
    BASE["simulate_ed"],
)


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
    for t in flips:
        ax_c.plot(t, es[t - 1], "^" if es[t] > es[t - 1] else "v", ms=6, color=RED, zorder=5)
    ax_c.annotate(
        "SR rounds 0.98 down to 1/3 (p = 3%)",
        xy=(34, es[34]),
        xytext=(44, 0.62),
        fontsize=6.5,
        color=INK,
        arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED),
    )
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
        "(c) simulated ED buffer, one weight: residual → next step; flips fire at the threshold",
        fontsize=8,
        color=INK,
        loc="left",
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
        "(d) flips follow demanded movement with a lag; clamp and SR noise let the gap exceed $\\pm C$",
        fontsize=8,
        color=INK,
        loc="left",
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

    out = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parent.parent / "results" / "halftone_concept_sim.png"
    )
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
