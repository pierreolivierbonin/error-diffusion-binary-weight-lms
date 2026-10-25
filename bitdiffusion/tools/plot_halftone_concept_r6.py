import json
import runpy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = runpy.run_path(str(Path(__file__).resolve().with_name("plot_halftone_concept.py")))
BLUE, INK, MUTED, RED = BASE["BLUE"], BASE["INK"], BASE["MUTED"], BASE["RED"]
floyd_steinberg, make_grayscale = BASE["floyd_steinberg"], BASE["make_grayscale"]

TRACE = Path(sys.argv[1])
STATS = Path(sys.argv[2])
OUT = Path(sys.argv[3])


def main():
    rows = [json.loads(line) for line in TRACE.open(encoding="utf-8")]
    head, rows = rows[0], rows[1:]
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    j = stats["shown_weight"]
    clamp = head["clamp"]
    u = np.array([r["u"][j] for r in rows])
    es = np.array([r["e"][j] for r in rows])
    ws = np.array([head["w0"][j]] + [r["w"][j] for r in rows])
    steps = len(u)
    flips = [t for t in range(steps) if ws[t + 1] != ws[t]]
    qstep = 2 * clamp / (2 ** head["buf_bits"] - 1)
    levels = sorted({float(np.clip(k * qstep, -clamp, clamp)) for k in range(-4, 5)})
    e_before = np.concatenate([[0.0], es[:-1]])

    gray = make_grayscale()
    half = floyd_steinberg(gray)
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
    for gl in levels:
        ax_c.axhline(gl, lw=0.7, color=MUTED, alpha=0.5, ls=":" if abs(gl) < clamp else "--")
    ax_c.step(range(steps), es, where="post", lw=1.0, color=BLUE)
    ax_c.plot(flips, [e_before[t] for t in flips], "v", ms=6, color=RED, zorder=5)
    ax_c.text(
        40,
        1.08,
        f"rail = flip threshold (C = {clamp:g})",
        fontsize=6.8,
        color=INK,
        ha="left",
        va="bottom",
    )
    ax_c.set_ylabel("buffer $e_t$ (flip units)", fontsize=7.5, color=INK)
    ax_c.set_ylim(-1.5, 1.62)
    ax_c.set_title(
        f"(c) traced ED buffer (MLP, {head['buf_bits']}-bit, C = {clamp:g}): residual → next step; "
        "flips fire at the threshold",
        fontsize=8,
        color=INK,
        loc="left",
    )
    ax_c.tick_params(labelsize=7, colors=MUTED)

    ax_d = fig.add_subplot(gs[1, 1], sharex=ax_c)
    demanded = np.cumsum(u)
    realized = np.cumsum([2.0 * ws[t + 1] if ws[t + 1] != ws[t] else 0.0 for t in range(steps)])
    ax_d.plot(range(steps), demanded, lw=1.2, color=BLUE)
    ax_d.step(range(steps), realized, where="post", lw=1.4, color=INK)
    ax_d.fill_between(
        range(steps), realized - clamp, realized + clamp, step="post", color=MUTED, alpha=0.15, lw=0
    )
    for t in flips:
        ax_d.axvline(t, lw=0.5, color=RED, alpha=0.3)
    ax_d.set_ylabel("cumulative movement\n(flip units)", fontsize=7.5, color=INK)
    ax_d.set_title(
        "(d) flips follow demanded movement; SR noise lets the gap exceed $\\pm C$",
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

    if OUT.exists():
        sys.exit(f"refusing to overwrite {OUT}")
    fig.savefig(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
