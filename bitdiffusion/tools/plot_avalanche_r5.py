import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, INK, GRID = "#2a78d6", "#eb6834", "#1d1d1b", "#dedbd2"


def load(path, init, seed):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    ev = [r for r in rows if r["type"] == "eval" and r["edinit"] == init and r["seed"] == seed]
    return sorted(ev, key=lambda r: r["step"])


def main(log, out):
    col = load(log, 0.5, 6)
    cln = load(log, 0.5, 4)
    panels = [
        ("val_loss", "validation loss (nats/byte)", lambda r: r["val_loss"]),
        ("flips_per_step", "flips per step (thousands)", lambda r: r["flips_per_step"] / 1e3),
        ("act_scale", "block-0 pre-activation scale", lambda r: r["act_scale"][0]),
        ("grad_bin", "gradient norm, binary proxies", lambda r: r["grad_bin"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.6), dpi=200)
    for ax, (_key, label, f) in zip(axes.flat, panels, strict=True):
        ax.plot([r["step"] for r in cln], [f(r) for r in cln], color=BLUE, lw=1.6, label="seed 4, clean")
        xs, ys = [r["step"] for r in col], [f(r) for r in col]
        ax.plot(xs, ys, color=ORANGE, lw=1.6, label="seed 6, collapsed")
        ev = next(r for r in col if r["step"] == 2000)
        ax.plot([2000], [f(ev)], "o", ms=5, color=ORANGE, mec="white", mew=1.2, zorder=5)
        ax.set_ylabel(label, fontsize=7.5, color=INK)
        ax.set_xlim(0, 3000)
        ax.set_xticks([0, 1000, 2000, 3000])
        ax.set_xticklabels(["0", "1k", "2k", "3k"], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[1, 0].set_xlabel("training step", fontsize=7.5, color=INK)
    axes[1, 1].set_xlabel("training step", fontsize=7.5, color=INK)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
