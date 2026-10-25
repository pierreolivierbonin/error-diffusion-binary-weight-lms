import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
WIDTHS = [64, 96, 128, 192, 256, 384]


def int8_c1():
    accs = {}
    for line in (ROOT / "results" / "r4_widthsweep_int8c1.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("name", "").startswith("edB b8 binact=True lr=1.0 C=1.0 w="):
            accs[int(r["name"].split("w=")[1])] = r["acc_mean"]
    return [accs[w] for w in WIDTHS]


SERIES = [
    ("latent reference (97 b/wt)", 97, [40.94, 47.33, 52.32, 57.53, 59.53, 62.35], "#2a78d6", "s"),
    ("FP32+Adam (96 b/wt, FP activations)", 96, [42.80, 46.88, 50.25, 55.83, 58.30, 61.18], "#eb6834", "o"),
    ("ED int8 (9 b/wt)", 9, int8_c1(), "#1baf7a", "D"),
    ("dithered Bop int8 (9 b/wt)", 9, [41.32, 47.75, 51.66, 57.18, 59.56, 61.95], "#eda100", "^"),
    ("ED int4 (5 b/wt)", 5, [40.30, 46.82, 50.35, 56.35, 59.76, 62.52], "#e87ba4", "v"),
]


def main():
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
    fig, ax = plt.subplots(figsize=(6.4, 2.35), dpi=300)
    for label, bits, accs, color, marker in SERIES:
        total = [2 * w * w * bits / 1e6 for w in WIDTHS]
        ax.plot(total, accs, marker=marker, ms=3.5, lw=1.2, color=color, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("total training state of the two binary layers (Mbit, log scale)")
    ax.set_ylabel("test accuracy (%)")
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(fontsize=6.2, loc="lower right", frameon=False, handlelength=1.6)
    arrow = {"arrowstyle": "<->", "lw": 0.8, "ls": "--", "color": "#555555"}
    ax.annotate("", xy=(28.6, 62.35), xytext=(2.65, 62.35), arrowprops=arrow)
    note = "same accuracy, ~11x less state (8-bit buffer)"
    ax.text(8.7, 62.6, note, ha="center", va="bottom", fontsize=6.2, color="#444444")
    ax.annotate("", xy=(28.6, 64.0), xytext=(1.47, 64.0), arrowprops=arrow)
    ax.text(
        6.5, 64.3, "~19x less state (4-bit buffer)", ha="center", va="bottom", fontsize=6.2, color="#444444"
    )
    ax.set_ylim(38, 66.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    out = ROOT / "paper_slmw2026" / "exchange_curve_20260828.png"
    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
