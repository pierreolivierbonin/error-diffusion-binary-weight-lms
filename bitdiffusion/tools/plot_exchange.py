from pathlib import Path

import matplotlib.pyplot as plt

WIDTHS = [64, 96, 128, 192, 256, 384]
SERIES = {
    "BitNet latent (97 b/wt, binact)": (97, [40.94, 47.33, 52.32, 57.53, 59.53, 62.35], "#d62728", "s"),
    "FP32+Adam (96 b/wt, FP acts)": (96, [42.80, 46.88, 50.25, 55.83, 58.30, 61.18], "#888888", "o"),
    "ED int8 (9 b/wt, binact)": (9, [40.02, 46.78, 51.45, 56.42, 59.33, 62.53], "#1f77b4", "D"),
    "ED int8 + 1-bit grads (9 b/wt)": (9, [39.95, 46.15, 49.34, 55.75, 58.47, 61.15], "#2ca02c", "^"),
    "dithered Bop int8 (9 b/wt, binact)": (9, [41.32, 47.75, 51.66, 57.18, 59.56, 61.95], "#ff7f0e", "P"),
    "ED int4 (5 b/wt, binact, C=1.0)": (5, [40.30, 46.82, 50.35, 56.35, 59.76, 62.52], "#17becf", "v"),
}


def main():
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150)
    for label, (bits, accs, color, marker) in SERIES.items():
        total = [2 * w * w * bits / 1e6 for w in WIDTHS]
        ax.plot(total, accs, marker=marker, ms=5, lw=1.6, color=color, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Total hidden training state (Mbit, log scale)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(
        "Training-state exchange at equal accuracy: latent-free vs FP baselines\n"
        "(teacher-student, 2 binary layers, 5 seeds)"
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    ax.annotate(
        "",
        xy=(28.6, 62.35),
        xytext=(2.65, 62.53),
        arrowprops={"arrowstyle": "<->", "lw": 0.9, "ls": "--", "color": "#666666"},
    )
    ax.text(
        8.7, 62.0, "same accuracy, ~11× less state (int8)", ha="center", va="top", fontsize=8, color="#444444"
    )
    ax.annotate(
        "",
        xy=(28.6, 62.9),
        xytext=(1.47, 62.9),
        arrowprops={"arrowstyle": "<->", "lw": 0.9, "ls": "--", "color": "#666666"},
    )
    ax.text(6.5, 63.05, "~19× (int4)", ha="center", va="bottom", fontsize=8, color="#444444")
    ax.set_ylim(top=64.2)
    out = Path(__file__).resolve().parent.parent / "results" / "exchange_curve.png"
    fig.tight_layout()
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
