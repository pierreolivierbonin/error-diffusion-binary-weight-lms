import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BAND = (1.0e-3, 1.3e-3)
FAMILIES = [
    ("shakespeare-char 2k", "Shakespeare char, 2k steps", "#2a78d6"),
    ("shakespeare-char 50k", "Shakespeare char, 50k steps", "#eb6834"),
    ("enwik8 15k", "enwik8, 15k steps", "#1baf7a"),
    ("wikitext103 BPE 15k", "wikitext-103, 15k steps", "#eda100"),
    ("deploy-then-adapt shakespeare_bytes 5k", "adapt to Shakespeare bytes, 5k steps", "#e87ba4"),
    ("shakespeare_bytes scratch 5k", "Shakespeare bytes from scratch, 5k steps", "#008300"),
]
EXCLUDE = ("1-bit", "adaptive", "parity")


def family(task):
    for key, label, color in FAMILIES:
        if task.startswith(key):
            return label, color
    return None, None


def main():
    rows = list(csv.DictReader((ROOT / "results" / "r4_drive_table.csv").open()))
    fig, ax = plt.subplots(figsize=(6.0, 3.4), dpi=200)
    ax.axhspan(BAND[0] * 1e3, BAND[1] * 1e3, color="#dddddd", alpha=0.6, lw=0, label="probe selection band")
    seen = set()
    for r in rows:
        if any(x in r["task"] for x in EXCLUDE):
            continue
        if "protocol-violation" in r["selected"] or "reused" in r["selected"]:
            continue
        label, color = family(r["task"])
        if label is None:
            continue
        try:
            lr = float(r["flip_lr"])
        except ValueError:
            continue
        drive = float(r["drive"]) * 1e3
        marker = "o" if r["bits"] == "8" else "^"
        selected = r["selected"].startswith("yes") or r["selected"] in ("tie", "bracket")
        lab = label if label not in seen else None
        seen.add(label)
        ax.plot(
            lr,
            drive,
            marker=marker,
            ms=6 if selected else 5,
            color=color,
            mec=color,
            mfc=color if selected else "none",
            lw=0,
            label=lab,
        )
    ax.set_xscale("log")
    ax.set_xlabel("candidate flip learning rate (log scale)")
    ax.set_ylabel("drive at step 250 (flip units/step, x1e-3)")
    ax.set_ylim(0, 2.6)
    ax.grid(True, which="both", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.plot([], [], marker="o", color="#444444", lw=0, label="8-bit buffer (filled = selected)")
    ax.plot([], [], marker="^", color="#444444", lw=0, label="4-bit buffer (filled = selected)")
    ax.legend(fontsize=6.5, loc="upper left", frameon=False, ncol=2)
    out = ROOT / "paper_slmw2026" / "drive_curve_20260828.png"
    fig.tight_layout()
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
