import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FILES = ["p3gpt", "audit_fix", "s2int4gpt", "r4_gpt_seed2"]
LR = {128: 100.0, 256: 60.0, 512: 80.0}
HEAD = "method & $d{=}128$ & $d{=}256$ & $d{=}512$ & $d{=}256$ \\\\\n\\midrule\n"


def finals():
    for f in FILES:
        for line in (ROOT / "results" / f"{f}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("type") != "final" or "mode" not in r:
                continue
            if r.get("data", "shakespeare") != "shakespeare" or r.get("steps", 2000) != 2000:
                continue
            if r.get("layers", 4) != 4 or r.get("binact", False) or r.get("loadck"):
                continue
            yield r


def cell(mode, embd, bits=None):
    vals = {}
    for r in finals():
        if r["mode"] != mode or r["embd"] != embd:
            continue
        if mode == "ed":
            if r.get("bits") != bits or r.get("edlr") != LR[embd] or r.get("qgrad", "fp") != "fp":
                continue
        vals.setdefault(r["seed"], r["best_val"])
    return np.array([vals[k] for k in sorted(vals)])


def fmt(v, bold):
    t = f"{v.mean():.3f} \\pm {v.std(ddof=0):.3f}"
    return f"$\\mathbf{{{t}}}$" if bold else f"${t}$"


def main():
    rows = [
        ("FP32+Adam (\\bwt{96})", "fp32", None, "1.591"),
        ("latent reference (\\bwt{97})", "bitnet", None, "$\\mathbf{1.556}$"),
        ("\\ed{}-int8 (\\bwt{9})", "ed", 8, "1.586 (1.583)"),
        ("\\ed{}-int4 (\\bwt{5})", "ed", 4, "1.681"),
    ]
    cells = {(m, b): [cell(m, e, b) for e in (128, 256, 512)] for _, m, b, _ in rows}
    mins = [min(cells[k][c].mean() for k in cells) for c in range(3)]
    out = []
    for label, mode, bits, right in rows:
        cs = cells[(mode, bits)]
        print(label, [(len(c), round(c.mean(), 3), round(c.std(ddof=0), 3)) for c in cs], file=sys.stderr)
        mid = " & ".join(fmt(c, abs(c.mean() - mins[k]) < 1e-9) for k, c in enumerate(cs))
        out.append(f"{label} & {mid} & {right} \\\\")
    lat, ed8 = cells[("bitnet", None)], cells[("ed", 8)]
    gaps = [f"${100 * (e.mean() / ref.mean() - 1):+.1f}\\%$" for e, ref in zip(ed8, lat, strict=True)]
    out.append(
        f"\\quad \\ed{{}}-int8 vs.\\ latent & {gaps[0]} & {gaps[1]} & {gaps[2]} & $+1.9\\%$ ($+1.7\\%$) \\\\"
    )
    body = "\n".join(out)
    print(body)
    if "--write" in sys.argv:
        p = ROOT / "paper_slmw2026" / "main_20260828.tex"
        s = p.read_text(encoding="utf-8")
        if "%%TABLE2-ROWS%%" in s:
            s = s.replace("%%TABLE2-ROWS%%", body)
        else:
            i = s.index(HEAD) + len(HEAD)
            j = s.index("\\bottomrule", i)
            s = s[:i] + body + "\n" + s[j:]
        p.write_text(s, encoding="utf-8")
        print("written", file=sys.stderr)


if __name__ == "__main__":
    main()
