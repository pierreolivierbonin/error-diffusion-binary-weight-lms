import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROWS = [
    ("MLP headline / exchange", ["headline_5seed", "headline", "widthsweep_5seed"]),
    ("Grid rule (Table 1)", ["p2clamp_5seed", "p2boundary_5seed", "p2clamp", "p2boundary"]),
    ("Bop alignment", ["s2bopalign"]),
    ("Integer latent", ["s2intlat"]),
    ("ECO-style / counter", ["s2baselines"]),
    ("Second-order buffer", ["s2sigma"]),
    ("GPT 2k cells (Table 2)", ["p3gpt"]),
    ("GPT converged 50k", ["s2converged"]),
    ("Byte ladder + controls", ["s2controls"]),
    ("Ladder seeds", ["s2ladderseeds"]),
    ("Tokenized ladder", ["s2tokladder"]),
    ("Deploy-then-adapt", ["s2adapt"]),
]

B = chr(92)


def fnum(x):
    return str(int(x)) if x == int(x) else repr(x)


def fmt_nums(vals, range_at=5):
    s = sorted(vals)
    if len(s) >= range_at:
        return f"{fnum(s[0])}--{fnum(s[-1])}"
    return "/".join(fnum(v) for v in s)


def fmt_strs(vals):
    s = sorted(vals)
    return "/".join(v.replace("_", B + "_") for v in s[:4]) + ("/\\ldots" if len(s) > 4 else "")


class Row:
    def __init__(self):
        self.widths, self.pairs, self.epochs, self.steps = set(), set(), set(), set()
        self.seed_ids, self.lrs, self.taus, self.bits, self.datas = set(), set(), set(), set(), set()
        self.seed_count = 0

    def add_meta(self, r):
        f = r.get("fields", {})
        if f.get("w") is not None:
            self.widths.add(int(f["w"]))
        if f.get("ep") is not None:
            self.epochs.add(float(f["ep"]))
        if f.get("embd") is not None:
            if f.get("L") is not None:
                self.pairs.add((int(f["L"]), int(f["embd"])))
            self.datas.add(f.get("data", "shakespeare"))
        if f.get("steps") is not None:
            self.steps.add(float(f["steps"]))
        if f.get("seed") is not None:
            self.seed_ids.add(int(f["seed"]))
        if f.get("edlr") is not None:
            self.lrs.add(float(f["edlr"]))
        if f.get("bits") is not None:
            self.bits.add(int(f["bits"]))
        m = re.search(r"seeds=\[([0-9,\s]+)\]", r.get("raw", ""))
        if m:
            self.seed_count = max(self.seed_count, len(m.group(1).split(",")))

    def add_result(self, r):
        name = r.get("name", "")
        if m := re.search(r"lr=([0-9.e+-]+)", name):
            self.lrs.add(float(m.group(1)))
        if m := re.search(r"w=(\d+)", name):
            self.widths.add(int(m.group(1)))
        for m in re.finditer(r"\bb(\d+)\b|(\d+)-bit|\bint(8)\b", name):
            self.bits.add(int(next(g for g in m.groups() if g)))

    def add_flat(self, r):
        if r.get("width") is not None:
            self.widths.add(int(r["width"]))
        if r.get("embd") is not None:
            if r.get("layers") is not None:
                self.pairs.add((int(r["layers"]), int(r["embd"])))
            self.datas.add(r.get("data") or "shakespeare")
        if r.get("epochs") is not None:
            self.epochs.add(float(r["epochs"]))
        if r.get("steps") is not None:
            self.steps.add(float(r["steps"]))
        if isinstance(r.get("seeds"), list):
            self.seed_count = max(self.seed_count, len(r["seeds"]))
        if r.get("seed") is not None:
            self.seed_ids.add(int(r["seed"]))
        if r.get("lr") is not None:
            self.lrs.add(float(r["lr"]))
        if r.get("edlr") is not None:
            self.lrs.add(float(r["edlr"]))
        if r.get("tau") is not None:
            self.taus.add(float(r["tau"]))
        for k in ("bits", "buf_bits"):
            if r.get(k) is not None:
                self.bits.add(int(r[k]))
        if r.get("mode") == "bop":
            self.bits.add(int(r.get("bop_bits") or 32))

    def arch(self):
        if self.pairs:
            layers = {L for L, _ in self.pairs}
            if len(layers) == 1:
                embds = sorted(e for _, e in self.pairs)
                L = layers.pop()
                if len(embds) >= 3:
                    return f"{L}L/{embds[0]}--{embds[-1]}"
                return "/".join(f"{L}L/{e}" for e in embds)
            return "/".join(f"{L}L/{e}" for L, e in sorted(self.pairs))
        ws = sorted(self.widths)
        if len(ws) >= 3:
            return f"MLP w{ws[0]}--{ws[-1]}"
        return "MLP w" + "/".join(str(w) for w in ws)

    def cells(self):
        fliplr = (
            fmt_nums(self.lrs)
            if self.lrs
            else ("\\ensuremath{\\tau} " + fmt_nums(self.taus) if self.taus else "")
        )
        return [
            fmt_strs(self.datas) if self.datas else "teacher",
            self.arch(),
            fmt_nums(self.steps if self.steps else self.epochs),
            fmt_nums(self.seed_ids) if self.seed_ids else str(self.seed_count),
            fliplr,
            fmt_nums(self.bits, range_at=99),
        ]


def main():
    lines = [
        "\\begin{tabular}{lllllll}",
        "\\toprule",
        "Experiment & data & arch & steps/ep & seeds & flip-lr & bits \\\\",
        "\\midrule",
    ]
    for label, files in ROWS:
        row = Row()
        for fn in files:
            p = ROOT / "results" / f"{fn}.jsonl"
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = r.get("type")
                if t in ("eval", "eval0", "text"):
                    continue
                if t == "meta":
                    row.add_meta(r)
                elif t == "result":
                    row.add_result(r)
                else:
                    row.add_flat(r)
        lines.append(" & ".join([label] + row.cells()) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out = ROOT / "paper" / "config_table.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{out}: {len(lines) - 6} rows")


if __name__ == "__main__":
    main()
