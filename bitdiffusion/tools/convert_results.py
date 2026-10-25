import json
import re
import sys
from pathlib import Path

ACC = re.compile(r"^(.*?)\s+acc\s+([\d.]+)%\s*.\s*([\d.]+)(.*)$")
FINAL = re.compile(r"^FINAL\s+(.*)$")
STEP = re.compile(
    r"^step\s+(\d+)\s*\|\s*train\s+([\d.]+)\s*\|\s*val\s+([\d.]+)\s*\|\s*"
    r"flips/step\s+([\d.eE+]+)\s*\|\s*buf\s+([\d.eE+-]+)\s*\|\s*(\d+)s$"
)
KV = re.compile(r"(\w+)=([^\s|,]+)")


def coerce(v):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return {"True": True, "False": False, "None": None}.get(v, v)


def parse_line(line, section):
    rec = {"section": section, "raw": line}
    m = ACC.match(line)
    if m:
        rec["type"] = "result"
        rec["name"] = m.group(1).strip()
        rec["acc_mean"] = float(m.group(2))
        rec["acc_std"] = float(m.group(3))
        rec["fields"] = {k: coerce(v) for k, v in KV.findall(m.group(4))}
        sb = re.search(r"state\s+(\d+)b/wt", m.group(4))
        if sb:
            rec["state_bits"] = int(sb.group(1))
        return rec
    m = FINAL.match(line)
    if m:
        rec["type"] = "final"
        rec.update({k: coerce(v) for k, v in KV.findall(m.group(1))})
        return rec
    m = STEP.match(line)
    if m:
        rec["type"] = "eval"
        rec["step"] = int(m.group(1))
        rec["train_loss"] = float(m.group(2))
        rec["val_loss"] = float(m.group(3))
        rec["flips_per_step"] = float(m.group(4))
        rec["buf_absmean"] = float(m.group(5))
        rec["elapsed_s"] = int(m.group(6))
        return rec
    if line.startswith("[") or line.startswith("EARLY-STOP") or line.startswith("("):
        rec["type"] = "meta"
        rec["fields"] = {k: coerce(v) for k, v in KV.findall(line)}
        return rec
    rec["type"] = "text"
    return rec


def convert(path):
    section = None
    out = path.with_suffix(".jsonl")
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("==="):
                section = line.strip("= ").strip()
                continue
            f.write(json.dumps(parse_line(line, section)) + "\n")
            n += 1
    return out, n


def main():
    targets = [Path(a) for a in sys.argv[1:]] or sorted(Path("results").glob("*.txt"))
    for p in targets:
        out, n = convert(p)
        print(f"{p} -> {out} ({n} records)")


if __name__ == "__main__":
    main()
