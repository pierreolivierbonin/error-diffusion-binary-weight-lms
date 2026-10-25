#!/bin/sh
cd "$(dirname "$0")/.."
export DATA=enwik8 BLOCK=256 EVAL_ITERS=50 STEPS=15000 EVAL_EVERY=1000 PATIENCE=5 SEED=0
export BINACT=1 BITS=4 EDSCHED=linear UCLIP=0.25 STALLCHECK=1 DIVERGE=0.3
export LOG=results/r5_e5d_int4_cap_enwik8.jsonl
[ -e "$LOG" ] && { echo "refusing to touch existing $LOG"; exit 1; }
T="uv run python -m bitdiffusion.train_gpt"
echo "commit $(git rev-parse --short HEAD) branch $(git branch --show-current)"

echo "=== int4 enwik8 ladder with the cap, S2.18 probe-set lrs 12 / 30 / 45 ==="
EMBD=128 LAYERS=2 HEADS=4 EDLR=12 $T ed || echo "128 exit $?"
EMBD=256 LAYERS=4 HEADS=4 EDLR=30 $T ed || echo "256 exit $?"
EMBD=512 LAYERS=6 HEADS=8 EDLR=45 $T ed || echo "512 exit $?"
echo "=== r5_e5d done ==="
