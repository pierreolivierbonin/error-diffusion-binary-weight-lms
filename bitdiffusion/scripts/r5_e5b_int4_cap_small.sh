#!/bin/sh
cd "$(dirname "$0")/.."
export DATA=wikitext103 BLOCK=128 EVAL_ITERS=25 STEPS=15000 EVAL_EVERY=1000 PATIENCE=5 SEED=0
export BINACT=1 BITS=4 EDSCHED=linear UCLIP=0.25 STALLCHECK=1 DIVERGE=0.3
export LOG=results/r5_e5b_int4_cap_small.jsonl
[ -e "$LOG" ] && { echo "refusing to touch existing $LOG"; exit 1; }
T="uv run python -m bitdiffusion.train_gpt"
echo "commit $(git rev-parse --short HEAD) branch $(git branch --show-current)"

echo "=== int4 wikitext-103, cap 0.25, the two smaller sizes (uncapped launches diverged repeatedly) ==="
EMBD=128 LAYERS=2 HEADS=4 EDLR=20 $T ed || echo "128 exit $?"
EMBD=256 LAYERS=4 HEADS=4 EDLR=45 $T ed || echo "256 exit $?"
echo "=== r5_e5b done ==="
