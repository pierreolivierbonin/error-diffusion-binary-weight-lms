#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export STEPS=50000 EVAL_EVERY=1000 PATIENCE=0 EMBD=256 LAYERS=4 HEADS=4 SEED=0
export BINACT=1 EDSCHED=linear SNAP=1
export LOG=results/r5_e1_int4_drift_v2.jsonl
[ -e "$LOG" ] && { echo "refusing to touch existing $LOG"; exit 1; }
T="uv run python -m bitdiffusion.train_gpt"
echo "commit $(git rev-parse --short HEAD) branch $(git branch --show-current)"

echo "=== int8 lr60 (S2.3 converged cell) with buffer histograms ==="
BITS=8 EDLR=60 $T ed
echo "=== int4 lr45 (S2.18 registered comparable cell) with buffer histograms ==="
BITS=4 EDLR=45 $T ed
echo "=== r5_e1 done ==="
