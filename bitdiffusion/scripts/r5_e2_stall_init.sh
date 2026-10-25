#!/bin/sh
cd "$(dirname "$0")/.."
export DATA=enwik8 STEPS=15000 EVAL_EVERY=500 STOPAT=3000 STALLCHECK=1
export EMBD=512 LAYERS=6 HEADS=8 BINACT=1 EDSCHED=linear EDLR=180 BITS=8
export LOG=results/r5_e2_stall_init.jsonl
[ -e "$LOG" ] && { echo "refusing to touch existing $LOG"; exit 1; }
T="uv run python -m bitdiffusion.train_gpt"
echo "commit $(git rev-parse --short HEAD) branch $(git branch --show-current)"

echo "=== A: zero-init baseline, seeds 5-9, stall rate to step 3000 ==="
for S in 5 6 7 8 9; do
  SEED=$S $T ed || echo "seed $S exit $?"
done

echo "=== B: EDINIT=0.5, seeds 0-9, stall rate to step 3000 ==="
for S in 0 1 2 3 4 5 6 7 8 9; do
  SEED=$S EDINIT=0.5 $T ed || echo "seed $S exit $?"
done
echo "=== r5_e2 A+B done ==="
