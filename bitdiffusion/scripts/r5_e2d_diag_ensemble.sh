#!/bin/sh
cd "$(dirname "$0")/.."
export DATA=enwik8 STEPS=15000 EMBD=512 LAYERS=6 HEADS=8 BINACT=1 EDSCHED=linear EDLR=180 BITS=8
export DIAG=1 EVAL_EVERY=100 STOPAT=3000
export LOG=results/r5_e2d_diag_ensemble.jsonl
[ -e "$LOG" ] && { echo "refusing to touch existing $LOG"; exit 1; }
T="uv run python -m bitdiffusion.train_gpt"
echo "commit $(git rev-parse --short HEAD) branch $(git branch --show-current)"

echo "=== EDINIT=0.5, seeds 0-9, full 3000 steps at 100-step diagnostics ==="
for S in 0 1 2 3 4 5 6 7 8 9; do
  SEED=$S EDINIT=0.5 $T ed || echo "seed $S exit $?"
done
echo "=== zero-init, seeds 0-9, full 3000 steps at 100-step diagnostics ==="
for S in 0 1 2 3 4 5 6 7 8 9; do
  SEED=$S $T ed || echo "seed $S exit $?"
done
echo "=== r5_e2d done ==="
