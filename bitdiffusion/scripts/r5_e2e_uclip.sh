#!/bin/sh
cd "$(dirname "$0")/.."
export DATA=enwik8 STEPS=15000 EMBD=512 LAYERS=6 HEADS=8 BINACT=1 EDSCHED=linear EDLR=180 BITS=8
export DIAG=1 EVAL_EVERY=100 STOPAT=3000 UCLIP=0.25
export LOG=results/r5_e2e_uclip.jsonl
[ -e "$LOG" ] && { echo "refusing to touch existing $LOG"; exit 1; }
T="uv run python -m bitdiffusion.train_gpt"
echo "commit $(git rev-parse --short HEAD) branch $(git branch --show-current)"

echo "=== zero-init, seeds 0-9, per-step update clipped at 0.25 flip units (normal max ~0.19) ==="
for S in 0 1 2 3 4 5 6 7 8 9; do
  SEED=$S $T ed || echo "seed $S exit $?"
done
echo "=== r5_e2e done ==="
