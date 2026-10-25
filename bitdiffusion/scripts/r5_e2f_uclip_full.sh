#!/bin/sh
cd "$(dirname "$0")/.."
export DATA=enwik8 STEPS=15000 EVAL_EVERY=500 PATIENCE=5 EMBD=512 LAYERS=6 HEADS=8 BINACT=1 EDSCHED=linear EDLR=180 BITS=8
export DIAG=1 UCLIP=0.25
export LOG=results/r5_e2f_uclip_full.jsonl
[ -e "$LOG" ] && { echo "refusing to touch existing $LOG"; exit 1; }
T="uv run python -m bitdiffusion.train_gpt"
echo "commit $(git rev-parse --short HEAD) branch $(git branch --show-current)"

echo "=== zero-init, seeds 0-2, full 15k with the 0.25 flip-unit update cap (S2.9 clean reference: 1.287-1.291) ==="
for S in 0 1 2; do
  SEED=$S $T ed || echo "seed $S exit $?"
done
echo "=== r5_e2f done ==="
