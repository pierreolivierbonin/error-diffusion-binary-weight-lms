#!/bin/sh
cd "$(dirname "$0")/.."
export DATA=wikitext103 BLOCK=128 EMBD=512 LAYERS=6 HEADS=8 BINACT=1 SEED=0
export STEPS=250 EVAL_EVERY=250 EVAL_ITERS=5 EDSCHED=const
export LOG=results/r5_e3_probe.jsonl
[ -e "$LOG" ] && { echo "refusing to touch existing $LOG"; exit 1; }
T="uv run python -m bitdiffusion.train_gpt"
echo "commit $(git rev-parse --short HEAD) branch $(git branch --show-current)"

echo "=== 250-step drive probes: occupancy at 250 / 250 must land in 1.0-1.3e-3 ==="
for FZ in "" "wte,wpe" "wte,wpe,head"; do
  for LR in 150 225 300; do
    FREEZE=$FZ EDLR=$LR $T ed
  done
done
for LR in 150 225 300; do
  ADAMLR=1e-4 EDLR=$LR $T ed
done
echo "=== r5_e3 probes done ==="
