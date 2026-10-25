#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export DATA=enwik8 STEPS=15000 EVAL_EVERY=1000 PATIENCE=5 SEED=0
export EMBD=256 LAYERS=4 HEADS=4 BINACT=1
export LOG=results/r5_e4c_bop_enwik8.jsonl
[ -e "$LOG" ] && { echo "refusing to touch existing $LOG"; exit 1; }
T="uv run python -m bitdiffusion.train_gpt"
echo "commit $(git rev-parse --short HEAD) branch $(git branch --show-current)"

echo "=== Bop FP32 EMA on the enwik8 4L/256 ladder cell (ED 1.505 / 1.516 +- 0.012; BitNet+binact 1.469): gamma x tau grid ==="
for G in 1e-3 1e-4; do
  for TAU in 1e-7 3e-7 1e-6; do
    GAMMA=$G TAU=$TAU $T bop
  done
done
echo "=== r5_e4c grid done; pick (gamma, tau), then 4-bit absmax vs tau-aligned ==="
