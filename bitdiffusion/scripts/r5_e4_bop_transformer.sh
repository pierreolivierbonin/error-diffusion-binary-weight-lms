#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export STEPS=2000 EVAL_EVERY=250 EMBD=256 LAYERS=4 HEADS=4 BINACT=1 SEED=0
export LOG=results/r5_e4_bop_transformer.jsonl
[ -e "$LOG" ] && { echo "refusing to touch existing $LOG"; exit 1; }
T="uv run python -m bitdiffusion.train_gpt"
echo "commit $(git rev-parse --short HEAD) branch $(git branch --show-current)"

echo "=== tau sweep, gamma 1e-2, FP32 EMA, 2k steps shakespeare-char d=256 ==="
for TAU in 1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3; do
  TAU=$TAU $T bop
done
echo "=== r5_e4 sweep done; pick tau, then run gamma bracket, 4-bit aligned/absmax, 3 seeds ==="
