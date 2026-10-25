#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export STEPS=2000 EVAL_EVERY=250 EMBD=256 LAYERS=4 HEADS=4 BINACT=1 SEED=0
export LOG=results/r5_e4b_bop_regime.jsonl
[ -e "$LOG" ] && { echo "refusing to touch existing $LOG"; exit 1; }
T="uv run python -m bitdiffusion.train_gpt"
echo "commit $(git rev-parse --short HEAD) branch $(git branch --show-current)"

echo "=== lower tau, gamma 1e-2 ==="
for TAU in 1e-7 3e-7; do TAU=$TAU $T bop; done
echo "=== gamma bracket at tau 1e-6 and 3e-7 ==="
for G in 1e-3 1e-1; do GAMMA=$G TAU=1e-6 $T bop; GAMMA=$G TAU=3e-7 $T bop; done
echo "=== r5_e4b done ==="
