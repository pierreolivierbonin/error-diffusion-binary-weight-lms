#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export DATA=enwik8 STEPS=15000 EVAL_EVERY=1000 PATIENCE=5
export EMBD=256 LAYERS=4 HEADS=4 BINACT=1
export LOG=results/r5_e4d_bop_bits.jsonl
[ -e "$LOG" ] && { echo "refusing to touch existing $LOG"; exit 1; }
T="uv run python -m bitdiffusion.train_gpt"
echo "commit $(git rev-parse --short HEAD) branch $(git branch --show-current)"
export GAMMA=1e-4 TAU=1e-7

echo "=== frozen-core control: Bop with tau=1 never flips; the FP remnant trains alone ==="
SEED=0 TAU=1 $T bop
echo "=== bracket below the grid edge: gamma 1e-4, tau 3e-8 ==="
SEED=0 TAU=3e-8 $T bop
echo "=== FP32-EMA Bop at the grid best setting (gamma 1e-4, tau 1e-7), seeds 1 and 2 ==="
for S in 1 2; do SEED=$S $T bop; done
echo "=== 4-bit EMA, absmax grid vs tau-aligned grid, 3 seeds (Table 1 on the transformer) ==="
for S in 0 1 2; do
  SEED=$S BOPBITS=4 $T bop
  SEED=$S BOPBITS=4 BOPALIGN=1 $T bop
done
echo "=== r5_e4d done ==="
