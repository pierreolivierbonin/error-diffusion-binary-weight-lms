#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export STEPS=250 EVAL_EVERY=250 EVAL_ITERS=10 SEED=0
export EMBD=384 LAYERS=6 HEADS=6
export LOG=results/parityprobe.jsonl
T="uv run python -m bitdiffusion.train_gpt"

for LR in 50 70 90 130; do
  DROPOUT=0.2 EDLR=$LR $T ed
done
for LR in 50 70 90 130; do
  DROPOUT=0.2 BINACT=1 EDLR=$LR $T ed
done
