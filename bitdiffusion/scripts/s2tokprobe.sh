#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export DATA=wikitext103 STEPS=250 EVAL_EVERY=250 EVAL_ITERS=10 SEED=0
export LOG=results/s2tokprobe.jsonl
T="uv run python -m bitdiffusion.train_gpt"

for LR in 15 25 40; do
  EMBD=128 LAYERS=2 HEADS=4 BINACT=1 EDLR=$LR $T ed
done
for LR in 40 60 90; do
  EMBD=256 LAYERS=4 HEADS=4 BINACT=1 EDLR=$LR $T ed
done
for LR in 120 180 270; do
  EMBD=512 LAYERS=6 HEADS=8 BINACT=1 EDLR=$LR $T ed
done
