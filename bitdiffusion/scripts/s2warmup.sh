#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export DATA=enwik8 STEPS=15000 EVAL_EVERY=1000 PATIENCE=5
export LOG=results/s2warmup.jsonl
T="uv run python -m bitdiffusion.train_gpt"

for S in 2 3; do
  SEED=$S EMBD=512 LAYERS=6 HEADS=8 BINACT=1 EDSCHED=linear EDLR=180 EDWARM=1500 $T ed
done
