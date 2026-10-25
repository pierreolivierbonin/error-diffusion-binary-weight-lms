#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export DATA=enwik8 STEPS=15000 EVAL_EVERY=1000 PATIENCE=5
export LOG=results/s2ladderseeds.jsonl
T="uv run python -m bitdiffusion.train_gpt"

for S in 1 2; do
  SEED=$S EMBD=128 LAYERS=2 HEADS=4 BINACT=1 $T bitnet
  SEED=$S EMBD=128 LAYERS=2 HEADS=4 BINACT=1 EDSCHED=linear EDLR=25 $T ed
  SEED=$S EMBD=256 LAYERS=4 HEADS=4 BINACT=1 $T bitnet
  SEED=$S EMBD=256 LAYERS=4 HEADS=4 BINACT=1 EDSCHED=linear EDLR=60 $T ed
  SEED=$S EMBD=512 LAYERS=6 HEADS=8 BINACT=1 $T bitnet
  SEED=$S EMBD=512 LAYERS=6 HEADS=8 BINACT=1 EDSCHED=linear EDLR=180 $T ed
done
