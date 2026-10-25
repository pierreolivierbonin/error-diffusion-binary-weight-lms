#!/bin/sh
set -e
cd "$(dirname "$0")/.."
R="uv run python -m bitdiffusion.run"
T="uv run python -m bitdiffusion.train_gpt"

DH=192 SEEDS=0,1,2,3,4 LOG=results/s2intlat.jsonl sh -c '
  for LR in 0.3 1.0 3.0; do
    uv run python -m bitdiffusion.run one intlat bits 3 binact lr $LR
  done
'

sh scripts/s2adapt1.sh

export DATA=enwik8 STEPS=15000 EVAL_EVERY=1000 PATIENCE=5
export LOG=results/s2ladderseeds.jsonl
for S in 3 4; do
  SEED=$S EMBD=512 LAYERS=6 HEADS=8 BINACT=1 EDSCHED=linear EDLR=180 $T ed
done
