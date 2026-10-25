#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export EMBD=256 LAYERS=4 HEADS=4 BINACT=1 SEED=0
export LOG=results/s2adapt.jsonl
T="uv run python -m bitdiffusion.train_gpt"

DATA=enwik8 STEPS=15000 EVAL_EVERY=1000 PATIENCE=5 SAVECK=results/ck_enwik8_bitnet.pt $T bitnet

export DATA=shakespeare_bytes STEPS=250 EVAL_EVERY=250 EVAL_ITERS=10
for LR in 30 60 120; do
  LOADCK=results/ck_enwik8_bitnet.pt EDLR=$LR $T ed
done
for LR in 60 120 240; do
  EDLR=$LR $T ed
done
