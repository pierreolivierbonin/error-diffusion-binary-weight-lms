#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export BITS=4 LOG=results/s2int4gpt.jsonl
T="uv run python -m bitdiffusion.train_gpt"

echo "=== (c') int4 adapt at lr 30, seeds 1 2 ==="
export EMBD=256 LAYERS=4 HEADS=4 BINACT=1 DATA=shakespeare_bytes STEPS=5000 EVAL_EVERY=250 PATIENCE=8 EVAL0=1
for SEED in 1 2; do
  SEED=$SEED LOADCK=results/ck_enwik8_bitnet.pt EDSCHED=linear EDLR=30 $T ed
done

echo "=== (d') int4 ladder 512 stall retries (seed+100) ==="
unset EVAL0
export DATA=enwik8 STEPS=15000 EVAL_EVERY=1000 PATIENCE=5 BINACT=1 EDSCHED=linear
for S in 100 102; do
  SEED=$S EMBD=512 LAYERS=6 HEADS=8 EDLR=180 STALLCHECK=1 $T ed || echo "ladder 512 seed $S exit $?"
done
