#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export BITS=4 LOG=results/s2int4gpt.jsonl
T="uv run python -m bitdiffusion.train_gpt"

echo "=== (a) int4 2k-step cells, Table 2 left (binact=0, lrs bracket int8 optimum) ==="
for SEED in 0 1; do
  export SEED
  for LR in 100 150; do EMBD=128 EDLR=$LR $T ed; done
  for LR in 60 90;   do EMBD=256 EDLR=$LR $T ed; done
  for LR in 80 120;  do EMBD=512 EDLR=$LR $T ed; done
done

echo "=== (c) int4 deploy-then-adapt, 3 seeds ==="
export EMBD=256 LAYERS=4 HEADS=4 BINACT=1 DATA=shakespeare_bytes STEPS=5000 EVAL_EVERY=250 PATIENCE=8 EVAL0=1
for SEED in 0 1 2; do
  export SEED
  LOADCK=results/ck_enwik8_bitnet.pt EDSCHED=linear EDLR=60 $T ed
done
SEED=0 LOADCK=results/ck_enwik8_bitnet.pt EDSCHED=linear EDLR=30 $T ed
SEED=0 LOADCK=results/ck_enwik8_bitnet.pt EDSCHED=linear EDLR=90 $T ed

echo "=== (d) int4 enwik8 ladder, 3 seeds ==="
export DATA=enwik8 STEPS=15000 EVAL_EVERY=1000 PATIENCE=5 BINACT=1 EDSCHED=linear
unset EVAL0
for S in 0 1 2; do
  SEED=$S EMBD=128 LAYERS=2 HEADS=4 EDLR=25  $T ed
  SEED=$S EMBD=256 LAYERS=4 HEADS=4 EDLR=60  $T ed
  SEED=$S EMBD=512 LAYERS=6 HEADS=8 EDLR=180 STALLCHECK=1 $T ed || echo "ladder 512 seed $S exit $?"
done
