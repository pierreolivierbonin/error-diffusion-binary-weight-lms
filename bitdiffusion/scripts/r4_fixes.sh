#!/bin/sh
set -e
cd "$(dirname "$0")/.."
T="uv run python -m bitdiffusion.train_gpt"
R="uv run python -m bitdiffusion.run"

echo "=== (a) MLP exchange curve: int8 at C=1.0 across widths, 5 seeds ==="
export SEEDS=0,1,2,3,4
for W in 64 96 128 192 256 384; do
  DH=$W LOG=results/r4_widthsweep_int8c1.jsonl $R one edB binact bits 8 clamp 1.0
done

echo "=== (b) MLP interaction: 1-bit grads on 8-bit buffer at C=1.0, w192, 5 seeds ==="
DH=192 LOG=results/r4_mlp.jsonl $R qgrad sr1_rht edB binact bits 8 clamp 1.0

echo "=== (c) Table 2 left, seed 2 for every 2k-step cell (binact=0) ==="
export SEED=2 LOG=results/r4_gpt_seed2.jsonl
unset SEEDS
for EMBD in 128 256 512; do
  export EMBD
  $T fp32
  $T bitnet
done
BITS=8 EMBD=128 EDLR=100 $T ed
BITS=8 EMBD=256 EDLR=60  $T ed
BITS=8 EMBD=512 EDLR=80  $T ed
for LR in 100 150; do BITS=4 EMBD=128 EDLR=$LR $T ed; done
for LR in 60 90;   do BITS=4 EMBD=256 EDLR=$LR $T ed; done
for LR in 80 120;  do BITS=4 EMBD=512 EDLR=$LR $T ed; done

echo "=== (d) deploy-then-adapt: Adam lr sweep for the latent continuation arm, 3 seeds ==="
export EMBD=256 LAYERS=4 HEADS=4 BINACT=1 DATA=shakespeare_bytes STEPS=5000 EVAL_EVERY=250 PATIENCE=8 EVAL0=1
export LOG=results/r4_adapt.jsonl
for SEED in 0 1 2; do
  export SEED
  for ADAMLR in 1e-4 1e-3; do
    ADAMLR=$ADAMLR LOADCK=results/ck_enwik8_bitnet.pt $T bitnet
  done
done
echo "=== r4_fixes done ==="
