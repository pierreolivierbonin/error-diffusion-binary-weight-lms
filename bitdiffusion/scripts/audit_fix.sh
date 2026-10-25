#!/bin/sh
set -e
cd "$(dirname "$0")/.."
T="uv run python -m bitdiffusion.train_gpt"
R="uv run python -m bitdiffusion.run"

echo "=== (1) Table 2 left: seed-1 ED int8 cells at the chosen lrs, binact=0 ==="
export LOG=results/audit_fix.jsonl BITS=8 SEED=1
EMBD=128 EDLR=100 $T ed
EMBD=256 EDLR=60  $T ed

echo "=== (2) Table 2 left d=512: BitNet and ED at BINACT=0, 2 seeds ==="
for SEED in 0 1; do
  export SEED
  EMBD=512 $T bitnet
  EMBD=512 EDLR=80 $T ed
done

echo "=== (6) MLP interaction: 1-bit grads on 4-bit and 2-bit buffers at C=1.0, 5 seeds ==="
export DH=192 SEEDS=0,1,2,3,4
LOG=results/audit_fix_mlp.jsonl $R qgrad sr1_rht edB binact bits 4 clamp 1.0
LOG=results/audit_fix_mlp.jsonl $R qgrad sr1_rht edB binact bits 2 clamp 1.0
