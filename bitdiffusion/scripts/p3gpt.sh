#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export LOG=results/p3gpt.jsonl
T="uv run python -m bitdiffusion.train_gpt"

for EMBD in 256 128; do
  export EMBD
  for SEED in 0 1; do
    export SEED
    echo "=== embd=$EMBD seed=$SEED ==="
    $T fp32
    $T bitnet
    EDLR=3 $T ed
    EDLR=10 $T ed
    EDLR=30 $T ed
  done
done
