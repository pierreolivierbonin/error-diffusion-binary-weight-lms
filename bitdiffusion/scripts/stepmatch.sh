#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export SEEDS=0,1,2,3,4
export LOG=results/stepmatch.jsonl
R="uv run python -m bitdiffusion.run"

for B in 128 512 2048; do
  EP=$((25 * B / 128))
  for BITS in 2 8; do
    echo "=== batch=$B epochs=$EP bits=$BITS (step-matched) ==="
    BATCH=$B EPOCHS=$EP DH=96 $R batchsweep $BITS
  done
done
