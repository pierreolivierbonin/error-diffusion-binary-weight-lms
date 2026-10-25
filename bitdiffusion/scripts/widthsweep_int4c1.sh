#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export SEEDS=0,1,2,3,4
export LOG=results/widthsweep_int4c1.jsonl
R="uv run python -m bitdiffusion.run"

for W in 64 96 128 192 256 384; do
  echo "=== width $W ==="
  DH=$W $R one edB binact bits 4 clamp 1.0
done
