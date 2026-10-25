#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export SEEDS=0,1,2,3,4
export LOG=results/widthsweep.jsonl
R="uv run python -m bitdiffusion.run"

for W in 64 96 128 192 256 384; do
  echo "=== width $W ==="
  DH=$W $R base
  DH=$W $R binact
  DH=$W $R one edB binact bits 4
  DH=$W $R qgrad sr1_rht edB binact bits 8
  GAMMA=1e-3 BOPBITS=8 DH=$W $R bop 1e-5 1e-4 1e-3 binact
done
