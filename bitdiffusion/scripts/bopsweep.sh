#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export SEEDS=0,1,2,3,4
export LOG=results/bopsweep.jsonl
R="uv run python -m bitdiffusion.run"

for G in 1e-3 1e-2 1e-1; do
  echo "=== bop gamma=$G ==="
  GAMMA=$G DH=96 $R bop 1e-5 1e-4 1e-3 1e-2
done

for G in 1e-3 1e-2 1e-1; do
  echo "=== bop gamma=$G int8 EMA ==="
  GAMMA=$G BOPBITS=8 DH=96 $R bop 1e-5 1e-4 1e-3 1e-2
done
