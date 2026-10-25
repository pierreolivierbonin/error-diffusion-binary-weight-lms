#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export DH=192 SEEDS=0,1,2,3,4
export LOG=results/s2intlat.jsonl
R="uv run python -m bitdiffusion.run"

for LR in 0.3 1.0 3.0; do
  $R one intlat bits 9 binact lr $LR
done
for LR in 0.3 1.0 3.0; do
  $R one intlat bits 5 binact lr $LR
done
