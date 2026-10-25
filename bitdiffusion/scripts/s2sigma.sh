#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export DH=192 SEEDS=0,1,2,3,4
export LOG=results/s2sigma.jsonl
R="uv run python -m bitdiffusion.run"

for LR in 0.2 0.5 1.0 2.0; do
  $R one ed2 bits 4 binact lr $LR clamp 1.0
done
for LR in 0.2 0.5 1.0 2.0; do
  $R one ed2 bits 4 binact lr $LR clamp 1.5
done
for LR in 0.2 0.5 1.0 2.0; do
  $R one ed2 bits 2 binact lr $LR clamp 1.0
done
$R one edB bits 8 binact lr 1.0 clamp 1.5
$R one edB bits 4 binact lr 1.0 clamp 1.0
