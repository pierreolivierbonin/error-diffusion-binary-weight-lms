#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export DH=192 SEEDS=0,1,2,3,4
export LOG=results/s2baselines.jsonl
R="uv run python -m bitdiffusion.run"

for MB in 0 8 4; do
  for LR in 10 30 100; do
    $R one edE bits 8 binact lr $LR clamp 1.0 mbits $MB beta 0.9
  done
done
for MB in 0 8; do
  for LR in 3 10 30; do
    $R one edE bits 8 binact lr $LR clamp 1.0 mbits $MB beta 0.99
  done
done
for CB in 8 4 2; do
  $R one cnt bits $CB binact
done
