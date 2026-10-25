#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export SEEDS=0,1,2,3,4
export DH=192
export LOG=results/s2budget.jsonl
R="uv run python -m bitdiffusion.run"

echo "=== 9 b/wt splits (reference: edB int8 = 56.42 from P0) ==="
$R one edM binact bits 4 mbits 4 beta 0.9 lr 0.1
$R one edM binact bits 4 mbits 4 beta 0.99 lr 0.01
$R one edM binact bits 6 mbits 2 beta 0.9 lr 0.1

echo "=== drive-rule check: mis-scaled lr should thrash ==="
$R one edM binact bits 4 mbits 4 beta 0.9 lr 1.0

echo "=== 5 b/wt splits (reference: edB int4 C=1.0 = 56.35) ==="
$R one edM binact bits 2 mbits 2 beta 0.9 lr 0.1 clamp 1.0
