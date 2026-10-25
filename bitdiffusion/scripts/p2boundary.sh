#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export SEEDS=0,1,2,3,4
export DH=192
export LOG=results/p2boundary.jsonl
R="uv run python -m bitdiffusion.run"

echo "=== 2-bit C fine-tune ==="
$R one edB binact bits 2 clamp 0.95
$R one edB binact bits 2 clamp 1.05

echo "=== 2-bit C=1.0 lr sweep ==="
$R one edB binact bits 2 clamp 1.0 lr 0.5
$R one edB binact bits 2 clamp 1.0 lr 2.0
$R one edB binact bits 2 clamp 1.0 lr 3.0

echo "=== top-level-at-threshold for higher bits ==="
$R one edB binact bits 3 clamp 1.0
$R one edB binact bits 4 clamp 1.0
$R one edB binact bits 8 clamp 1.0

echo "=== full 1-bit stack at 3 b/wt ==="
$R qgrad sr1_rht edB binact bits 2 clamp 1.0
