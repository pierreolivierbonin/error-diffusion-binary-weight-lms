#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export SEEDS=0,1,2,3,4
export DH=192
export LOG=results/p2clamp.jsonl
R="uv run python -m bitdiffusion.run"

echo "=== edB 2-bit, clamp inside flip threshold ==="
$R one edB binact bits 2 clamp 0.9
$R one edB binact bits 2 clamp 0.75
$R one edB binact bits 2 clamp 0.5

echo "=== edB 3-bit, clamp tuned ==="
$R one edB binact bits 3 clamp 0.9
$R one edB binact bits 3 clamp 0.75

echo "=== control: 4-bit with reduced clamp ==="
$R one edB binact bits 4 clamp 0.75
