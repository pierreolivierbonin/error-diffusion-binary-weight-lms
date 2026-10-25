#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export SEEDS=0,1,2,3,4
export DH=192
export LOG=results/p2floor.jsonl
R="uv run python -m bitdiffusion.run"

echo "=== floor characterization (edB baseline) ==="
$R one edB binact bits 3
$R one edB binact bits 2

echo "=== edS spatial, 2-bit ==="
$R one edS binact bits 2 alpha 0.75
$R one edS binact bits 2 alpha 0.5
$R one edS binact bits 2 alpha 0.25

echo "=== edS spatial, 3-bit ==="
$R one edS binact bits 3 alpha 0.5

echo "=== edH hysteresis, 2-bit ==="
$R one edH binact bits 2 delta 0.25
$R one edH binact bits 2 delta 0.5

echo "=== edH hysteresis, 3-bit ==="
$R one edH binact bits 3 delta 0.5
