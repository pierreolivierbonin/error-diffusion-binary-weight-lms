#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export SEEDS=0 DH=192 EPOCHS=25 DEVICE=cuda
export LOG=results/r6_halftone_trace.jsonl
export TRACE=results/r6_halftone_trace_w
[ -e "$LOG" ] && { echo "refusing to touch existing $LOG"; exit 1; }
[ -e "$TRACE.seed0.jsonl" ] && { echo "refusing to touch existing $TRACE.seed0.jsonl"; exit 1; }
echo "commit $(git rev-parse --short HEAD) branch $(git branch --show-current)"
echo "run.py sha256 $(sha256sum src/bitdiffusion/run.py | cut -c1-16) optim.py sha256 $(sha256sum src/bitdiffusion/optim.py | cut -c1-16)"
uv run python -m bitdiffusion.run one edB binact bits 2 clamp 1.0
echo "=== r6 done ==="
