#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export EMBD=256 SEED=0 BINACT=1 EDLR=10 PATIENCE=4
export LOG=results/p3qfix.jsonl
T="uv run python -m bitdiffusion.train_gpt"

echo "=== screen: RHT block size ==="
QGRAD=sr1_rht QBLOCK=64 $T ed
QGRAD=sr1_rht QBLOCK=128 $T ed

echo "=== screen: clip-then-SR ==="
QGRAD=sr1_clip QCLIP=3 $T ed
QGRAD=sr1_clip QCLIP=5 $T ed

echo "=== screen: RHT + clip combo ==="
QGRAD=sr1_rht_clip QBLOCK=128 QCLIP=3 $T ed
