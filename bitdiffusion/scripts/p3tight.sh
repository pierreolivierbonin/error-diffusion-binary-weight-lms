#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export EMBD=256
export LOG=results/p3tight.jsonl
T="uv run python -m bitdiffusion.train_gpt"

echo "=== frozen controls (EDLR=0) ==="
SEED=0 EDLR=0 $T ed
SEED=0 EDLR=0 BINACT=1 $T ed

echo "=== bitnet + binact reference ==="
SEED=0 BINACT=1 $T bitnet
SEED=1 BINACT=1 $T bitnet

echo "=== ed + binact lr sweep (seed 0) ==="
SEED=0 BINACT=1 EDLR=30 $T ed
SEED=0 BINACT=1 EDLR=60 $T ed
SEED=0 BINACT=1 EDLR=100 $T ed

echo "=== ed + binact + sr1_rht grads lr sweep (seed 0) ==="
SEED=0 BINACT=1 QGRAD=sr1_rht EDLR=3 $T ed
SEED=0 BINACT=1 QGRAD=sr1_rht EDLR=10 $T ed
SEED=0 BINACT=1 QGRAD=sr1_rht EDLR=30 $T ed

echo "=== int4 buffer at C=1.0, best-lr configs (seed 0) ==="
SEED=0 BINACT=1 EDLR=60 BITS=4 $T ed
SEED=0 BINACT=1 QGRAD=sr1_rht EDLR=10 BITS=4 $T ed
