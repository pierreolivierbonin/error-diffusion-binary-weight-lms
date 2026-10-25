#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export DATA=enwik8 STEPS=15000 EVAL_EVERY=1000 PATIENCE=5 SEED=0
export LOG=results/s2ladder.jsonl
T="uv run python -m bitdiffusion.train_gpt"

echo "=== size S1: 2L / 128 ==="
EMBD=128 LAYERS=2 HEADS=4 $T fp32
EMBD=128 LAYERS=2 HEADS=4 $T bitnet
EMBD=128 LAYERS=2 HEADS=4 BINACT=1 EDSCHED=linear EDLR=100 $T ed

echo "=== size S2: 4L / 256 ==="
EMBD=256 LAYERS=4 HEADS=4 $T fp32
EMBD=256 LAYERS=4 HEADS=4 $T bitnet
EMBD=256 LAYERS=4 HEADS=4 BINACT=1 EDSCHED=linear EDLR=60 $T ed

echo "=== size S3: 6L / 512 ==="
EMBD=512 LAYERS=6 HEADS=8 $T fp32
EMBD=512 LAYERS=6 HEADS=8 $T bitnet
EMBD=512 LAYERS=6 HEADS=8 BINACT=1 EDSCHED=linear EDLR=80 $T ed
