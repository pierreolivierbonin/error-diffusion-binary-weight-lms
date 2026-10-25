#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export DATA=enwik8 STEPS=15000 EVAL_EVERY=1000 PATIENCE=5 SEED=0
export LOG=results/s2controls.jsonl
T="uv run python -m bitdiffusion.train_gpt"

EMBD=128 LAYERS=2 HEADS=4 BINACT=1 $T bitnet
EMBD=256 LAYERS=4 HEADS=4 BINACT=1 $T bitnet
EMBD=512 LAYERS=6 HEADS=8 BINACT=1 $T bitnet

EMBD=128 LAYERS=2 HEADS=4 BINACT=1 EDSCHED=linear EDLR=25 $T ed
EMBD=512 LAYERS=6 HEADS=8 BINACT=1 EDSCHED=linear EDLR=180 $T ed
