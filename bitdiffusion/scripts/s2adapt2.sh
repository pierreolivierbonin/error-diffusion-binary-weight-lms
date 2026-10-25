#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export EMBD=256 LAYERS=4 HEADS=4 BINACT=1 SEED=0
export DATA=shakespeare_bytes STEPS=5000 EVAL_EVERY=250 PATIENCE=8 EVAL0=1
export LOG=results/s2adapt.jsonl
CK=results/ck_enwik8_bitnet.pt
T="uv run python -m bitdiffusion.train_gpt"

LOADCK=$CK EDSCHED=linear EDLR=30 $T ed
LOADCK=$CK EDSCHED=linear EDLR=60 $T ed
LOADCK=$CK $T bitnet
EDSCHED=linear EDLR=120 $T ed
