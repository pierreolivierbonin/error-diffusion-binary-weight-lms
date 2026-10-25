#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export BITS=4 LOG=results/s2int4probe.jsonl EMBD=256 LAYERS=4 HEADS=4 BINACT=1 SEED=0
export DATA=shakespeare_bytes STEPS=250 EVAL_EVERY=250 EVAL_ITERS=10
T="uv run python -m bitdiffusion.train_gpt"
for LR in 15 30 45 60; do LOADCK=results/ck_enwik8_bitnet.pt EDLR=$LR $T ed; done
