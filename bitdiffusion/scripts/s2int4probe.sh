#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export BITS=4 LOG=results/s2int4probe.jsonl DATA=enwik8 STEPS=250 EVAL_EVERY=250 EVAL_ITERS=10 BINACT=1 SEED=0
T="uv run python -m bitdiffusion.train_gpt"
for LR in 45 90 135 180; do EMBD=512 LAYERS=6 HEADS=8 EDLR=$LR $T ed; done
for LR in 15 30 45 60; do EMBD=256 LAYERS=4 HEADS=4 EDLR=$LR $T ed; done
for LR in 6 12 25;     do EMBD=128 LAYERS=2 HEADS=4 EDLR=$LR $T ed; done
