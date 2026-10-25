#!/bin/sh
cd "$(dirname "$0")/.."
export DATA=wikitext103 BLOCK=128 BINACT=1 SEED=0 BITS=4
export STEPS=250 EVAL_EVERY=250 EVAL_ITERS=5 EDSCHED=const
export LOG=results/r5_e5_int4_probe.jsonl
[ -e "$LOG" ] && { echo "refusing to touch existing $LOG"; exit 1; }
T="uv run python -m bitdiffusion.train_gpt"
echo "commit $(git rev-parse --short HEAD) branch $(git branch --show-current)"

echo "=== int4 drive probes on wikitext-103; int8 picks were 40 / 90 / 225, enwik8 int4 picks were 2-4x below int8 ==="
for LR in 10 20 40; do EMBD=128 LAYERS=2 HEADS=4 EDLR=$LR $T ed; done
for LR in 25 45 90; do EMBD=256 LAYERS=4 HEADS=4 EDLR=$LR $T ed; done
for LR in 45 90 150; do EMBD=512 LAYERS=6 HEADS=8 EDLR=$LR $T ed; done
echo "=== r5_e5 probes done ==="
