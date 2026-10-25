#!/bin/sh
cd "$(dirname "$0")/.."
export EVAL_EVERY=1000 PATIENCE=5 BINACT=1 EDSCHED=linear UCLIP=0.25 STALLCHECK=1 DIVERGE=0.3
export LOG=results/r5_e3c_cap_ladder.jsonl
[ -e "$LOG" ] && { echo "refusing to touch existing $LOG"; exit 1; }
T="uv run python -m bitdiffusion.train_gpt"
echo "commit $(git rev-parse --short HEAD) branch $(git branch --show-current)"

echo "=== wikitext-103 6L/512, cap 0.25, lr 225, seeds 1 and 2, 15k (cell k replicates) ==="
export DATA=wikitext103 BLOCK=128 EVAL_ITERS=25 STEPS=15000
for S in 1 2; do
  SEED=$S EMBD=512 LAYERS=6 HEADS=8 EDLR=225 $T ed || echo "seed $S exit $?"
done
echo "=== wikitext-103 6L/512, cap 0.25, lr 225, seed 0, 30k (paper's doubled-budget cell; BitNet 30k = 4.358) ==="
STEPS=30000 SEED=0 EMBD=512 LAYERS=6 HEADS=8 EDLR=225 $T ed || echo "30k exit $?"
echo "=== wikitext-103 smaller rungs with the cap, registered lrs 40 / 90 ==="
SEED=0 EMBD=128 LAYERS=2 HEADS=4 EDLR=40 $T ed || echo "128 exit $?"
SEED=0 EMBD=256 LAYERS=4 HEADS=4 EDLR=90 $T ed || echo "256 exit $?"
echo "=== enwik8 smaller rungs with the cap, registered lrs 25 / 60 (E2 owed) ==="
export DATA=enwik8 BLOCK=256 EVAL_ITERS=50
SEED=0 EMBD=128 LAYERS=2 HEADS=4 EDLR=25 $T ed || echo "e128 exit $?"
SEED=0 EMBD=256 LAYERS=4 HEADS=4 EDLR=60 $T ed || echo "e256 exit $?"
echo "=== r5_e3c done ==="
