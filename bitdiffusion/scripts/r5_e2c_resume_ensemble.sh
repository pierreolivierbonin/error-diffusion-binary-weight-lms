#!/bin/sh
cd "$(dirname "$0")/.."
export DATA=enwik8 STEPS=15000 EMBD=512 LAYERS=6 HEADS=8 BINACT=1 EDSCHED=linear EDLR=180 BITS=8
export EDINIT=0.5 DIAG=1 EVAL_EVERY=100 STOPAT=3000
export LOG=results/r5_e2c_resume_ensemble.jsonl
[ -e "$LOG" ] && { echo "refusing to touch existing $LOG"; exit 1; }
T="uv run python -m bitdiffusion.train_gpt"
echo "commit $(git rev-parse --short HEAD) branch $(git branch --show-current)"

for S in 0 4; do
  CK=results/r5_e2b_ck_seed${S}_step2000.pt
  for REP in 1 2 3 4 5 6; do
    echo "=== seed $S resume rep $REP: identical checkpoint and RNG state, 2000 -> 3000 ==="
    SEED=$S RESUME=$CK $T ed || echo "seed $S rep $REP exit $?"
  done
done
echo "=== r5_e2c done ==="
