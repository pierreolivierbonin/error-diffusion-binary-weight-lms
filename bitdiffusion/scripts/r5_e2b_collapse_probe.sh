#!/bin/sh
cd "$(dirname "$0")/.."
export DATA=enwik8 STEPS=15000 EMBD=512 LAYERS=6 HEADS=8 BINACT=1 EDSCHED=linear EDLR=180 BITS=8
export EDINIT=0.5 DIAG=1
export LOG=results/r5_e2b_collapse_probe.jsonl
[ -e "$LOG" ] && { echo "refusing to touch existing $LOG"; exit 1; }
T="uv run python -m bitdiffusion.train_gpt"
echo "commit $(git rev-parse --short HEAD) branch $(git branch --show-current)"

for S in 0 4; do
  CK=results/r5_e2b_ck_seed${S}_step2000.pt
  [ -e "$CK" ] && { echo "refusing to touch existing $CK"; exit 1; }
  echo "=== seed $S: replay to step 2000 with checkpoint (seed 0 collapsed at 2.5k in r5_e2; seed 4 was clean) ==="
  SEED=$S EVAL_EVERY=500 STOPAT=2000 FULLCK=$CK $T ed || echo "seed $S exit $?"
  echo "=== seed $S: resume 2000 -> 3000 at 100-step evals with diagnostics ==="
  SEED=$S EVAL_EVERY=100 STOPAT=3000 RESUME=$CK $T ed || echo "seed $S resume exit $?"
done
echo "=== r5_e2b done ==="
