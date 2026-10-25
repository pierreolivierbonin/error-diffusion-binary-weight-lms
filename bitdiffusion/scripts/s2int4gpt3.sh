#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export BITS=4 LOG=results/s2int4gpt3.jsonl BINACT=1 EDSCHED=linear
T="uv run python -m bitdiffusion.train_gpt"

echo "=== int4 enwik8 ladder at probe-set lrs (12/30/45), 3 seeds ==="
export DATA=enwik8 STEPS=15000 EVAL_EVERY=1000 PATIENCE=5
run512() {
  s=$1; tries=0
  while [ "$tries" -lt 3 ]; do
    rc=0; SEED=$s EMBD=512 LAYERS=6 HEADS=8 EDLR=45 STALLCHECK=1 $T ed || rc=$?
    [ "$rc" -ne 3 ] && return 0
    tries=$((tries+1)); s=$((s+100)); echo "512 stall, retry seed $s"
  done
}
for S in 0 1 2; do
  SEED=$S EMBD=128 LAYERS=2 HEADS=4 EDLR=12 $T ed
  SEED=$S EMBD=256 LAYERS=4 HEADS=4 EDLR=30 $T ed
  run512 $S
done

echo "=== int4 50k convergence, d=256 shakespeare-char, lr 30 and 45 ==="
unset DATA
export STEPS=50000 EVAL_EVERY=1000 PATIENCE=0 EMBD=256 LAYERS=4 HEADS=4 SEED=0
for LR in 30 45; do EDLR=$LR $T ed; done
