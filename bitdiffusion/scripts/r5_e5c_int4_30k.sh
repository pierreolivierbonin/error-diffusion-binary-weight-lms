#!/bin/sh
cd "$(dirname "$0")/.."
export DATA=wikitext103 BLOCK=128 EVAL_ITERS=25 STEPS=30000 EVAL_EVERY=1000 PATIENCE=5 SEED=0
export BINACT=1 BITS=4 EDSCHED=linear EMBD=512 LAYERS=6 HEADS=8 EDLR=45
export LOG=results/r5_e5c_int4_30k.jsonl
[ -e "$LOG" ] && { echo "refusing to touch existing $LOG"; exit 1; }
T="uv run python -m bitdiffusion.train_gpt"
echo "commit $(git rev-parse --short HEAD) branch $(git branch --show-current)"

run_ed() {
  s=$SEED; tries=0
  while [ "$tries" -lt 3 ]; do
    rc=0; SEED=$s STALLCHECK=1 DIVERGE=0.3 $T ed || rc=$?
    [ "$rc" -ne 3 ] && return 0
    tries=$((tries+1)); s=$((s+100)); echo "ed stall/diverge, retry seed $s"
  done
}

echo "=== int4 wikitext-103 19M, 30k steps, uncapped (Table 3 protocol) ==="
run_ed
echo "=== int4 wikitext-103 19M, 30k steps, cap 0.25 ==="
UCLIP=0.25 run_ed
echo "=== r5_e5c done ==="
