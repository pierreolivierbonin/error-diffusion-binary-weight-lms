#!/bin/sh
cd "$(dirname "$0")/.."
export DATA=wikitext103 STEPS=15000 EVAL_EVERY=1000 PATIENCE=5
export BLOCK=128 EVAL_ITERS=25 EMBD=512 LAYERS=6 HEADS=8 BINACT=1
export LOG=results/r5_e3b_remnant_lr.jsonl
[ -e "$LOG" ] && { echo "refusing to touch existing $LOG"; exit 1; }
T="uv run python -m bitdiffusion.train_gpt"
echo "commit $(git rev-parse --short HEAD) branch $(git branch --show-current)"

run_ed() {
  lr=$1; s=$2; tries=0
  while [ "$tries" -lt 3 ]; do
    rc=0; SEED=$s EDSCHED=linear EDLR=$lr STALLCHECK=1 DIVERGE=0.3 $T ed || rc=$?
    [ "$rc" -ne 3 ] && return 0
    tries=$((tries+1)); s=$((s+100)); echo "ed stall/diverge, retry seed $s"
  done
}

echo "=== (g) BitNet control at remnant lr 1e-4 (fairness for cell c) ==="
SEED=0 ADAMLR=1e-4 $T bitnet
echo "=== (h) ED remnant lr 1e-4, flip-lr 250, second seed ==="
ADAMLR=1e-4 run_ed 250 1
echo "=== (i) ED remnant lr 3e-5, flip-lr from probe (fill in) ==="
ADAMLR=3e-5 run_ed 400 0
echo "=== (j) BitNet control at remnant lr 3e-5 ==="
SEED=0 ADAMLR=3e-5 $T bitnet
echo "=== (k) ED plain arm, remnant 3e-4, lr 225, per-step cap 0.25 (E2 cap on wikitext) ==="
UCLIP=0.25 run_ed 225 0
echo "=== (l) ED remnant 1e-4, lr 250, cap 0.25, the seed that diverged uncapped ==="
UCLIP=0.25 ADAMLR=1e-4 run_ed 250 0
echo "=== r5_e3b done ==="
