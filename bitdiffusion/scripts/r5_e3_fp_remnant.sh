#!/bin/sh
cd "$(dirname "$0")/.."
export DATA=wikitext103 STEPS=15000 EVAL_EVERY=1000 PATIENCE=5 SEED=0
export BLOCK=128 EVAL_ITERS=25 EMBD=512 LAYERS=6 HEADS=8 BINACT=1
export LOG=results/r5_e3_fp_remnant.jsonl
[ -e "$LOG" ] && { echo "refusing to touch existing $LOG"; exit 1; }
T="uv run python -m bitdiffusion.train_gpt"
echo "commit $(git rev-parse --short HEAD) branch $(git branch --show-current)"

run_ed() {
  lr=$1; s=$SEED; tries=0
  while [ "$tries" -lt 3 ]; do
    rc=0; SEED=$s EDSCHED=linear EDLR=$lr STALLCHECK=1 DIVERGE=0.3 $T ed || rc=$?
    [ "$rc" -ne 3 ] && return 0
    tries=$((tries+1)); s=$((s+100)); echo "ed stall/diverge, retry seed $s"
  done
}

echo "=== (a) ED, frozen token+position embeddings, probe-set lr 150 ==="
FREEZE=wte,wpe run_ed 150
echo "=== (b) BitNet control, frozen token+position embeddings ==="
FREEZE=wte,wpe $T bitnet
echo "=== (c) ED, remnant Adam lr 1e-4, probe-set lr 250 ==="
ADAMLR=1e-4 run_ed 250
echo "=== (d) ED, remnant Adam lr decays linearly with the flip-lr, lr 225 ==="
ADAMSCHED=linear run_ed 225
echo "=== (e) ED, remnant frozen except LayerNorms (wte, wpe, head), probe-set lr 300 ==="
FREEZE=wte,wpe,head run_ed 300
echo "=== (f) BitNet control, remnant frozen except LayerNorms ==="
FREEZE=wte,wpe,head $T bitnet
echo "=== r5_e3 done ==="
