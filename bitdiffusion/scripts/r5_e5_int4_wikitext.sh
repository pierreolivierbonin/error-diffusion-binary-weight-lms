#!/bin/sh
cd "$(dirname "$0")/.."
export DATA=wikitext103 BLOCK=128 EVAL_ITERS=25 STEPS=15000 EVAL_EVERY=1000 PATIENCE=5 SEED=0
export BINACT=1 BITS=4 EDSCHED=linear
export LOG=results/r5_e5_int4_wikitext.jsonl
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

echo "=== int4 wikitext-103 ladder, uncapped (Table 3 protocol), probe-set lrs 20 / 45 / 45 ==="
EMBD=128 LAYERS=2 HEADS=4 EDLR=20 run_ed
EMBD=256 LAYERS=4 HEADS=4 EDLR=45 run_ed
EMBD=512 LAYERS=6 HEADS=8 EDLR=45 run_ed
echo "=== int4 wikitext-103 19M with the cap 0.25 (Appendix F protocol) ==="
UCLIP=0.25 EMBD=512 LAYERS=6 HEADS=8 EDLR=45 run_ed
echo "=== r5_e5 done ==="
