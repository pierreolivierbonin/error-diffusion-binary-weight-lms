#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export STEPS=50000 EVAL_EVERY=1000 PATIENCE=8 EMBD=384 LAYERS=6 HEADS=6
export EDSCHED=linear STALLCHECK=1 DROPOUT=0.2
export LOG=results/parity.jsonl
T="uv run python -m bitdiffusion.train_gpt"
EDLR_GELU="${EDLR_GELU:-300}"
EDLR_BINACT="${EDLR_BINACT:-280}"

run_ed() {
  seed=$1
  lr=$2
  ba=$3
  tag=$4
  tries=0
  while [ "$tries" -lt 3 ]; do
    rc=0
    SEED=$seed EDLR=$lr BINACT=$ba FULLCK=results/ck/parity_${tag}_s$seed.pt $T ed || rc=$?
    [ "$rc" -ne 3 ] && return $rc
    tries=$((tries + 1))
    seed=$((seed + 100))
  done
  echo "parity: $tag exhausted stall retries" >&2
  return 1
}

for S in ${PARITY_SEEDS:-0}; do
  run_ed "$S" "$EDLR_GELU" 0 gelu
  run_ed "$S" "$EDLR_BINACT" 1 binact
  SEED=$S EDLR=$EDLR_GELU BINACT=0 DROPOUT=0 FULLCK=results/ck/parity_nodrop_s$S.pt $T ed
done
