#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export SEEDS=0,1,2,3,4
export LOG=results/headline.jsonl
R="uv run python -m bitdiffusion.run"

echo "=== w96 fp-acts ==="
DH=96 $R base
DH=96 $R edA 1.0
DH=96 $R edB 1.0
DH=96 $R edBbits 4 2 1
DH=96 $R sr 1.0
DH=96 $R bop 1e-4 1e-3

echo "=== w96 binact ==="
DH=96 $R binact

echo "=== w192 binact ==="
DH=192 $R one frozen binact bits 8
DH=192 $R one edB binact bits 8
DH=192 $R one edB binact bits 4

echo "=== w192 binact 1-bit grads ==="
DH=192 $R qgrad det1 edB binact bits 8
DH=192 $R qgrad sr1 edB binact bits 8
DH=192 $R qgrad sr1_rht edB binact bits 8
DH=192 $R qgrad sr1_rht edB binact bits 4
