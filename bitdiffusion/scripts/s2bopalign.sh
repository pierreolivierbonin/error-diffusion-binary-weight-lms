#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export DH=192 SEEDS=0,1,2,3,4
export LOG=results/s2bopalign.jsonl
R="uv run python -m bitdiffusion.run"

$R bop 3e-5 1e-4 3e-4 binact
BOPBITS=4 $R bop 3e-5 1e-4 3e-4 binact
BOPBITS=4 BOPALIGN=1 $R bop 3e-5 1e-4 3e-4 binact
BOPBITS=2 BOPALIGN=1 $R bop 3e-5 1e-4 3e-4 binact
