#!/bin/sh
set -e
cd "$(dirname "$0")/.."
export DH=192 SEEDS=0,1,2,3,4
export LOG=results/s2bopalign2.jsonl
R="uv run python -m bitdiffusion.run"

BOPBITS=8 $R bop 3e-5 1e-4 3e-4 binact
BOPBITS=8 BOPALIGN=1 $R bop 3e-5 1e-4 3e-4 binact
BOPBITS=3 $R bop 3e-5 1e-4 3e-4 binact
BOPBITS=3 BOPALIGN=1 $R bop 3e-5 1e-4 3e-4 binact
BOPBITS=2 $R bop 3e-5 1e-4 3e-4 binact
