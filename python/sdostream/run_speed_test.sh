#!/usr/bin/env bash
# SDOstream: Laufzeit vs. Blockgröße und k. Usage: ./run_speed_test.sh [--points 500] [--k-values 100,200,400] [--block-sizes 1,25,50,100]
set -e
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
exec python3 python/sdostream/test_sdostream_speed_block_k.py "$@"
