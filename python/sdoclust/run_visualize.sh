#!/usr/bin/env bash
# Run SDOclust visualization. Usage: ./run_visualize.sh [--arff file.arff] [--out-dir dir] [--no-plot]
set -e
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
exec python3 python/sdoclust/visualize_sdoclust.py "$@"
