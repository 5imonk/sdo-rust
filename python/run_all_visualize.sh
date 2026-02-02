#!/usr/bin/env bash
# Run all visualizations (SDOclust, SDOstreamclust). Execute from project root or python/.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "========== SDOclust visualize =========="
"$SCRIPT_DIR/sdoclust/run_visualize.sh" "$@" || true

echo "========== SDOstreamclust visualize =========="
"$SCRIPT_DIR/sdostreamclust/run_visualize.sh" "$@" || true

echo "========== All visualization scripts finished =========="
