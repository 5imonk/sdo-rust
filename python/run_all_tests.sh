#!/usr/bin/env bash
# Run all tests (sdo, sdoclust, sdostream, sdostreamclust). Execute from project root or python/.
# Usage: ./run_all_tests.sh [optional args passed to each test script]
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

for name in sdo sdoclust sdostream sdostreamclust; do
    echo "========== $name =========="
    "$SCRIPT_DIR/$name/run_tests.sh" "$@" || true
done

echo "========== All test scripts finished =========="
