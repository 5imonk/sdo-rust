#!/usr/bin/env bash
# Run SDO tests from project root. Usage: ./run_tests.sh [--test 1 2 4]
set -e
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
exec python3 python/sdo/test_sdo.py "$@"
