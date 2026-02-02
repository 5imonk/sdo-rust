#!/usr/bin/env bash
# Run SDOclust tests from project root.
set -e
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
exec python3 python/sdoclust/test_sdoclust.py "$@"
