#!/usr/bin/env bash
# Run SDOstreamclust tests from project root.
set -e
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
exec python3 python/sdostreamclust/test_sdostreamclust.py "$@"
