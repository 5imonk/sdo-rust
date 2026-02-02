#!/usr/bin/env bash
# Run SDOstreamclust visualization (streaming + frames/video).
set -e
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
exec python3 python/sdostreamclust/visualize_sdostreamclust.py "$@"
