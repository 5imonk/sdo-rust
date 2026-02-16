#!/bin/bash
# Öffne das Flamegraph im Browser

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [ -f flamegraph.svg ]; then
    echo "Opening flamegraph.svg..."
    if command -v xdg-open &> /dev/null; then
        xdg-open flamegraph.svg
    elif command -v open &> /dev/null; then
        open flamegraph.svg
    else
        echo "Flamegraph saved to: $(pwd)/flamegraph.svg"
        echo "Open it manually in your browser."
    fi
else
    echo "Error: flamegraph.svg not found."
    echo "Run: ./scripts/profile.sh flamegraph <benchmark_name>"
    exit 1
fi
