#!/bin/bash
# Fix permissions for target/prof directory if it was created with sudo

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [ -d "target/prof" ]; then
    echo "Fixing permissions for target/prof..."
    sudo chown -R "$USER:$USER" target/prof 2>/dev/null || {
        echo "Could not fix permissions. You may need to:"
        echo "  sudo chown -R $USER:$USER target/prof"
        echo "Or remove it:"
        echo "  sudo rm -rf target/prof"
    }
else
    echo "target/prof does not exist, no action needed."
fi

echo "Done."
