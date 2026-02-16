#!/bin/bash
# Setup script für Profiling - setzt perf_event_paranoid auf 1

set -e

echo "Setting up profiling environment..."
echo "===================================="

# Check current value
CURRENT=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "N/A")
echo "Current perf_event_paranoid: $CURRENT"

if [ "$CURRENT" != "1" ] && [ "$CURRENT" != "-1" ]; then
    echo ""
    echo "Setting perf_event_paranoid to 1 (requires sudo)..."
    echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid
    
    NEW_VALUE=$(cat /proc/sys/kernel/perf_event_paranoid)
    if [ "$NEW_VALUE" = "1" ]; then
        echo "✓ Successfully set perf_event_paranoid to 1"
    else
        echo "❌ Failed to set perf_event_paranoid"
        exit 1
    fi
else
    echo "✓ perf_event_paranoid is already set correctly ($CURRENT)"
fi

echo ""
echo "To make this permanent, add to /etc/sysctl.conf:"
echo "  kernel.perf_event_paranoid = 1"
echo ""
echo "Then run: sudo sysctl -p"
echo ""
echo "You can now run profiling:"
echo "  ./scripts/profile.sh flamegraph benchmark_sdostream_learn_impl"
