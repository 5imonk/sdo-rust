#!/bin/bash
# Profiling-Skript für SDO mit cargo flamegraph oder perf
# Usage: ./scripts/profile.sh [flamegraph|perf|bench] [benchmark_name]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Preserve original user's HOME and PATH when running with sudo
if [ -n "$SUDO_USER" ]; then
    ORIGINAL_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
    export HOME="$ORIGINAL_HOME"
    export PATH="$ORIGINAL_HOME/.cargo/bin:$PATH"
fi

MODE="${1:-flamegraph}"
BENCHMARK="${2:-all}"

echo "SDO Profiling - Mode: $MODE, Benchmark: $BENCHMARK"
echo "=========================================="

case "$MODE" in
    flamegraph)
        # Check if flamegraph is available (PATH should already include ~/.cargo/bin from above)
        if ! command -v flamegraph &> /dev/null; then
            echo "Error: flamegraph not found."
            echo ""
            echo "Install with:"
            echo "  cargo install flamegraph"
            echo ""
            echo "If running with sudo, the script should preserve your PATH."
            echo "If it doesn't work, try:"
            echo "  sudo -E $0 flamegraph $BENCHMARK"
            echo ""
            echo "Or use 'bench' mode instead:"
            echo "  $0 bench $BENCHMARK"
            exit 1
        fi
        
        # Export PATH to include cargo bin if needed
        if [ -d "$HOME/.cargo/bin" ] && [[ ":$PATH:" != *":$HOME/.cargo/bin:"* ]]; then
            export PATH="$HOME/.cargo/bin:$PATH"
        fi
        
        # Check perf_event_paranoid setting
        if [ -f /proc/sys/kernel/perf_event_paranoid ]; then
            PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid)
            if [ "$PARANOID" -gt 1 ]; then
                echo "⚠️  Warning: perf_event_paranoid is $PARANOID (needs to be <= 1 for profiling)"
                echo ""
                echo "To fix this, run as root:"
                echo "  sudo sysctl -w kernel.perf_event_paranoid=1"
                echo ""
                echo "Or temporarily:"
                echo "  echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid"
                echo ""
                echo "Trying with --no-default-features (may be slower)..."
                FLAMEGRAPH_ARGS="--no-default-features"
            else
                FLAMEGRAPH_ARGS=""
            fi
        else
            FLAMEGRAPH_ARGS=""
        fi
        
        # Use user's tmp directory for log file
        LOG_FILE="${TMPDIR:-/tmp}/flamegraph_${USER:-user}_$$.log"
        
        # Remove old flamegraph if it exists and is owned by root
        if [ -f flamegraph.svg ]; then
            if [ ! -w flamegraph.svg ]; then
                echo "Removing old flamegraph.svg (permission issue)..."
                sudo rm -f flamegraph.svg 2>/dev/null || rm -f flamegraph.svg 2>/dev/null || true
            else
                rm -f flamegraph.svg
            fi
        fi
        
        echo "Running cargo flamegraph --bench profiling_benchmarks..."
        echo "Log file: $LOG_FILE"
        
        if cargo flamegraph $FLAMEGRAPH_ARGS --bench profiling_benchmarks --profile prof -- "$BENCHMARK" 2>&1 | tee "$LOG_FILE"; then
            if [ -f flamegraph.svg ]; then
                # Fix ownership if created by root
                CURRENT_OWNER=$(stat -c '%U' flamegraph.svg 2>/dev/null || echo "")
                if [ -n "$SUDO_USER" ] && [ "$CURRENT_OWNER" != "$SUDO_USER" ] && [ "$CURRENT_OWNER" = "root" ]; then
                    echo "Fixing ownership of flamegraph.svg..."
                    sudo chown "$SUDO_USER:$SUDO_USER" flamegraph.svg 2>/dev/null || true
                fi
                # Ensure file is writable by user
                chmod 644 flamegraph.svg 2>/dev/null || sudo chmod 644 flamegraph.svg 2>/dev/null || true
                
                echo ""
                echo "✓ Flamegraph saved to: flamegraph.svg"
                echo "  Open with: xdg-open flamegraph.svg (Linux) or open flamegraph.svg (macOS)"
                echo "  File size: $(du -h flamegraph.svg 2>/dev/null | cut -f1 || echo 'unknown')"
            else
                echo "⚠️  Warning: flamegraph.svg was not created. Check $LOG_FILE for details."
                echo ""
                echo "Common issues:"
                echo "  - Permission denied: Try 'sudo rm -f flamegraph.svg' first"
                echo "  - perf_event_paranoid: Run './scripts/setup_profiling.sh'"
            fi
        else
            echo ""
            echo "❌ Flamegraph failed. Common solutions:"
            echo "  1. Run as root: sudo $0 flamegraph $BENCHMARK"
            echo "  2. Set perf_event_paranoid: sudo sysctl -w kernel.perf_event_paranoid=1"
            echo "  3. Use 'bench' mode instead: $0 bench $BENCHMARK"
            exit 1
        fi
        ;;
    
    perf)
        if ! command -v perf &> /dev/null; then
            echo "Error: perf not found. Install perf for your Linux distribution."
            exit 1
        fi
        echo "Running perf record on benchmark..."
        cargo build --release --bench profiling_benchmarks 2>&1 | grep -v "^   Compiling" || true
        LOG_FILE="${TMPDIR:-/tmp}/perf_${USER:-user}_$$.log"
        echo "Log file: $LOG_FILE"
        perf record --call-graph dwarf \
            target/release/deps/profiling_benchmarks-* \
            --bench "$BENCHMARK" 2>&1 | tee "$LOG_FILE"
        echo "✓ Perf data saved to: perf.data"
        echo "  Analyze with: perf report"
        echo "  Or: perf report --stdio | less"
        ;;
    
    bench)
        LOG_FILE="${TMPDIR:-/tmp}/bench_${USER:-user}_$$.log"
        echo "Running criterion benchmarks..."
        echo "Log file: $LOG_FILE"
        cargo bench --bench profiling_benchmarks -- "$BENCHMARK" 2>&1 | tee "$LOG_FILE"
        echo ""
        echo "✓ Benchmark results saved to: target/criterion/"
        echo "  View HTML reports in: target/criterion/<benchmark_name>/report/index.html"
        ;;
    
    bench-simple)
        LOG_FILE="${TMPDIR:-/tmp}/bench_simple_${USER:-user}_$$.log"
        echo "Running simple benchmark (no profiling overhead)..."
        cargo bench --bench profiling_benchmarks -- "$BENCHMARK" 2>&1 | grep -E "(test|benchmark|time:|ns/iter)" | tee "$LOG_FILE"
        ;;
    
    *)
        echo "Usage: $0 [flamegraph|perf|bench] [benchmark_name]"
        echo ""
        echo "Modes:"
        echo "  flamegraph    - Generate flamegraph.svg (requires: cargo install flamegraph)"
        echo "  perf          - Record with perf (Linux only, requires perf)"
        echo "  bench         - Run criterion benchmarks"
        echo ""
        echo "Examples:"
        echo "  $0 flamegraph benchmark_sdostream_learn_impl"
        echo "  $0 perf benchmark_search_neighbors_unified_batch"
        echo "  $0 bench benchmark_distance_matrix_rebuild"
        exit 1
        ;;
esac
