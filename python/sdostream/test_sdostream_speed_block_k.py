#!/usr/bin/env python3
"""
SDOstream: Laufzeit in Abhängigkeit von Blockgröße und Modellgröße (k).
Misst total time und time per point für verschiedene (k, block_size).
"""

import sys
import os
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from sdo import SDOstream
except ImportError as e:
    print(f"Fehler: sdo-Modul nicht gefunden: {e}")
    print("Bitte im Projektroot 'maturin develop' ausführen.")
    sys.exit(1)

import numpy as np


def run_sdostream_with_timing(
    data: np.ndarray,
    k: int,
    block_size: int,
    dimension: int,
    t_fading: float = 100.0,
    t_sampling: float = 100.0,
    x: int = 3,
    rho: float = 0.2,
):
    """
    Führt SDOstream.learn() in Batches der Größe block_size aus und misst die reine Learn-Zeit.
    Returns: (total_seconds, time_per_point_ms, num_batches).
    """
    model = SDOstream(
        k=k,
        x=x,
        t_fading=t_fading,
        t_sampling=t_sampling,
        rho=rho,
        dimension=dimension,
    )
    n = len(data)
    times_arr = np.arange(n, dtype=np.float64)
    t0 = time.perf_counter()
    for i in range(0, n, block_size):
        chunk = data[i : i + block_size]
        chunk_times = times_arr[i : i + block_size]
        model.learn(chunk, time=chunk_times)
    total_seconds = time.perf_counter() - t0
    num_batches = (n + block_size - 1) // block_size
    time_per_point_ms = (total_seconds / n) * 1000.0 if n else 0.0
    return total_seconds, time_per_point_ms, num_batches


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="SDOstream – Laufzeit vs. Blockgröße und k"
    )
    parser.add_argument(
        "--points",
        type=int,
        default=1000,
        help="Anzahl Datenpunkte (default: 1000)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=2,
        help="Dimension (default: 2)",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="100,200,400,800",
        help="Komma-getrennte k-Werte (default: 100,200,400,800)",
    )
    parser.add_argument(
        "--block-sizes",
        type=str,
        default="1,10,25,50,100",
        help="Komma-getrennte Blockgrößen (default: 1,10,25,50,100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed für reproduzierbare Daten (default: 42)",
    )
    args = parser.parse_args()

    n_points = args.points
    dimension = args.dim
    k_values = [int(x.strip()) for x in args.k_values.split(",")]
    block_sizes = [int(x.strip()) for x in args.block_sizes.split(",")]

    np.random.seed(args.seed)
    data = np.random.rand(n_points, dimension).astype(np.float64)

    print("=" * 70)
    print("SDOstream – Laufzeit vs. Blockgröße und k")
    print("=" * 70)
    print(f"  Punkte: {n_points}, Dimension: {dimension}")
    print(f"  k-Werte: {k_values}")
    print(f"  Blockgrößen: {block_sizes}")
    print()

    # Header
    header = f"{'k':>6} | {'block':>6} | {'total(s)':>10} | {'ms/pt':>8} | {'batches':>8}"
    print(header)
    print("-" * len(header))

    results = []
    for k in k_values:
        if k > n_points:
            continue
        for block_size in block_sizes:
            if block_size < 1 or block_size > n_points:
                continue
            total_s, ms_per_pt, num_batches = run_sdostream_with_timing(
                data, k=k, block_size=block_size, dimension=dimension
            )
            results.append((k, block_size, total_s, ms_per_pt, num_batches))
            print(f"{k:>6} | {block_size:>6} | {total_s:>10.4f} | {ms_per_pt:>8.3f} | {num_batches:>8}")

    print("-" * len(header))
    print()
    print("  total(s) = Gesamtzeit für alle learn()-Aufrufe")
    print("  ms/pt    = Millisekunden pro Datenpunkt")
    print("  batches  = Anzahl Batches (ceil(punkte / block_size))")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
