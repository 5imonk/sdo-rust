#!/usr/bin/env python3
"""
Tests für SDOstream (Sparse Data Observers Streaming – Outlier-Detection).
Dimension-Only-Init, grundlegendes Streaming, Observer-Updates, verschiedene Parameter.
"""

import sys
import os

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
from sklearn.preprocessing import MinMaxScaler


def test_dimension_only_initialization():
    """Initialisierung nur mit dimension (kein Warmup)."""
    print("=" * 60)
    print("Test 1: Dimension-Only Initialization")
    print("=" * 60)
    np.random.seed(42)
    k, x, dimension = 5, 3, 2
    t_fading = 10.0
    sdostream = SDOstream(k=k, x=x, t_fading=t_fading, dimension=dimension)
    assert sdostream.k == k and sdostream.x == x
    # dimension-only init generates k random points → data_points_processed == k
    assert sdostream.observer_count == k and sdostream.data_points_processed == k
    for i in range(k):
        observations, age, time, is_active, label = sdostream.get_observer_info(i)
        assert observations == 1.0 and age == 1.0 and is_active == True and label is None
    print("✓ Test 1 bestanden")
    return True


def test_basic_streaming_controlled():
    """Grundlegendes Streaming mit kontrollierten Punkten."""
    print("\n" + "=" * 60)
    print("Test 2: Basic Streaming (kontrolliert)")
    print("=" * 60)
    np.random.seed(42)
    init_data = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    sdostream = SDOstream(k=3, x=2, t_fading=1000.0, data=init_data)
    test_points = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.2, 0.2]]
    for point in test_points:
        point_2d = np.array([point], dtype=np.float64)
        sdostream.predict(point_2d)
        sdostream.learn(point_2d)
        assert np.isfinite(sdostream.predict(point_2d))
    # init uses len(init_data) points, then we learn len(test_points) more
    assert sdostream.data_points_processed == len(init_data) + len(test_points)
    print("✓ Test 2 bestanden")
    return True


def test_basic_streaming():
    """Grundlegendes Streaming mit MinMax-skalierte Daten."""
    print("\n" + "=" * 60)
    print("Test 3: Grundlegende Streaming-Funktionalität")
    print("=" * 60)
    np.random.seed(42)
    cluster1 = np.random.randn(5, 2) * 0.5 + np.array([2.0, 2.0])
    cluster2 = np.random.randn(5, 2) * 0.5 + np.array([8.0, 8.0])
    init_data = MinMaxScaler().fit_transform(np.vstack([cluster1, cluster2]).astype(np.float64))
    sdostream = SDOstream(k=10, x=5, t_fading=10.0, data=init_data)
    for point in ([2.0, 2.0], [8.0, 8.0], [2.5, 2.5], [8.5, 8.5], [15.0, 15.0]):
        point_2d = np.array([point], dtype=np.float64)
        sdostream.predict(point_2d)
        sdostream.learn(point_2d)
    print("✓ Test 3 bestanden")
    return True


def test_cluster_evolution():
    """Streaming über mehrere Phasen (Cluster-Evolution)."""
    print("\n" + "=" * 60)
    print("Test 4: Cluster-Evolution über Zeit")
    print("=" * 60)
    np.random.seed(42)
    init_data = MinMaxScaler().fit_transform(np.random.randn(10, 2).astype(np.float64))
    sdostream = SDOstream(k=3, x=2, t_fading=10.0, data=init_data)
    for pts in [
        np.random.randn(5, 2) * 0.5 + np.array([3.0, 3.0]),
        np.random.randn(5, 2) * 0.5 + np.array([10.0, 10.0]),
        np.random.randn(3, 2) * 0.5 + np.array([3.0, 3.0]),
    ]:
        for i in range(len(pts)):
            point = pts[i : i + 1, :].astype(np.float64)
            sdostream.learn(point)
    print("✓ Test 4 bestanden")
    return True


def test_observation_updates():
    """Observer-Updates bei Streaming (nahe vs. ferne Observer)."""
    print("\n" + "=" * 60)
    print("Test 5: Observation Update Verification")
    print("=" * 60)
    np.random.seed(42)
    init_data = np.array([
        [0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [5.0, 5.0], [6.0, 6.0],
    ], dtype=np.float64)
    sdostream = SDOstream(k=5, x=3, t_fading=1000.0, rho=0.4, data=init_data)
    initial_obs = [sdostream.get_observer_info(i)[0] for i in range(5)]
    for _ in range(20):
        point = np.random.randn(2) * 0.3 + np.array([1.0, 1.0])
        sdostream.learn(np.array([point], dtype=np.float64))
    all_info = sdostream.all_observer_info
    final_obs = [info[1] for info in all_info]
    observers_gained = sum(1 for i in range(3) if final_obs[i] > initial_obs[i])
    assert observers_gained >= 2
    assert np.mean(final_obs[:3]) > np.mean(final_obs[3:])
    print("✓ Test 5 bestanden")
    return True


def test_different_parameters():
    """Verschiedene Parameter (t_fading)."""
    print("\n" + "=" * 60)
    print("Test 6: Verschiedene Parameter")
    print("=" * 60)
    np.random.seed(42)
    init_data = MinMaxScaler().fit_transform(
        np.vstack([
            np.random.randn(8, 2) * 0.5 + np.array([2.0, 2.0]),
            np.random.randn(8, 2) * 0.5 + np.array([8.0, 8.0]),
        ]).astype(np.float64)
    )
    for t_fading in (5.0, 10.0, 20.0):
        sdostream = SDOstream(k=3, x=2, t_fading=t_fading, data=init_data)
        scores = [sdostream.predict(np.array([p]).reshape(1, -1)) for p in [[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]]
        assert all(np.isfinite(s) for s in scores)
    print("✓ Test 6 bestanden")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SDOstream – Tests (Streaming Outlier Detection)")
    parser.add_argument("--test", nargs="*", type=int, metavar="N",
                        help="Nur diese Testnummern ausführen (z.B. 1 2 4). Ohne Angabe: alle.")
    args = parser.parse_args()

    all_tests = [
        test_dimension_only_initialization,
        test_basic_streaming_controlled,
        test_basic_streaming,
        test_cluster_evolution,
        test_observation_updates,
        test_different_parameters,
    ]
    if args.test is not None and len(args.test) > 0:
        indices = [i - 1 for i in args.test if 1 <= i <= len(all_tests)]
        tests = [all_tests[i] for i in indices]
    else:
        tests = all_tests

    print("=" * 60)
    print("SDOstream (Streaming Outlier Detection) – Tests")
    print("=" * 60)
    passed = 0
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} FEHLGESCHLAGEN: {e}")
            import traceback
            traceback.print_exc()
    print("\n" + "=" * 60)
    print(f"Ergebnis: {passed}/{len(tests)} Tests bestanden")
    print("=" * 60)
    return passed == len(tests)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
