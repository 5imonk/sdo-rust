#!/usr/bin/env python3
"""
Tests für SDOstreamclust (Sparse Data Observers Streaming Clustering).
Integration, grundlegendes Streaming-Clustering, Cluster-Evolution.
SDOstream-Tests: python/sdostream/test_sdostream.py
"""

import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _THIS_DIR)

try:
    from sdo import SDOstreamclust
except ImportError as e:
    print(f"Fehler: sdo-Modul nicht gefunden: {e}")
    print("Bitte im Projektroot 'maturin develop' ausführen.")
    sys.exit(1)

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def test_sdostreamclust_integration():
    """SDOstreamclust: Integration mit Clustering."""
    print("=" * 60)
    print("Test 1: SDOstreamclust Integration")
    print("=" * 60)
    np.random.seed(42)
    sdostreamclust = SDOstreamclust(
        k=10, x=3, t_fading=20.0,
        chi_min=1, chi_prop=0.1, zeta=0.5, min_cluster_size=2,
        dimension=2,
    )
    cluster1_points = np.random.randn(5, 2) * 0.5 + np.array([0, 0])
    cluster2_points = np.random.randn(5, 2) * 0.5 + np.array([3, 3])
    for point in list(cluster1_points) + list(cluster2_points):
        label, score = sdostreamclust.learn(point.reshape(1, -1))
        assert np.isfinite(score)
    label1, score1 = sdostreamclust.predict(np.array([[0.1, 0.1]], dtype=np.float64))
    label2, score2 = sdostreamclust.predict(np.array([[3.1, 3.1]], dtype=np.float64))
    assert (label1 >= 0 or label1 == -1) and (label2 >= 0 or label2 == -1)
    assert np.isfinite(score1) and np.isfinite(score2)
    print("✓ Test 1 bestanden")
    return True


def test_basic_streaming_clustering():
    """SDOstreamclust: Grundlegendes Streaming-Clustering."""
    print("\n" + "=" * 60)
    print("Test 2: Grundlegende Streaming-Clustering")
    print("=" * 60)
    np.random.seed(42)
    cluster1 = np.random.randn(5, 2) * 0.5 + np.array([2.0, 2.0])
    cluster2 = np.random.randn(5, 2) * 0.5 + np.array([8.0, 8.0])
    init_data = MinMaxScaler().fit_transform(np.vstack([cluster1, cluster2]).astype(np.float64))
    model = SDOstreamclust(
        k=10, x=3, t_fading=10.0,
        chi_min=1, chi_prop=0.05, zeta=0.5, min_cluster_size=2,
        data=init_data,
    )
    for point in ([2.0, 2.0], [8.0, 8.0], [2.5, 2.5], [8.5, 8.5], [15.0, 15.0]):
        point_2d = np.array([point], dtype=np.float64)
        model.predict(point_2d)
        model.learn(point_2d)
    print("✓ Test 2 bestanden")
    return True


def test_cluster_evolution():
    """SDOstreamclust: Cluster-Evolution über Zeit."""
    print("\n" + "=" * 60)
    print("Test 3: Cluster-Evolution (SDOstreamclust)")
    print("=" * 60)
    np.random.seed(42)
    scaler = MinMaxScaler()
    init_data = scaler.fit_transform(np.random.randn(10, 2).astype(np.float64))
    model = SDOstreamclust(
        k=3, x=2, t_fading=10.0,
        chi_min=1, chi_prop=0.05, zeta=0.5, min_cluster_size=2,
        data=init_data,
    )
    for pts in [
        np.random.randn(5, 2) * 0.5 + np.array([3.0, 3.0]),
        np.random.randn(5, 2) * 0.5 + np.array([10.0, 10.0]),
        np.random.randn(3, 2) * 0.5 + np.array([3.0, 3.0]),
    ]:
        pts = scaler.transform(pts)
        for i in range(len(pts)):
            model.learn(pts[i : i + 1, :])
    print("✓ Test 3 bestanden")
    return True


def main():
    print("=" * 60)
    print("SDOstreamclust – Tests")
    print("=" * 60)
    tests = [
        test_sdostreamclust_integration,
        test_basic_streaming_clustering,
        test_cluster_evolution,
    ]
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
