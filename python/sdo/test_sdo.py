#!/usr/bin/env python3
"""
Tests für SDO (Sparse Data Observers – statische Outlier-Detection).
"""

import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from sdo import SDO
except ImportError as e:
    print(f"Fehler: sdo-Modul nicht gefunden: {e}")
    print("Bitte im Projektroot 'maturin develop' ausführen.")
    sys.exit(1)

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def test_basic_usage():
    """Grundlegende Verwendung des SDO-Algorithmus."""
    print("=" * 60)
    print("Test 1: Grundlegende Verwendung")
    print("=" * 60)
    sdo = SDO(k=8, x=3, rho=0.2)
    data = np.array([
        [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0],
        [6.0, 7.0], [7.0, 8.0], [8.0, 9.0], [9.0, 10.0], [1.5, 2.5],
        [2.5, 3.5], [3.5, 4.5], [4.5, 5.5], [5.5, 6.5], [10.0, 11.0],
    ], dtype=np.float64)
    data = MinMaxScaler().fit_transform(data)
    sdo.learn(data)
    assert sdo.x >= 1
    observers = sdo.get_active_observers()
    assert observers.shape[0] >= 1
    test_points = np.array([[4.0, 5.0], [20.0, 21.0]], dtype=np.float64)
    scores_result = sdo.predict(test_points)
    # Batch gibt Liste zurück
    if isinstance(scores_result, list):
        scores = np.array(scores_result)
    else:
        scores = np.array([scores_result])
    for score in scores:
        assert np.isfinite(score)
    print("✓ Test 1 bestanden")
    return True


def test_larger_dataset():
    """Test mit größerem Datensatz."""
    print("\n" + "=" * 60)
    print("Test 2: Größerer Datensatz")
    print("=" * 60)
    np.random.seed(42)
    normal_data = np.random.randn(100, 2) * 2 + np.array([5.0, 5.0])
    outlier_data = np.array([[20.0, 20.0], [-10.0, -10.0], [15.0, -5.0], [0.0, 20.0], [-5.0, 15.0]])
    all_data = MinMaxScaler().fit_transform(np.vstack([normal_data, outlier_data]).astype(np.float64))
    sdo = SDO(k=20, x=5, rho=0.3)
    sdo.learn(all_data)
    scores_result = sdo.predict(all_data.astype(np.float64))
    if isinstance(scores_result, (list, np.ndarray)):
        scores = np.array(scores_result)
    else:
        scores = np.array([scores_result])
    assert np.mean(scores[:100]) < np.mean(scores[100:])
    print("✓ Test 2 bestanden")
    return True


def test_different_parameters():
    """Test mit verschiedenen Parametern."""
    print("\n" + "=" * 60)
    print("Test 3: Verschiedene Parameter")
    print("=" * 60)
    np.random.seed(123)
    all_data = MinMaxScaler().fit_transform(
        np.vstack([
            np.random.randn(50, 2) * 1.5 + np.array([3.0, 3.0]),
            np.array([[15.0, 15.0], [-5.0, -5.0]]),
        ]).astype(np.float64)
    )
    test_point = all_data[-1:].copy()
    for k, x, rho in [(10, 3, 0.2), (20, 5, 0.3), (30, 7, 0.4)]:
        sdo = SDO(k=k, x=x, rho=rho)
        sdo.learn(all_data)
        score = sdo.predict(test_point)
        assert np.isfinite(score)
    print("✓ Test 3 bestanden")
    return True


def test_edge_cases():
    """Edge Cases: leer, ein Punkt, wenige Daten, 1D."""
    print("\n" + "=" * 60)
    print("Test 4: Edge Cases")
    print("=" * 60)
    try:
        sdo = SDO(k=5, x=3, rho=0.2)
        sdo.learn(np.array([[]], dtype=np.float64).reshape(0, 2))
        print("  4.1 Leere Daten: ok")
    except Exception:
        pass
    single = np.array([[1.0, 2.0]], dtype=np.float64)
    sdo = SDO(k=1, x=1, rho=0.0)
    sdo.learn(single)
    assert np.isfinite(sdo.predict(single))
    few = MinMaxScaler().fit_transform(np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float64))
    sdo = SDO(k=2, x=1, rho=0.1)
    sdo.learn(few)
    pt = MinMaxScaler().fit(few).transform(np.array([[10.0, 10.0]], dtype=np.float64))
    assert np.isfinite(sdo.predict(pt))
    data_1d = MinMaxScaler().fit_transform(np.array([[1.0], [2.0], [3.0], [10.0]], dtype=np.float64))
    sdo = SDO(k=2, x=2, rho=0.2)
    sdo.learn(data_1d)
    assert np.isfinite(sdo.predict(MinMaxScaler().fit(data_1d).transform(np.array([[5.0]], dtype=np.float64))))
    print("✓ Test 4 bestanden")
    return True


def test_performance():
    """Kurzer Performance-Check."""
    print("\n" + "=" * 60)
    print("Test 5: Performance")
    print("=" * 60)
    import time
    np.random.seed(42)
    data = np.random.randn(500, 2).astype(np.float64)
    start = time.time()
    sdo = SDO(k=50, x=5, rho=0.2)
    sdo.learn(data)
    t = time.time() - start
    assert t < 30.0
    print(f"  500 Punkte: learn = {t*1000:.1f} ms")
    print("✓ Test 5 bestanden")
    return True


def test_3d_data():
    """Test mit 3D-Daten."""
    print("\n" + "=" * 60)
    print("Test 6: 3D-Daten")
    print("=" * 60)
    np.random.seed(42)
    normal_data = np.random.randn(50, 3) * 1.0 + np.array([0.0, 0.0, 0.0])
    outlier_data = np.array([[10.0, 10.0, 10.0], [-10.0, -10.0, -10.0]])
    all_data = np.vstack([normal_data, outlier_data]).astype(np.float64)
    sdo = SDO(k=15, x=5, rho=0.2)
    sdo.learn(all_data)
    test_points = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]], dtype=np.float64)
    scores_result = sdo.predict(test_points)
    # Batch gibt Liste zurück
    if isinstance(scores_result, list):
        scores = np.array(scores_result)
    else:
        scores = np.array([scores_result])
    for score in scores:
        assert np.isfinite(score)
    print("✓ Test 6 bestanden")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SDO – Tests (statische Outlier-Detection)")
    parser.add_argument("--test", nargs="*", type=int, metavar="N",
                        help="Nur diese Testnummern ausführen (z.B. 1 2 4). Ohne Angabe: alle.")
    args = parser.parse_args()

    all_tests = [
        test_basic_usage,
        test_larger_dataset,
        test_different_parameters,
        test_edge_cases,
        test_performance,
        test_3d_data,
    ]
    if args.test is not None and len(args.test) > 0:
        indices = [i - 1 for i in args.test if 1 <= i <= len(all_tests)]
        tests = [all_tests[i] for i in indices]
    else:
        tests = all_tests

    print("=" * 60)
    print("SDO (Sparse Data Observers) – Tests")
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
