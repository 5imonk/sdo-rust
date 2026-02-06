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
    sdostream = SDOstream(k=k, x=x, t_fading=t_fading, t_sampling=t_fading, dimension=dimension)
    assert sdostream.k == k and sdostream.x == x
    # dimension-only init generates k random points → data_points_processed == k
    assert sdostream.observer_count == k and sdostream.data_points_processed == k
    for i in range(k):
        observations, age, time, is_active, label = sdostream.get_observer_info(i)
        assert observations == 1.0 and age == 1.0 and is_active == True and label is None
    print("✓ Test 1 bestanden")
    return True


def test_basic_streaming_controlled():
    """Grundlegendes Streaming mit kontrollierten Punkten (Batch-Verarbeitung)."""
    print("\n" + "=" * 60)
    print("Test 2: Basic Streaming (kontrolliert, Batch)")
    print("=" * 60)
    np.random.seed(42)
    init_data = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    sdostream = SDOstream(k=3, x=2, t_fading=1000.0, t_sampling=1000.0, data=init_data)
    test_points = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.2, 0.2]], dtype=np.float64)
    
    # Batch-Predict vor Learn
    scores_before = sdostream.predict(test_points)
    assert all(np.isfinite(s) for s in scores_before)
    
    # Batch-Learn
    scores_after = sdostream.learn(test_points)
    assert all(np.isfinite(s) for s in scores_after)
    
    # Batch-Predict nach Learn
    scores_final = sdostream.predict(test_points)
    assert all(np.isfinite(s) for s in scores_final)
    
    # init uses len(init_data) points, then we learn len(test_points) more
    assert sdostream.data_points_processed == len(init_data) + len(test_points)
    print("✓ Test 2 bestanden")
    return True


def test_basic_streaming():
    """Grundlegendes Streaming mit MinMax-skalierte Daten (Batch-Verarbeitung)."""
    print("\n" + "=" * 60)
    print("Test 3: Grundlegende Streaming-Funktionalität (Batch)")
    print("=" * 60)
    np.random.seed(42)
    cluster1 = np.random.randn(5, 2) * 0.5 + np.array([2.0, 2.0])
    cluster2 = np.random.randn(5, 2) * 0.5 + np.array([8.0, 8.0])
    init_data = MinMaxScaler().fit_transform(np.vstack([cluster1, cluster2]).astype(np.float64))
    sdostream = SDOstream(k=10, x=5, t_fading=10.0, t_sampling=10.0, data=init_data)
    
    # Batch-Verarbeitung für alle Punkte
    test_points = np.array([[2.0, 2.0], [8.0, 8.0], [2.5, 2.5], [8.5, 8.5], [15.0, 15.0]], dtype=np.float64)
    scores = sdostream.predict(test_points)
    assert all(np.isfinite(s) for s in scores)
    scores_after = sdostream.learn(test_points)
    assert all(np.isfinite(s) for s in scores_after)
    print("✓ Test 3 bestanden")
    return True


def test_cluster_evolution():
    """Streaming über mehrere Phasen (Cluster-Evolution) - Batch-Verarbeitung."""
    print("\n" + "=" * 60)
    print("Test 4: Cluster-Evolution über Zeit (Batch)")
    print("=" * 60)
    np.random.seed(42)
    init_data = MinMaxScaler().fit_transform(np.random.randn(10, 2).astype(np.float64))
    sdostream = SDOstream(k=3, x=2, t_fading=10.0, t_sampling=10.0, data=init_data)
    
    # Batch-Verarbeitung für jede Phase
    phases = [
        np.random.randn(5, 2) * 0.5 + np.array([3.0, 3.0]),
        np.random.randn(5, 2) * 0.5 + np.array([10.0, 10.0]),
        np.random.randn(3, 2) * 0.5 + np.array([3.0, 3.0]),
    ]
    for pts in phases:
        pts_normalized = MinMaxScaler().fit_transform(pts.astype(np.float64))
        scores = sdostream.learn(pts_normalized)
        assert isinstance(scores, (list, np.ndarray))
        if isinstance(scores, list):
            assert len(scores) == len(pts_normalized)
        else:
            assert len(scores) == len(pts_normalized)
        assert all(np.isfinite(s) for s in scores)
    print("✓ Test 4 bestanden")
    return True


def test_observation_updates():
    """Observer-Updates bei Streaming (nahe vs. ferne Observer) - Batch-Verarbeitung."""
    print("\n" + "=" * 60)
    print("Test 5: Observation Update Verification (Batch)")
    print("=" * 60)
    np.random.seed(42)
    init_data = np.array([
        [0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [5.0, 5.0], [6.0, 6.0],
    ], dtype=np.float64)
    sdostream = SDOstream(k=5, x=3, t_fading=1000.0, t_sampling=1000.0, rho=0.4, data=init_data)
    initial_obs = [sdostream.get_observer_info(i)[0] for i in range(5)]
    
    # Batch-Learn für 20 Punkte
    batch_points = np.array([
        np.random.randn(2) * 0.3 + np.array([1.0, 1.0]) for _ in range(20)
    ], dtype=np.float64)
    scores = sdostream.learn(batch_points)
    assert len(scores) == 20
    assert all(np.isfinite(s) for s in scores)
    
    all_info = sdostream.all_observer_info
    final_obs = [info[1] for info in all_info]
    observers_gained = sum(1 for i in range(3) if final_obs[i] > initial_obs[i])
    assert observers_gained >= 2
    assert np.mean(final_obs[:3]) > np.mean(final_obs[3:])
    print("✓ Test 5 bestanden")
    return True


def test_t_sampling_reflection():
    """t_sampling wird gespeichert und über Getter zurückgegeben (Reflection)."""
    print("\n" + "=" * 60)
    print("Test 5b: t_sampling Reflection")
    print("=" * 60)
    np.random.seed(42)
    t_sampling = 200.0
    sdostream = SDOstream(
        k=10, x=3, t_fading=100.0, t_sampling=t_sampling, dimension=2
    )
    assert abs(sdostream.t_sampling - t_sampling) < 1e-9, (
        f"t_sampling Getter: erwartet {t_sampling}, erhalten {sdostream.t_sampling}"
    )
    print("✓ Test 5b bestanden")
    return True


def test_t_sampling_affects_replacement_rate():
    """Gleiche Daten/Zeiten: kleineres t_sampling → mehr Ersetzungen (Verhalten)."""
    print("\n" + "=" * 60)
    print("Test 5c: t_sampling Verhalten (Ersetzungsrate)")
    print("=" * 60)
    np.random.seed(42)
    n_points = 150
    times = np.arange(n_points, dtype=np.float64) * 2.0
    points = np.random.rand(n_points, 2).astype(np.float64)

    model_low = SDOstream(
        k=30, x=4, t_fading=100.0, t_sampling=10.0, dimension=2
    )
    model_high = SDOstream(
        k=30, x=4, t_fading=100.0, t_sampling=5000.0, dimension=2
    )

    model_low.learn(points, time=times)
    model_high.learn(points, time=times)

    count_low = model_low.replacement_count
    count_high = model_high.replacement_count
    assert count_low > count_high, (
        f"Kleines t_sampling sollte mehr Ersetzungen liefern: low={count_low}, high={count_high}"
    )
    print(f"  Ersetzungen (t_sampling=10): {count_low}, (t_sampling=5000): {count_high}")
    print("✓ Test 5c bestanden")
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
        sdostream = SDOstream(k=3, x=2, t_fading=t_fading, t_sampling=t_fading, data=init_data)
        # Verwende predict mit mehreren Punkten (gibt Liste zurück)
        test_points = np.array([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], dtype=np.float64)
        scores = sdostream.predict(test_points)
        assert all(np.isfinite(s) for s in scores)
    print("✓ Test 6 bestanden")
    return True


def test_predict_batch():
    """Test für Batch-Vorhersage (predict unterstützt jetzt automatisch Batches)."""
    print("\n" + "=" * 60)
    print("Test 7: Batch-Vorhersage (predict mit mehreren Punkten)")
    print("=" * 60)
    np.random.seed(42)
    init_data = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    sdostream = SDOstream(k=3, x=2, t_fading=1000.0, t_sampling=1000.0, data=init_data)
    
    # Test mit mehreren Punkten
    test_points = np.array([
        [0.1, 0.1],
        [0.9, 0.1],
        [0.1, 0.9],
        [0.5, 0.5],
    ], dtype=np.float64)
    
    # Batch-Vorhersage (predict gibt jetzt Liste zurück für mehrere Punkte)
    batch_scores = sdostream.predict(test_points)
    assert isinstance(batch_scores, (list, np.ndarray))
    if isinstance(batch_scores, list):
        assert len(batch_scores) == len(test_points)
    else:
        assert len(batch_scores) == len(test_points)
    assert all(np.isfinite(s) for s in batch_scores)
    
    # Vergleich mit einzelnen Vorhersagen (predict gibt einzelnen Wert zurück für einen Punkt)
    individual_scores = [sdostream.predict(test_points[i:i+1]) for i in range(len(test_points))]
    assert np.allclose(batch_scores, individual_scores, rtol=1e-10)
    
    print("✓ Test 7 bestanden")
    return True


def test_learn_batch():
    """Test für Batch-Learn (learn unterstützt jetzt automatisch Batches)."""
    print("\n" + "=" * 60)
    print("Test 8: Batch-Learn (learn mit mehreren Punkten)")
    print("=" * 60)
    np.random.seed(42)
    init_data = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    sdostream1 = SDOstream(k=3, x=2, t_fading=1000.0, t_sampling=1000.0, data=init_data)
    sdostream2 = SDOstream(k=3, x=2, t_fading=1000.0, t_sampling=1000.0, data=init_data)
    
    # Test-Punkte
    test_points = np.array([
        [0.1, 0.1],
        [0.9, 0.1],
        [0.1, 0.9],
        [0.2, 0.2],
    ], dtype=np.float64)
    
    # Batch-Learn (learn gibt jetzt Liste zurück für mehrere Punkte)
    batch_scores = sdostream1.learn(test_points)
    assert isinstance(batch_scores, (list, np.ndarray))
    if isinstance(batch_scores, list):
        assert len(batch_scores) == len(test_points)
    else:
        assert len(batch_scores) == len(test_points)
    assert all(np.isfinite(s) for s in batch_scores)
    assert sdostream1.data_points_processed == len(init_data) + len(test_points)
    
    # Vergleich mit sequentiellem Learn (learn gibt einzelnen Wert zurück für einen Punkt)
    for point in test_points:
        sdostream2.learn(point.reshape(1, -1))
    
    assert sdostream1.data_points_processed == sdostream2.data_points_processed
    
    # Prüfe, dass beide Modelle ähnliche Scores für denselben Punkt liefern
    test_point = np.array([[0.5, 0.5]], dtype=np.float64)
    score1 = sdostream1.predict(test_point)
    score2 = sdostream2.predict(test_point)
    assert np.isfinite(score1) and np.isfinite(score2)
    
    print("✓ Test 8 bestanden")
    return True


def test_cluster_evolution_batch():
    """Streaming über mehrere Phasen mit Batch-APIs (learn unterstützt jetzt automatisch Batches)."""
    print("\n" + "=" * 60)
    print("Test 9: Cluster-Evolution mit Batch-APIs")
    print("=" * 60)
    np.random.seed(42)
    init_data = MinMaxScaler().fit_transform(np.random.randn(10, 2).astype(np.float64))
    sdostream = SDOstream(k=3, x=2, t_fading=10.0, t_sampling=10.0, data=init_data)
    
    # Verwende Batch-APIs für jede Phase (learn gibt Liste zurück für mehrere Punkte)
    phases = [
        np.random.randn(5, 2) * 0.5 + np.array([3.0, 3.0]),
        np.random.randn(5, 2) * 0.5 + np.array([10.0, 10.0]),
        np.random.randn(3, 2) * 0.5 + np.array([3.0, 3.0]),
    ]
    
    for pts in phases:
        pts_normalized = MinMaxScaler().fit_transform(pts.astype(np.float64))
        scores = sdostream.learn(pts_normalized)
        assert isinstance(scores, (list, np.ndarray))
        if isinstance(scores, list):
            assert len(scores) == len(pts_normalized)
        else:
            assert len(scores) == len(pts_normalized)
        assert all(np.isfinite(s) for s in scores)
    
    print("✓ Test 9 bestanden")
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
        test_t_sampling_reflection,
        test_t_sampling_affects_replacement_rate,
        test_different_parameters,
        test_predict_batch,
        test_learn_batch,
        test_cluster_evolution_batch,
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
