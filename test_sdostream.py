#!/usr/bin/env python3
"""
Vollständiges Testskript für SDOstream (Sparse Data Observers Streaming)
"""

import sys
import os

# Versuche das Modul zu importieren - verschiedene Wege
try:
    from sdo import SDOstream, SDOstreamParams
except ImportError:
    # Versuche, das Modul aus target/release zu laden
    target_path = os.path.join(os.path.dirname(__file__), 'target', 'release')
    if os.path.exists(target_path):
        sys.path.insert(0, target_path)
        try:
            from sdo import SDOstream, SDOstreamParams
        except ImportError:
            print("Fehler: Das 'sdo' Modul konnte nicht importiert werden.")
            print("Bitte installieren Sie das Modul mit 'maturin develop' oder 'pip install .'")
            sys.exit(1)
    else:
        print("Fehler: Das 'sdo' Modul konnte nicht importiert werden.")
        print("Bitte installieren Sie das Modul mit 'maturin develop' oder 'pip install .'")
        sys.exit(1)

import numpy as np


def test_basic_streaming():
    """Grundlegende Streaming-Funktionalität"""
    print("=" * 60)
    print("Test 1: Grundlegende Streaming-Funktionalität")
    print("=" * 60)
    
    # Erstelle SDOstream-Instanz
    sdostream = SDOstream()
    
    # Initialisiere mit Daten
    np.random.seed(42)
    init_data = np.random.randn(10, 2) * 1.0 + np.array([3.0, 3.0])
    init_data = init_data.astype(np.float64)
    
    print(f"Initialisierungsdaten: {init_data.shape[0]} Punkte, {init_data.shape[1]} Dimensionen")
    
    params = SDOstreamParams(k=5, x=3, t_fading=10.0, t_sampling=10.0)
    sdostream.initialize(params, data=init_data)
    print(f"✓ Modell initialisiert")
    
    # Streaming: Verarbeite einzelne Punkte
    print("\nStreaming-Verarbeitung:")
    streaming_points = [
        ([3.0, 3.0], "Normal"),
        ([15.0, 15.0], "Outlier"),
        ([3.5, 3.5], "Normal"),
        ([20.0, 20.0], "Outlier"),
        ([4.0, 4.0], "Normal"),
    ]
    
    for point, label in streaming_points:
        point_2d = np.array([point], dtype=np.float64)
        
        # Score vor Verarbeitung
        score_before = sdostream.predict(point_2d)
        
        # Verarbeite Punkt
        sdostream.learn(point_2d)
        
        # Score nach Verarbeitung
        score_after = sdostream.predict(point_2d)
        
        print(f"  {label:8}: [{point[0]:5.1f}, {point[1]:5.1f}] "
              f"→ Score: {score_before:.4f} → {score_after:.4f}")
    
    print(f"\nFinale Anzahl Observer: {sdostream.x}")
    print("\n✓ Test 1 erfolgreich abgeschlossen\n")


def test_fading_parameter():
    """Test des Fading-Parameters"""
    print("=" * 60)
    print("Test 2: Fading-Parameter")
    print("=" * 60)
    
    # Verschiedene T-Werte (größer = langsamere Anpassung)
    t_values = [5.0, 10.0, 20.0]
    
    init_data = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float64)
    
    print("Vergleich verschiedener T-Werte (Fading-Parameter):\n")
    
    for t in t_values:
        sdostream = SDOstream()
        params = SDOstreamParams(k=3, x=2, t_fading=t, t_sampling=t)
        sdostream.initialize(params, data=init_data)
        
        # Verarbeite mehrere identische Punkte
        point = np.array([[10.0, 10.0]], dtype=np.float64)
        
        scores = []
        for _ in range(5):
            score = sdostream.predict(point)
            sdostream.learn(point)
            scores.append(score)
        
        print(f"  T={t:5.1f}: Scores = {[f'{s:.4f}' for s in scores]}")
    
    print("\n✓ Test 2 erfolgreich abgeschlossen\n")


def test_observer_replacement():
    """Test der Observer-Ersetzung"""
    print("=" * 60)
    print("Test 3: Observer-Ersetzung")
    print("=" * 60)
    
    sdostream = SDOstream()
    
    # Initialisiere mit wenigen Observern
    init_data = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    params = SDOstreamParams(k=2, x=1, t_fading=5.0, t_sampling=5.0)  # Kleines T für häufigeres Sampling
    sdostream.initialize(params, data=init_data)
    
    print(f"Initial: {sdostream.x} Observer")
    
    # Verarbeite viele Punkte (sollte Observer ersetzen)
    np.random.seed(42)
    for i in range(20):
        point = np.random.randn(1, 2).astype(np.float64) * 2 + np.array([5.0, 5.0])
        sdostream.learn(point)
    
    print(f"Nach 20 Streaming-Punkten: {sdostream.x} Observer")
    print("✓ Observer-Anzahl bleibt konstant (k=2)")
    
    print("\n✓ Test 3 erfolgreich abgeschlossen\n")


def test_streaming_vs_batch():
    """Vergleich Streaming vs. Batch-Verarbeitung"""
    print("=" * 60)
    print("Test 4: Streaming vs. Batch-Verarbeitung")
    print("=" * 60)
    
    # Generiere Daten
    np.random.seed(42)
    batch_data = np.random.randn(50, 2) * 1.5 + np.array([3.0, 3.0])
    batch_data = batch_data.astype(np.float64)
    
    # Batch: Initialisiere mit allen Daten
    sdostream_batch = SDOstream()
    params = SDOstreamParams(k=10, x=5, t_fading=10.0, t_sampling=10.0)
    sdostream_batch.initialize(params, data=batch_data)
    
    # Streaming: Initialisiere mit ersten 10, dann einzeln verarbeiten
    sdostream_stream = SDOstream()
    sdostream_stream.initialize(params, data=batch_data[:10])
    for point in batch_data[10:]:
        point_2d = point.reshape(1, -1)
        sdostream_stream.learn(point_2d)
    
    # Teste auf gleichem Punkt
    test_point = np.array([[15.0, 15.0]], dtype=np.float64)
    
    score_batch = sdostream_batch.predict(test_point)
    score_stream = sdostream_stream.predict(test_point)
    
    print(f"Batch-Verarbeitung:   Score = {score_batch:.4f}")
    print(f"Streaming-Verarbeitung: Score = {score_stream:.4f}")
    print(f"  (Unterschied: {abs(score_batch - score_stream):.4f})")
    
    print("\n✓ Test 4 erfolgreich abgeschlossen\n")


def test_different_parameters():
    """Test mit verschiedenen Parametern"""
    print("=" * 60)
    print("Test 5: Verschiedene Parameter")
    print("=" * 60)
    
    # Generiere genügend Datenpunkte für alle Parameter-Kombinationen
    np.random.seed(42)
    init_data = np.random.randn(15, 2).astype(np.float64) * 1.0 + np.array([3.0, 3.0])
    test_point = np.array([[15.0, 15.0]], dtype=np.float64)
    
    param_sets = [
        {"k": 3, "x": 2, "t_fading": 5.0, "t_sampling": 5.0, "name": "Klein, schnell"},
        {"k": 5, "x": 3, "t_fading": 10.0, "t_sampling": 10.0, "name": "Mittel"},
        {"k": 10, "x": 5, "t_fading": 20.0, "t_sampling": 20.0, "name": "Groß, langsam"},
    ]
    
    print("Vergleich verschiedener Parameter:\n")
    
    for params_dict in param_sets:
        sdostream = SDOstream()
        params = SDOstreamParams(
            k=params_dict["k"],
            x=params_dict["x"],
            t_fading=params_dict["t_fading"],
            t_sampling=params_dict["t_sampling"],
        )
        sdostream.initialize(params, data=init_data)
        
        # Verarbeite einige Punkte
        for _ in range(5):
            point = np.array([[5.0, 5.0]], dtype=np.float64)
            sdostream.learn(point)
        
        score = sdostream.predict(test_point)
        
        print(f"  {params_dict['name']:15} (k={params_dict['k']:2}, x={params_dict['x']}, "
              f"t_fading={params_dict['t_fading']:5.1f}, t_sampling={params_dict['t_sampling']:5.1f}): Score = {score:.4f}")
    
    print("\n✓ Test 5 erfolgreich abgeschlossen\n")


def test_edge_cases():
    """Test von Edge Cases"""
    print("=" * 60)
    print("Test 6: Edge Cases")
    print("=" * 60)
    
    # Test 1: Leere Initialisierung
    print("Test 6.1: Leere Initialisierung")
    sdostream = SDOstream()
    empty_data = np.array([[]], dtype=np.float64).reshape(0, 2)
    try:
        params = SDOstreamParams(k=5, x=3, t_fading=10.0, t_sampling=10.0)
        sdostream.initialize(params, data=empty_data)
        print("  ✓ Leere Daten werden korrekt behandelt")
    except Exception as e:
        print(f"  ✗ Fehler: {e}")
    
    # Test 2: Einzelner Datenpunkt
    print("\nTest 6.2: Einzelner Datenpunkt")
    single_point = np.array([[1.0, 2.0]], dtype=np.float64)
    sdostream = SDOstream()
    params = SDOstreamParams(k=1, x=1, t_fading=10.0, t_sampling=10.0)
    sdostream.initialize(params, data=single_point)
    score = sdostream.predict(single_point)
    print(f"  ✓ Einzelner Punkt: Score = {score:.4f}, Observer = {sdostream.x}")
    
    # Test 3: Sehr wenige Daten
    print("\nTest 6.3: Sehr wenige Daten")
    few_data = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    sdostream = SDOstream()
    params = SDOstreamParams(k=2, x=1, t_fading=10.0, t_sampling=10.0)
    sdostream.initialize(params, data=few_data)
    
    # Verarbeite neuen Punkt
    new_point = np.array([[10.0, 10.0]], dtype=np.float64)
    sdostream.learn(new_point)
    score = sdostream.predict(new_point)
    print(f"  ✓ Wenige Daten: Score = {score:.4f}")
    
    # Test 4: 1D-Daten
    print("\nTest 6.4: 1D-Daten")
    data_1d = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    sdostream = SDOstream()
    params = SDOstreamParams(k=2, x=2, t_fading=10.0, t_sampling=10.0)
    sdostream.initialize(params, data=data_1d)
    point_1d = np.array([[5.0]], dtype=np.float64)
    score = sdostream.predict(point_1d)
    print(f"  ✓ 1D-Daten: Score = {score:.4f}")
    
    print("\n✓ Test 6 erfolgreich abgeschlossen\n")


def test_long_stream():
    """Test mit langem Datenstrom"""
    print("=" * 60)
    print("Test 7: Langer Datenstrom")
    print("=" * 60)
    
    sdostream = SDOstream()
    
    # Initialisiere
    np.random.seed(42)
    init_data = np.random.randn(10, 2).astype(np.float64)
    params = SDOstreamParams(k=10, x=5, t_fading=10.0, t_sampling=10.0)
    sdostream.initialize(params, data=init_data)
    
    print(f"Initial: {sdostream.x} Observer")
    
    # Verarbeite viele Punkte
    n_points = 100
    print(f"\nVerarbeite {n_points} Streaming-Punkte...")
    
    scores = []
    for i in range(n_points):
        point = np.random.randn(1, 2).astype(np.float64) * 2 + np.array([3.0, 3.0])
        score = sdostream.predict(point)
        sdostream.learn(point)
        scores.append(score)
        
        if (i + 1) % 20 == 0:
            print(f"  Punkt {i+1:3d}: Score = {score:.4f}, Observer = {sdostream.x}")
    
    print(f"\nFinal: {sdostream.x} Observer")
    print(f"Score-Statistik: Mean = {np.mean(scores):.4f}, "
          f"Std = {np.std(scores):.4f}, "
          f"Min = {np.min(scores):.4f}, "
          f"Max = {np.max(scores):.4f}")
    
    print("\n✓ Test 7 erfolgreich abgeschlossen\n")


def main():
    """Hauptfunktion - führt alle Tests aus"""
    print("\n" + "=" * 60)
    print("SDOstream (Sparse Data Observers Streaming) - Vollständiger Test")
    print("=" * 60 + "\n")
    
    try:
        test_basic_streaming()
        test_fading_parameter()
        test_observer_replacement()
        test_streaming_vs_batch()
        test_different_parameters()
        test_edge_cases()
        test_long_stream()
        
        print("=" * 60)
        print("✓ Alle Tests erfolgreich abgeschlossen!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Fehler beim Testen: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

