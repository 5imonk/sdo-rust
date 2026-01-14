#!/usr/bin/env python3
"""
Vollständiges Testskript für SDOstreamclust (Sparse Data Observers Streaming Clustering)
"""

import sys
import os

# Versuche das Modul zu importieren - verschiedene Wege
try:
    from sdo import SDOstreamclust, SDOstreamclustParams
except ImportError:
    # Versuche, das Modul aus target/release zu laden
    target_path = os.path.join(os.path.dirname(__file__), 'target', 'release')
    if os.path.exists(target_path):
        sys.path.insert(0, target_path)
        try:
            from sdo import SDOstreamclust, SDOstreamclustParams
        except ImportError:
            print("Fehler: Das 'sdo' Modul konnte nicht importiert werden.")
            print("Bitte installieren Sie das Modul mit 'maturin develop' oder 'pip install .'")
            sys.exit(1)
    else:
        print("Fehler: Das 'sdo' Modul konnte nicht importiert werden.")
        print("Bitte installieren Sie das Modul mit 'maturin develop' oder 'pip install .'")
        sys.exit(1)

import numpy as np


def test_basic_streaming_clustering():
    """Grundlegende Streaming-Clustering-Funktionalität"""
    print("=" * 60)
    print("Test 1: Grundlegende Streaming-Clustering-Funktionalität")
    print("=" * 60)
    
    # Erstelle SDOstreamclust-Instanz
    sdostreamclust = SDOstreamclust()
    
    # Initialisiere mit Daten
    np.random.seed(42)
    # Erstelle zwei Cluster
    cluster1 = np.random.randn(5, 2) * 0.5 + np.array([2.0, 2.0])
    cluster2 = np.random.randn(5, 2) * 0.5 + np.array([8.0, 8.0])
    init_data = np.vstack([cluster1, cluster2]).astype(np.float64)
    
    print(f"Initialisierungsdaten: {init_data.shape[0]} Punkte, {init_data.shape[1]} Dimensionen")
    
    params = SDOstreamclustParams(
        k=10, x=3, t_fading=10.0, t_sampling=10.0,
        chi=4, zeta=0.5, min_cluster_size=2
    )
    sdostreamclust.initialize(params, data=init_data)
    print(f"✓ Modell initialisiert")
    
    # Streaming: Verarbeite einzelne Punkte
    print("\nStreaming-Verarbeitung mit Clustering:")
    streaming_points = [
        ([2.0, 2.0], "Cluster 1"),
        ([8.0, 8.0], "Cluster 2"),
        ([2.5, 2.5], "Cluster 1"),
        ([8.5, 8.5], "Cluster 2"),
        ([15.0, 15.0], "Outlier/Neuer Cluster"),
    ]
    
    for point, label in streaming_points:
        point_2d = np.array([point], dtype=np.float64)
        
        # Label vor Verarbeitung
        label_before = sdostreamclust.predict(point_2d)
        
        # Verarbeite Punkt
        sdostreamclust.learn(point_2d)
        
        # Label nach Verarbeitung
        label_after = sdostreamclust.predict(point_2d)
        
        print(f"  {label:20}: [{point[0]:5.1f}, {point[1]:5.1f}] "
              f"→ Label: {label_before} → {label_after}")
    
    print(f"\nFinale Anzahl Observer: {sdostreamclust.x}")
    print("\n✓ Test 1 erfolgreich abgeschlossen\n")


def test_cluster_evolution():
    """Test der Cluster-Evolution über Zeit"""
    print("=" * 60)
    print("Test 2: Cluster-Evolution über Zeit")
    print("=" * 60)
    
    sdostreamclust = SDOstreamclust()
    
    # Initialisiere mit einem Cluster
    np.random.seed(42)
    init_data = np.random.randn(10, 2) * 0.5 + np.array([3.0, 3.0])
    init_data = init_data.astype(np.float64)
    
    params = SDOstreamclustParams(
        k=10, x=3, t_fading=10.0, t_sampling=10.0,
        chi=4, zeta=0.5, min_cluster_size=2
    )
    sdostreamclust.initialize(params, data=init_data)
    
    print("Verarbeite Punkte aus verschiedenen Clustern über Zeit:\n")
    
    # Phase 1: Cluster 1
    print("Phase 1: Cluster 1 (Punkte um [3, 3])")
    for i in range(5):
        point = np.random.randn(1, 2) * 0.5 + np.array([[3.0, 3.0]])
        point = point.astype(np.float64)
        label = sdostreamclust.predict(point)
        sdostreamclust.learn(point)
        print(f"  Punkt {i+1}: Label = {label}")
    
    # Phase 2: Cluster 2 erscheint
    print("\nPhase 2: Cluster 2 erscheint (Punkte um [10, 10])")
    for i in range(5):
        point = np.random.randn(1, 2) * 0.5 + np.array([[10.0, 10.0]])
        point = point.astype(np.float64)
        label = sdostreamclust.predict(point)
        sdostreamclust.learn(point)
        print(f"  Punkt {i+1}: Label = {label}")
    
    # Phase 3: Gemischt
    print("\nPhase 3: Gemischte Punkte")
    for i in range(3):
        if i % 2 == 0:
            point = np.random.randn(1, 2) * 0.5 + np.array([[3.0, 3.0]])
        else:
            point = np.random.randn(1, 2) * 0.5 + np.array([[10.0, 10.0]])
        point = point.astype(np.float64)
        label = sdostreamclust.predict(point)
        sdostreamclust.learn(point)
        print(f"  Punkt {i+1}: Label = {label}")
    
    print("\n✓ Test 2 erfolgreich abgeschlossen\n")


def test_fading_and_cluster_observations():
    """Test des Fading-Mechanismus für Cluster-Beobachtungen"""
    print("=" * 60)
    print("Test 3: Fading und Cluster-Beobachtungen")
    print("=" * 60)
    
    # Verschiedene T-Werte
    t_values = [5.0, 10.0, 20.0]
    
    # Generiere genügend Datenpunkte für k=5
    np.random.seed(42)
    cluster1 = np.random.randn(3, 2) * 0.3 + np.array([2.0, 2.0])
    cluster2 = np.random.randn(3, 2) * 0.3 + np.array([8.0, 8.0])
    init_data = np.vstack([cluster1, cluster2]).astype(np.float64)
    
    print("Vergleich verschiedener T-Werte (Fading-Parameter):\n")
    
    for t in t_values:
        sdostreamclust = SDOstreamclust()
        params = SDOstreamclustParams(
            k=5, x=2, t_fading=t, t_sampling=t,
            chi=2, zeta=0.5, min_cluster_size=1
        )
        sdostreamclust.initialize(params, data=init_data)
        
        # Verarbeite mehrere Punkte aus Cluster 1
        point = np.array([[2.0, 2.0]], dtype=np.float64)
        
        labels = []
        for _ in range(5):
            label = sdostreamclust.predict(point)
            sdostreamclust.learn(point)
            labels.append(label)
        
        print(f"  T={t:5.1f}: Labels = {labels}")
    
    print("\n✓ Test 3 erfolgreich abgeschlossen\n")


def test_cluster_labeling():
    """Test der Cluster-Labeling-Logik"""
    print("=" * 60)
    print("Test 4: Cluster-Labeling")
    print("=" * 60)
    
    sdostreamclust = SDOstreamclust()
    
    # Initialisiere mit zwei getrennten Clustern
    np.random.seed(42)
    cluster1 = np.random.randn(8, 2) * 0.3 + np.array([2.0, 2.0])
    cluster2 = np.random.randn(8, 2) * 0.3 + np.array([8.0, 8.0])
    init_data = np.vstack([cluster1, cluster2]).astype(np.float64)
    
    params = SDOstreamclustParams(
        k=10, x=3, t_fading=10.0, t_sampling=10.0,
        chi=4, zeta=0.5, min_cluster_size=2
    )
    sdostreamclust.initialize(params, data=init_data)
    
    print("Initiale Cluster-Labels nach Verarbeitung von Punkten:\n")
    
    # Verarbeite Punkte aus beiden Clustern abwechselnd
    for i in range(10):
        if i % 2 == 0:
            point = np.random.randn(1, 2) * 0.3 + np.array([[2.0, 2.0]])
        else:
            point = np.random.randn(1, 2) * 0.3 + np.array([[8.0, 8.0]])
        point = point.astype(np.float64)
        
        label = sdostreamclust.predict(point)
        sdostreamclust.learn(point)
        
        cluster_id = "Cluster 1" if i % 2 == 0 else "Cluster 2"
        print(f"  Punkt {i+1:2d} ({cluster_id:10}): Label = {label}")
    
    print("\n✓ Test 4 erfolgreich abgeschlossen\n")


def test_different_parameters():
    """Test mit verschiedenen Parametern"""
    print("=" * 60)
    print("Test 5: Verschiedene Parameter")
    print("=" * 60)
    
    # Generiere genügend Datenpunkte
    np.random.seed(42)
    cluster1 = np.random.randn(8, 2) * 0.5 + np.array([2.0, 2.0])
    cluster2 = np.random.randn(8, 2) * 0.5 + np.array([8.0, 8.0])
    init_data = np.vstack([cluster1, cluster2]).astype(np.float64)
    test_point = np.array([[5.0, 5.0]], dtype=np.float64)
    
    param_sets = [
        {"chi": 2, "zeta": 0.3, "min_cluster_size": 1, "name": "Konservativ"},
        {"chi": 4, "zeta": 0.5, "min_cluster_size": 2, "name": "Moderat"},
        {"chi": 6, "zeta": 0.7, "min_cluster_size": 3, "name": "Aggressiv"},
    ]
    
    print("Vergleich verschiedener Parameter:\n")
    
    for params_dict in param_sets:
        sdostreamclust = SDOstreamclust()
        params = SDOstreamclustParams(
            k=10,
            x=3,
            t_fading=10.0,
            t_sampling=10.0,
            chi=params_dict["chi"],
            zeta=params_dict["zeta"],
            min_cluster_size=params_dict["min_cluster_size"],
        )
        sdostreamclust.initialize(params, data=init_data)
        
        # Verarbeite einige Punkte
        for _ in range(5):
            point = np.array([[3.0, 3.0]], dtype=np.float64)
            sdostreamclust.learn(point)
        
        label = sdostreamclust.predict(test_point)
        
        print(f"  {params_dict['name']:12} (χ={params_dict['chi']}, ζ={params_dict['zeta']:.1f}, "
              f"e={params_dict['min_cluster_size']}): Label = {label}")
    
    print("\n✓ Test 5 erfolgreich abgeschlossen\n")


def test_edge_cases():
    """Test von Edge Cases"""
    print("=" * 60)
    print("Test 6: Edge Cases")
    print("=" * 60)
    
    # Test 1: Sehr wenige Daten
    print("Test 6.1: Sehr wenige Daten")
    few_data = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    sdostreamclust = SDOstreamclust()
    try:
        params = SDOstreamclustParams(
            k=2, x=1, t_fading=10.0, t_sampling=10.0,
            chi=1, zeta=0.5, min_cluster_size=1
        )
        sdostreamclust.initialize(params, data=few_data)
        point = np.array([[1.5, 1.5]], dtype=np.float64)
        label = sdostreamclust.predict(point)
        sdostreamclust.learn(point)
        print(f"  ✓ Wenige Daten: Label = {label}")
    except Exception as e:
        print(f"  ✗ Fehler: {e}")
    
    # Test 2: Einzelner Datenpunkt
    print("\nTest 6.2: Einzelner Datenpunkt")
    single_point = np.array([[1.0, 2.0]], dtype=np.float64)
    sdostreamclust = SDOstreamclust()
    params = SDOstreamclustParams(
        k=1, x=1, t_fading=10.0, t_sampling=10.0,
        chi=1, zeta=0.5, min_cluster_size=1
    )
    sdostreamclust.initialize(params, data=single_point)
    label = sdostreamclust.predict(single_point)
    print(f"  ✓ Einzelner Punkt: Label = {label}, Observer = {sdostreamclust.x}")
    
    # Test 3: Hoher min_cluster_size
    print("\nTest 6.3: Hoher min_cluster_size")
    np.random.seed(42)
    data = np.random.randn(10, 2).astype(np.float64)
    sdostreamclust = SDOstreamclust()
    params = SDOstreamclustParams(
        k=5, x=2, t_fading=10.0, t_sampling=10.0,
        chi=2, zeta=0.5, min_cluster_size=10
    )
    sdostreamclust.initialize(params, data=data)
    point = np.array([[5.0, 5.0]], dtype=np.float64)
    label = sdostreamclust.predict(point)
    sdostreamclust.learn(point)
    print(f"  ✓ Hoher min_cluster_size: Label = {label}")
    
    print("\n✓ Test 6 erfolgreich abgeschlossen\n")


def test_long_stream():
    """Test mit langem Datenstrom"""
    print("=" * 60)
    print("Test 7: Langer Datenstrom")
    print("=" * 60)
    
    sdostreamclust = SDOstreamclust()
    
    # Initialisiere
    np.random.seed(42)
    init_data = np.random.randn(10, 2).astype(np.float64)
    params = SDOstreamclustParams(
        k=10, x=5, t_fading=10.0, t_sampling=10.0,
        chi=4, zeta=0.5, min_cluster_size=2
    )
    sdostreamclust.initialize(params, data=init_data)
    
    print(f"Initial: {sdostreamclust.x} Observer")
    
    # Verarbeite viele Punkte aus verschiedenen Clustern
    n_points = 50
    print(f"\nVerarbeite {n_points} Streaming-Punkte...")
    
    labels = []
    for i in range(n_points):
        # Abwechselnd aus verschiedenen Clustern
        if i % 3 == 0:
            point = np.random.randn(1, 2) * 0.5 + np.array([[2.0, 2.0]])
        elif i % 3 == 1:
            point = np.random.randn(1, 2) * 0.5 + np.array([[8.0, 8.0]])
        else:
            point = np.random.randn(1, 2) * 0.5 + np.array([[5.0, 5.0]])
        point = point.astype(np.float64)
        
        label = sdostreamclust.predict(point)
        sdostreamclust.learn(point)
        labels.append(label)
        
        if (i + 1) % 10 == 0:
            unique_labels = len(set([l for l in labels[-10:] if l >= 0]))
            print(f"  Punkt {i+1:3d}: Label = {label}, "
                  f"Eindeutige Labels (letzte 10): {unique_labels}")
    
    print(f"\nFinal: {sdostreamclust.x} Observer")
    unique_labels_all = len(set([l for l in labels if l >= 0]))
    print(f"Eindeutige Labels (gesamt): {unique_labels_all}")
    
    print("\n✓ Test 7 erfolgreich abgeschlossen\n")


def main():
    """Hauptfunktion - führt alle Tests aus"""
    print("\n" + "=" * 60)
    print("SDOstreamclust (Sparse Data Observers Streaming Clustering) - Vollständiger Test")
    print("=" * 60 + "\n")
    
    try:
        test_basic_streaming_clustering()
        test_cluster_evolution()
        test_fading_and_cluster_observations()
        test_cluster_labeling()
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
