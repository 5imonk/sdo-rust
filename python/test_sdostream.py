#!/usr/bin/env python3
"""
Vollständiges Testskript für SDOstream (Sparse Data Observers Streaming)
"""

import sys
import os

# Add paths for sdo module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append('/home/simon/sdo/.venv/lib/python3.12/site-packages')

try:
    from sdo import SDOstream
except ImportError:
    print("Fehler: Das 'sdo' Modul konnte nicht importiert werden.")
    print("Bitte installieren Sie das Modul mit 'maturin develop' oder 'pip install .'")
    sys.exit(1)

import numpy as np

def test_basic_streaming_clustering():
    """Test grundlegende Streaming-Clustering-Funktionalität"""
    print("=" * 60)
    print("Test 1: Grundlegende Streaming-Clustering-Funktionalität")
    print("=" * 60)
    
    # Generiere Beispieldaten
    np.random.seed(42)
    cluster1 = np.random.randn(5, 2) * 0.5 + np.array([2.0, 2.0])
    cluster2 = np.random.randn(5, 2) * 0.5 + np.array([8.0, 8.0])
    init_data = np.vstack([cluster1, cluster2]).astype(np.float64)
    
    print(f"Initialisierungsdaten: {init_data.shape[0]} Punkte, {init_data.shape[1]} Dimensionen")
    
    sdostream = SDOstream(
        k=10, x=5, t_fading=10.0,
        distance="euclidean",
        minkowski_p=None,
        data=init_data
    )
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
        
        old_score = sdostream.predict(point_2d)
        sdostream.learn(point_2d)
        new_score = sdostream.predict(point_2d)
        
        print(f"  {label:10s} → Score: {old_score:.4f} → {new_score:.4f}")
    
    print(f"\nFinale Anzahl Observer: {sdostream.x}")

    print(f"\n✓ Test 1 erfolgreich abgeschlossen\n")

def test_cluster_evolution():
    """Test der Cluster-Evolution über Zeit"""
    print("=" * 60)
    print("Test 2: Cluster-Evolution über Zeit")
    print("=" * 60)
    
    # Initialisiere
    np.random.seed(42)
    init_data = np.random.randn(10, 2).astype(np.float64)
    
    sdostream = SDOstream(
        k=3, x=2, t_fading=10.0,
        data=init_data
    )
    
    print("Verarbeite Punkte aus verschiedenen Clustern über Zeit:\n")
    
    # Phase 1: Cluster 1 (Punkte um [3, 3])
    cluster1_points = np.random.randn(5, 2) * 0.5 + np.array([3.0, 3.0])
    for i in range(5):
        point = cluster1_points[i:i+1, :]
        old_score = sdostream.predict(point)
        sdostream.learn(point)
        new_score = sdostream.predict(point)
        print(f"  Punkt {i+1}: Score = {old_score:.4f} → {new_score:.4f}")
    
    # Phase 2: Cluster 2 erscheint (Punkte um [10, 10])
    cluster2_points = np.random.randn(5, 2) * 0.5 + np.array([10.0, 10.0])
    for i in range(5):
        point = cluster2_points[i:i+1, :]
        old_score = sdostream.predict(point)
        sdostream.learn(point)
        new_score = sdostream.predict(point)
        print(f"  Punkt {i+1}: Score = {old_score:.4f} → {new_score:.4f}")
    
    # Phase 3: Gemischte Punkte
    mixed_points = np.random.randn(3, 2) * 0.5 + np.array([3.0, 3.0])
    for i in range(3):
        point = mixed_points[i:i+1, :]
        old_score = sdostream.predict(point)
        sdostream.learn(point)
        new_score = sdostream.predict(point)
        print(f"  Punkt {i+1}: Score = {old_score:.4f} → {new_score:.4f}")
    
    print(f"\n✓ Test 2 erfolgreich abgeschlossen\n")

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
    
    param_sets = [
        {"t_fading": 5.0, "name": "Konservativ"},
        {"t_fading": 10.0, "name": "Moderat"},
        {"t_fading": 20.0, "name": "Aggressiv"},
    ]
    
    print("Vergleich verschiedener Parameter:\n")
    
    for params_dict in param_sets:
        t_fading = 10.0  # Default value
        if 't_fading' in params_dict:
            t_fading = params_dict['t_fading']
            
        sdostream = SDOstream(
            k=3,
            x=2,
            t_fading=t_fading,
            distance="euclidean",
            minkowski_p=None,
            data=init_data,
        )
        
        scores = [sdostream.predict(np.array([point]).reshape(1, -1)) for point in [
            [2.0, 2.0], 
            [2.0, 2.0], 
            [3.0, 3.0], 
            [4.0, 4.0]
        ]]
        print(f"  {params_dict['name']}: Scores = {scores}")
    
    print(f"\n✓ Test 5 erfolgreich abgeschlossen\n")

def main():
    print("=" * 60)
    print("SDOstream (Sparse Data Observers Streaming) - Vollständiger Test")
    print("=" * 60 + "\n")
    
    test_basic_streaming_clustering()
    test_cluster_evolution()
    test_different_parameters()
    
    print("=" * 60)
    print("✓ Alle Tests erfolgreich abgeschlossen!")
    print("=" * 60)

if __name__ == "__main__":
    main()