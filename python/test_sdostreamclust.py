#!/usr/bin/env python3
"""
Vollständiges Testskript für SDOstreamclust (Sparse Data Observers Streaming Clustering)
"""

import sys
import os

# Add paths for sdo module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append('/home/simon/sdo/.venv/lib/python3.12/site-packages')

try:
    from sdostreamclust_sklearn import SDOstreamclust
except ImportError:
    print("Fehler: Das 'sdostreamclust' Modul konnte nicht importiert werden.")
    print("Bitte installieren Sie das Modul mit 'maturin develop' oder 'pip install .'")
    sys.exit(1)

import numpy as np
from sklearn.preprocessing import MinMaxScaler

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
    
    # Normalisiere Daten
    scaler = MinMaxScaler()
    init_data = scaler.fit_transform(init_data)
    
    print(f"Initialisierungsdaten: {init_data.shape[0]} Punkte, {init_data.shape[1]} Dimensionen")
    
    sdostreamclust = SDOstreamclust(
        k=10, x=3, t_fading=10.0,
        chi_min=1, chi_prop=0.05, zeta=0.5, min_cluster_size=2,
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
        
        old_label = sdostreamclust.predict(point_2d)
        sdostreamclust.learn(point_2d)
        new_label = sdostreamclust.predict(point_2d)
        
        print(f"  {label:10s} → Label: {old_label:2d} → {new_label:2d}")
    
    print(f"\nFinale Anzahl Observer: {sdostreamclust.x}")

    print(f"\n✓ Test 1 erfolgreich abgeschlossen\n")

def test_cluster_evolution():
    """Test der Cluster-Evolution über Zeit"""
    print("=" * 60)
    print("Test 2: Cluster-Evolution über Zeit")
    print("=" * 60)
    
    # Initialisiere
    np.random.seed(42)
    init_data = np.random.randn(10, 2).astype(np.float64)
    
    # Normalisiere Daten
    scaler = MinMaxScaler()
    init_data = scaler.fit_transform(init_data)
    
    sdostreamclust = SDOstreamclust(
        k=3, x=2, t_fading=10.0,
        chi_min=1, chi_prop=0.05, zeta=0.5, min_cluster_size=2,
        data=init_data
    )
    
    print("Verarbeite Punkte aus verschiedenen Clustern über Zeit:\n")
    
    # Phase 1: Cluster 1 (Punkte um [3, 3])
    cluster1_points = np.random.randn(5, 2) * 0.5 + np.array([3.0, 3.0])
    cluster1_points = scaler.transform(cluster1_points)
    for i in range(5):
        point = cluster1_points[i:i+1, :]
        old_label = sdostreamclust.predict(point)
        sdostreamclust.learn(point)
        new_label = sdostreamclust.predict(point)
        print(f"  Punkt {i+1}: Label = {old_label} → {new_label}")
    
    # Phase 2: Cluster 2 erscheint (Punkte um [10, 10])
    cluster2_points = np.random.randn(5, 2) * 0.5 + np.array([10.0, 10.0])
    cluster2_points = scaler.transform(cluster2_points)
    for i in range(5):
        point = cluster2_points[i:i+1, :]
        old_label = sdostreamclust.predict(point)
        sdostreamclust.learn(point)
        new_label = sdostreamclust.predict(point)
        print(f"  Punkt {i+1}: Label = {old_label} → {new_label}")
    
    # Phase 3: Gemischte Punkte
    mixed_points = np.random.randn(3, 2) * 0.5 + np.array([3.0, 3.0])
    mixed_points = scaler.transform(mixed_points)
    for i in range(3):
        point = mixed_points[i:i+1, :]
        old_label = sdostreamclust.predict(point)
        sdostreamclust.learn(point)
        new_label = sdostreamclust.predict(point)
        print(f"  Punkt {i+1}: Label = {old_label} → {new_label}")
    
    print(f"\n✓ Test 2 erfolgreich abgeschlossen\n")

def main():
    print("=" * 60)
    print("SDOstreamclust (Sparse Data Observers Streaming Clustering) - Vollständiger Test")
    print("=" * 60)
    
    test_basic_streaming_clustering()
    test_cluster_evolution()
    
    print("=" * 60)
    print("✓ Alle Tests erfolgreich abgeschlossen!")
    print("=" * 60)

if __name__ == "__main__":
    main()