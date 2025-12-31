#!/usr/bin/env python3
"""
Vollständiges Testskript für SDOclust (Sparse Data Observers Clustering)
"""

import numpy as np
from sdo import SDOclust

def test_basic_clustering():
    """Grundlegende Clustering-Funktionalität"""
    print("=" * 60)
    print("Test 1: Grundlegende Clustering-Funktionalität")
    print("=" * 60)
    
    # Erstelle SDOclust-Instanz
    sdoclust = SDOclust()
    
    # Erstelle Daten mit zwei klar getrennten Clustern
    np.random.seed(42)
    cluster1 = np.random.randn(30, 2) * 0.5 + np.array([2.0, 2.0])
    cluster2 = np.random.randn(30, 2) * 0.5 + np.array([8.0, 8.0])
    data = np.vstack([cluster1, cluster2]).astype(np.float64)
    
    print(f"Trainingsdaten: {data.shape[0]} Punkte, {data.shape[1]} Dimensionen")
    print(f"  - Cluster 1: 30 Punkte um [2, 2]")
    print(f"  - Cluster 2: 30 Punkte um [8, 8]")
    
    # Trainiere das Modell
    print("\nTrainiere SDOclust-Modell...")
    sdoclust.learn(data, k=20, x=5, rho=0.2, chi=4, zeta=0.5, min_cluster_size=2)
    print(f"✓ Modell trainiert")
    print(f"  - Anzahl Cluster: {sdoclust.n_clusters()}")
    print(f"  - x (Nachbarn): {sdoclust.x}")
    print(f"  - chi (für Thresholds): {sdoclust.chi}")
    print(f"  - zeta (Mixing-Parameter): {sdoclust.zeta}")
    
    # Teste Clustering auf Trainingsdaten
    print("\nTeste Clustering auf Trainingsdaten:")
    labels = []
    for point in data:
        point_2d = point.reshape(1, -1)
        label = sdoclust.predict(point_2d)
        labels.append(label)
    
    labels = np.array(labels)
    unique_labels = np.unique(labels[labels >= 0])
    print(f"  - Gefundene Cluster: {len(unique_labels)}")
    for label in unique_labels:
        count = np.sum(labels == label)
        print(f"    Cluster {label}: {count} Punkte")
    
    # Teste neue Punkte
    print("\nTeste neue Punkte:")
    test_points = [
        ([2.0, 2.0], "Cluster 1 Zentrum"),
        ([8.0, 8.0], "Cluster 2 Zentrum"),
        ([5.0, 5.0], "Zwischen den Clustern"),
        ([15.0, 15.0], "Outlier"),
    ]
    
    for point, description in test_points:
        point_2d = np.array([point], dtype=np.float64)
        label = sdoclust.predict(point_2d)
        print(f"  {description:25}: Label = {label}")
    
    print("\n✓ Test 1 erfolgreich abgeschlossen\n")


def test_three_clusters():
    """Test mit drei Clustern"""
    print("=" * 60)
    print("Test 2: Drei Cluster")
    print("=" * 60)
    
    np.random.seed(123)
    cluster1 = np.random.randn(25, 2) * 0.4 + np.array([1.0, 1.0])
    cluster2 = np.random.randn(25, 2) * 0.4 + np.array([5.0, 1.0])
    cluster3 = np.random.randn(25, 2) * 0.4 + np.array([3.0, 5.0])
    data = np.vstack([cluster1, cluster2, cluster3]).astype(np.float64)
    
    print(f"Trainingsdaten: {data.shape[0]} Punkte in 3 Clustern")
    
    sdoclust = SDOclust()
    sdoclust.learn(data, k=30, x=5, rho=0.2, chi=4, zeta=0.5, min_cluster_size=2)
    
    print(f"✓ Gefundene Cluster: {sdoclust.n_clusters()}")
    
    # Teste Clustering
    labels = []
    for point in data:
        point_2d = point.reshape(1, -1)
        label = sdoclust.predict(point_2d)
        labels.append(label)
    
    labels = np.array(labels)
    unique_labels = np.unique(labels[labels >= 0])
    
    print("\nCluster-Verteilung:")
    for label in unique_labels:
        count = np.sum(labels == label)
        percentage = 100.0 * count / len(data)
        print(f"  Cluster {label}: {count} Punkte ({percentage:.1f}%)")
    
    print("\n✓ Test 2 erfolgreich abgeschlossen\n")


def test_different_parameters():
    """Test mit verschiedenen Parametern"""
    print("=" * 60)
    print("Test 3: Verschiedene Parameter")
    print("=" * 60)
    
    np.random.seed(42)
    cluster1 = np.random.randn(30, 2) * 0.5 + np.array([2.0, 2.0])
    cluster2 = np.random.randn(30, 2) * 0.5 + np.array([8.0, 8.0])
    data = np.vstack([cluster1, cluster2]).astype(np.float64)
    
    param_sets = [
        {"chi": 2, "zeta": 0.3, "min_cluster_size": 2, "name": "Konservativ"},
        {"chi": 4, "zeta": 0.5, "min_cluster_size": 3, "name": "Moderat"},
        {"chi": 6, "zeta": 0.7, "min_cluster_size": 5, "name": "Aggressiv"},
    ]
    
    print("Vergleich verschiedener Parameter:\n")
    for params in param_sets:
        sdoclust = SDOclust()
        sdoclust.learn(
            data,
            k=20,
            x=5,
            rho=0.2,
            chi=params["chi"],
            zeta=params["zeta"],
            min_cluster_size=params["min_cluster_size"],
        )
        
        n_clusters = sdoclust.n_clusters()
        print(f"  {params['name']:12} (χ={params['chi']}, ζ={params['zeta']:.1f}, "
              f"e={params['min_cluster_size']}): {n_clusters} Cluster")
    
    print("\n✓ Test 3 erfolgreich abgeschlossen\n")


def test_non_convex_clusters():
    """Test mit nicht-konvexen Clustern (Spiral)"""
    print("=" * 60)
    print("Test 4: Nicht-konvexe Cluster (Spiral)")
    print("=" * 60)
    
    # Erstelle Spiral-Daten
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, 50)
    spiral1 = np.column_stack([
        2 * t * np.cos(t) + np.random.randn(50) * 0.3,
        2 * t * np.sin(t) + np.random.randn(50) * 0.3,
    ])
    
    # Zweiter Cluster (Kreis)
    t2 = np.linspace(0, 2 * np.pi, 30)
    circle = np.column_stack([
        15 + 3 * np.cos(t2) + np.random.randn(30) * 0.2,
        15 + 3 * np.sin(t2) + np.random.randn(30) * 0.2,
    ])
    
    data = np.vstack([spiral1, circle]).astype(np.float64)
    
    print(f"Trainingsdaten: {data.shape[0]} Punkte (Spiral + Kreis)")
    
    sdoclust = SDOclust()
    sdoclust.learn(data, k=40, x=5, rho=0.2, chi=4, zeta=0.5, min_cluster_size=3)
    
    print(f"✓ Gefundene Cluster: {sdoclust.n_clusters()}")
    
    # Teste einige Punkte
    labels = []
    for point in data[::5]:  # Jeden 5. Punkt testen
        point_2d = point.reshape(1, -1)
        label = sdoclust.predict(point_2d)
        labels.append(label)
    
    unique_labels = np.unique([l for l in labels if l >= 0])
    print(f"  - Eindeutige Labels in Stichprobe: {len(unique_labels)}")
    
    print("\n✓ Test 4 erfolgreich abgeschlossen\n")


def test_observer_labels():
    """Test der Observer-Labels"""
    print("=" * 60)
    print("Test 5: Observer-Labels")
    print("=" * 60)
    
    np.random.seed(42)
    cluster1 = np.random.randn(20, 2) * 0.5 + np.array([2.0, 2.0])
    cluster2 = np.random.randn(20, 2) * 0.5 + np.array([8.0, 8.0])
    data = np.vstack([cluster1, cluster2]).astype(np.float64)
    
    sdoclust = SDOclust()
    sdoclust.learn(data, k=15, x=5, rho=0.2, chi=4, zeta=0.5, min_cluster_size=2)
    
    observer_labels = sdoclust.get_observer_labels()
    observers = sdoclust.get_active_observers()
    
    print(f"Anzahl aktiver Observer: {len(observer_labels)}")
    print(f"Shape der Observer: {observers.shape}")
    
    unique_labels = np.unique([l for l in observer_labels if l >= 0])
    print(f"Eindeutige Cluster-Labels: {unique_labels}")
    
    print("\nLabel-Verteilung der Observer:")
    for label in unique_labels:
        count = sum(1 for l in observer_labels if l == label)
        print(f"  Cluster {label}: {count} Observer")
    
    print("\n✓ Test 5 erfolgreich abgeschlossen\n")


def test_edge_cases():
    """Test von Edge Cases"""
    print("=" * 60)
    print("Test 6: Edge Cases")
    print("=" * 60)
    
    # Test 1: Sehr wenige Daten
    print("Test 6.1: Sehr wenige Daten")
    few_data = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float64)
    sdoclust = SDOclust()
    try:
        sdoclust.learn(few_data, k=2, x=1, rho=0.1, chi=1, zeta=0.5, min_cluster_size=1)
        print(f"  ✓ Wenige Daten: {sdoclust.n_clusters()} Cluster")
    except Exception as e:
        print(f"  ✗ Fehler: {e}")
    
    # Test 2: Einzelner Cluster
    print("\nTest 6.2: Einzelner kompakter Cluster")
    np.random.seed(42)
    single_cluster = np.random.randn(30, 2) * 0.3 + np.array([5.0, 5.0])
    sdoclust = SDOclust()
    sdoclust.learn(single_cluster, k=15, x=5, rho=0.2, chi=4, zeta=0.5, min_cluster_size=2)
    print(f"  ✓ Einzelner Cluster: {sdoclust.n_clusters()} Cluster gefunden")
    
    # Test 3: Hoher min_cluster_size
    print("\nTest 6.3: Hoher min_cluster_size")
    np.random.seed(42)
    data = np.random.randn(20, 2).astype(np.float64)
    sdoclust = SDOclust()
    sdoclust.learn(data, k=10, x=3, rho=0.2, chi=2, zeta=0.5, min_cluster_size=10)
    print(f"  ✓ Hoher min_cluster_size: {sdoclust.n_clusters()} Cluster")
    
    print("\n✓ Test 6 erfolgreich abgeschlossen\n")


def main():
    """Hauptfunktion - führt alle Tests aus"""
    print("\n" + "=" * 60)
    print("SDOclust (Sparse Data Observers Clustering) - Vollständiger Test")
    print("=" * 60 + "\n")
    
    try:
        test_basic_clustering()
        test_three_clusters()
        test_different_parameters()
        test_non_convex_clusters()
        test_observer_labels()
        test_edge_cases()
        
        print("=" * 60)
        print("✓ Alle Tests erfolgreich abgeschlossen!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Fehler beim Testen: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

