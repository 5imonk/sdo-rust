#!/usr/bin/env python3
"""
Erweiterte Beispiele für die Verwendung von SDO, SDOclust und SDOstream
"""

import numpy as np
from sdo import SDO, SDOclust, SDOstream
from sklearn.preprocessing import MinMaxScaler


def example_sdo():
    """Beispiel für SDO (Sparse Data Observers)"""
    print("=" * 70)
    print("Beispiel 1: SDO (Sparse Data Observers) - Outlier Detection")
    print("=" * 70)
    
    # Generiere Beispiel-Daten mit normalen Punkten und Outliern
    np.random.seed(42)
    normal_data = np.random.randn(50, 2) * 1.5 + np.array([3.0, 3.0])
    outlier_data = np.array([
        [15.0, 15.0],
        [-5.0, -5.0],
        [20.0, 20.0],
    ])
    data = np.vstack([normal_data, outlier_data]).astype(np.float64)
    
    # Normalisiere Daten
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    
    print(f"\nDaten: {data.shape[0]} Punkte, {data.shape[1]} Dimensionen")
    print(f"  - Normale Punkte: {len(normal_data)}")
    print(f"  - Outlier: {len(outlier_data)}")
    
    # Trainiere das Modell
    print("\nTrainiere SDO-Modell...")
    sdo = SDO(k=20, x=5, rho=0.2)
    sdo.learn(data)
    print(f"✓ Fertig! {sdo.x} aktive Observer")
    
    # Berechne Scores für alle Punkte
    print("\nBerechne Outlier-Scores...")
    scores = []
    for point in data:
        point_2d = point.reshape(1, -1)
        score = sdo.predict(point_2d)
        scores.append(score)
    
    scores = np.array(scores)
    
    # Zeige Top-Outlier
    print("\nTop 5 Outlier (höchste Scores):")
    top_indices = np.argsort(scores)[::-1][:5]
    for i, idx in enumerate(top_indices, 1):
        point = data[idx]
        score = scores[idx]
        is_outlier = idx >= len(normal_data)
        marker = "✓" if is_outlier else "✗"
        print(f"  {i}. {marker} Punkt [{point[0]:6.2f}, {point[1]:6.2f}]: Score = {score:.4f}")
    
    # Teste neue Punkte
    print("\nTeste neue Punkte:")
    test_points = [
        ([3.0, 3.0], "Normal"),
        ([15.0, 15.0], "Outlier"),
        ([5.0, 5.0], "Normal"),
    ]
    
    for point, label in test_points:
        point_2d = np.array([point], dtype=np.float64)
        score = sdo.predict(point_2d)
        print(f"  {label:8}: [{point[0]:5.1f}, {point[1]:5.1f}] → Score = {score:.4f}")
    
    print("\n" + "=" * 70 + "\n")


def example_sdoclust():
    """Beispiel für SDOclust (Sparse Data Observers Clustering)"""
    print("=" * 70)
    print("Beispiel 2: SDOclust (Sparse Data Observers Clustering)")
    print("=" * 70)
    
    # Erstelle Daten mit zwei klar getrennten Clustern
    np.random.seed(42)
    cluster1 = np.random.randn(30, 2) * 0.5 + np.array([2.0, 2.0])
    cluster2 = np.random.randn(30, 2) * 0.5 + np.array([8.0, 8.0])
    data = np.vstack([cluster1, cluster2]).astype(np.float64)
    
    # Normalisiere Daten
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    
    print(f"\nDaten: {data.shape[0]} Punkte, {data.shape[1]} Dimensionen")
    print(f"  - Cluster 1: 30 Punkte um [2, 2]")
    print(f"  - Cluster 2: 30 Punkte um [8, 8]")
    
    # Trainiere das Modell
    print("\nTrainiere SDOclust-Modell...")
    sdoclust = SDOclust(k=20, x=5, rho=0.2, chi=4, zeta=0.5, min_cluster_size=2)
    sdoclust.learn(data)
    print(f"✓ Fertig! {sdoclust.n_clusters()} Cluster gefunden")
    
    # Teste Clustering auf Trainingsdaten
    print("\nClustering auf Trainingsdaten:")
    labels = []
    for point in data:
        point_2d = point.reshape(1, -1)
        label, _ = sdoclust.predict(point_2d, False)
        labels.append(label)
    
    labels = np.array(labels)
    unique_labels = np.unique(labels[labels >= 0])
    
    print(f"  Gefundene Cluster: {len(unique_labels)}")
    for label in unique_labels:
        count = np.sum(labels == label)
        percentage = 100.0 * count / len(data)
        print(f"    Cluster {label}: {count} Punkte ({percentage:.1f}%)")
    
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
        label, _ = sdoclust.predict(point_2d, False)
        print(f"  {description:25}: Label = {label}")
    
    # Outlier-Scores
    print("\nOutlier-Scores für Test-Punkte:")
    for point, description in test_points:
        point_2d = np.array([point], dtype=np.float64)
        _, score = sdoclust.predict(point_2d, True)
        print(f"  {description:25}: Score = {score:.4f}")
    
    print("\n" + "=" * 70 + "\n")


def example_sdostream():
    """Beispiel für SDOstream (Sparse Data Observers Streaming)"""
    print("=" * 70)
    print("Beispiel 3: SDOstream (Sparse Data Observers Streaming)")
    print("=" * 70)
    
    # Initialisiere mit Daten
    np.random.seed(42)
    init_data = np.random.randn(10, 2) * 1.0 + np.array([3.0, 3.0])
    init_data = init_data.astype(np.float64)
    
    # Normalisiere Daten
    scaler = MinMaxScaler()
    init_data = scaler.fit_transform(init_data)
    
    print(f"\nInitialisierungsdaten: {init_data.shape[0]} Punkte")
    
    sdostream = SDOstream(k=10, x=5, t_fading=10.0, data=init_data)
    print(f"✓ Modell initialisiert mit {sdostream.x} Observern")
    print(f"  Fading-Parameter f = exp(-1/T_fading) = {np.exp(-1.0/10.0):.4f}")
    print(f"  Sampling-Rate T_sampling = t_fading/k = {10.0/10:.2f} (automatisch berechnet)")
    
    # Streaming: Verarbeite einzelne Punkte
    print("\nStreaming-Verarbeitung (ein Punkt nach dem anderen):")
    streaming_points = [
        ([3.0, 3.0], "Normal"),
        ([15.0, 15.0], "Outlier"),
        ([3.5, 3.5], "Normal"),
        ([20.0, 20.0], "Outlier"),
        ([4.0, 4.0], "Normal"),
        ([25.0, 25.0], "Outlier"),
        ([3.2, 3.2], "Normal"),
    ]
    
    # Batch-Vorhersage für alle Punkte vor Verarbeitung
    points_array = np.array([p[0] for p in streaming_points], dtype=np.float64)
    scores_before = sdostream.predict_batch(points_array)
    
    # Batch-Learn für alle Punkte
    scores_after_batch = sdostream.learn_batch(points_array)
    
    # Zeige Ergebnisse
    for i, ((point, label), score_before, score_after) in enumerate(
        zip(streaming_points, scores_before, scores_after_batch), 1
    ):
        print(f"  {i}. {label:8}: [{point[0]:5.1f}, {point[1]:5.1f}] "
              f"→ Score: {score_before:.4f} → {score_after:.4f}")
    
    print(f"\nFinale Anzahl Observer: {sdostream.x}")
    
    # Zeige, wie sich das Modell an neue Daten anpasst
    print("\nAnpassung an neue Daten (Fading-Effekt):")
    print("  Das Modell verwendet Exponential Moving Average:")
    print("  - Pω ← f · Pω + 1 wenn ω unter den x-nächsten")
    print("  - Pω ← f · Pω sonst")
    print("  - Observer werden basierend auf normalisierter Qualität P̃ω = Pω / Hω ersetzt")
    
    print("\n" + "=" * 70 + "\n")


def example_comparison():
    """Vergleich der drei Algorithmen"""
    print("=" * 70)
    print("Beispiel 4: Vergleich SDO vs. SDOclust vs. SDOstream")
    print("=" * 70)
    
    # Generiere Daten
    np.random.seed(42)
    normal_data = np.random.randn(40, 2) * 1.5 + np.array([3.0, 3.0])
    outlier_data = np.array([[15.0, 15.0], [-5.0, -5.0]])
    data = np.vstack([normal_data, outlier_data]).astype(np.float64)
    
    # Normalisiere Daten
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    
    test_point = scaler.transform(np.array([[15.0, 15.0]], dtype=np.float64))
    
    print(f"\nDaten: {data.shape[0]} Punkte")
    print(f"Test-Punkt (Outlier): [{test_point[0,0]}, {test_point[0,1]}]")
    
    # SDO
    print("\n1. SDO (Outlier Detection):")
    sdo = SDO(k=15, x=5, rho=0.2)
    sdo.learn(data)
    sdo_score = sdo.predict(test_point)
    print(f"   Score: {sdo_score:.4f}")
    print(f"   Aktive Observer: {sdo.x}")
    
    # SDOclust
    print("\n2. SDOclust (Clustering + Outlier Detection):")
    sdoclust = SDOclust(k=15, x=5, rho=0.2, chi=4, zeta=0.5, min_cluster_size=2)
    sdoclust.learn(data)
    sdoclust_label, sdoclust_score = sdoclust.predict(test_point, True)
    print(f"   Cluster-Label: {sdoclust_label}")
    print(f"   Outlier-Score: {sdoclust_score:.4f}")
    print(f"   Anzahl Cluster: {sdoclust.n_clusters()}")
    
    # SDOstream
    print("\n3. SDOstream (Streaming Outlier Detection):")
    sdostream = SDOstream(k=15, x=5, t_fading=10.0, data=data)
    
    # Simuliere Streaming: Verarbeite einige Punkte mit Batch-API (learn unterstützt jetzt automatisch Batches)
    streaming_data = data[::5]  # Jeden 5. Punkt
    if len(streaming_data) > 0:
        sdostream.learn(streaming_data)
    
    sdostream_score = sdostream.predict(test_point)
    print(f"   Score: {sdostream_score:.4f}")
    print(f"   Anzahl Observer: {sdostream.x}")
    print(f"   Fading-Parameter f = {np.exp(-1.0/10.0):.4f}")
    
    print("\n" + "=" * 70 + "\n")

def main():
    """Hauptfunktion - führt alle Beispiele aus"""
    print("\n" + "=" * 70)
    print("SDO, SDOclust und SDOstream - Erweiterte Beispiele")
    print("=" * 70 + "\n")
    
    try:
        example_sdo()
        example_sdoclust()
        example_sdostream()
        example_comparison()
        
        print("=" * 70)
        print("✓ Alle Beispiele erfolgreich abgeschlossen!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Fehler beim Ausführen der Beispiele: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
