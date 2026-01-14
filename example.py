#!/usr/bin/env python3
"""
Erweiterte Beispiele für die Verwendung von SDO, SDOclust und SDOstream
"""

import numpy as np
from sdo import SDO, SDOParams, SDOclust, SDOclustParams, SDOstream, SDOstreamParams


def example_sdo():
    """Beispiel für SDO (Sparse Data Observers)"""
    print("=" * 70)
    print("Beispiel 1: SDO (Sparse Data Observers) - Outlier Detection")
    print("=" * 70)
    
    # Erstelle SDO-Instanz
    sdo = SDO()
    
    # Generiere Beispiel-Daten mit normalen Punkten und Outliern
    np.random.seed(42)
    normal_data = np.random.randn(50, 2) * 1.5 + np.array([3.0, 3.0])
    outlier_data = np.array([
        [15.0, 15.0],
        [-5.0, -5.0],
        [20.0, 20.0],
    ])
    data = np.vstack([normal_data, outlier_data]).astype(np.float64)
    
    print(f"\nDaten: {data.shape[0]} Punkte, {data.shape[1]} Dimensionen")
    print(f"  - Normale Punkte: {len(normal_data)}")
    print(f"  - Outlier: {len(outlier_data)}")
    
    # Trainiere das Modell
    print("\nTrainiere SDO-Modell...")
    params = SDOParams(k=20, x=5, rho=0.2)
    sdo.initialize(params)
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
    
    # Erstelle SDOclust-Instanz
    sdoclust = SDOclust()
    
    # Erstelle Daten mit zwei klar getrennten Clustern
    np.random.seed(42)
    cluster1 = np.random.randn(30, 2) * 0.5 + np.array([2.0, 2.0])
    cluster2 = np.random.randn(30, 2) * 0.5 + np.array([8.0, 8.0])
    data = np.vstack([cluster1, cluster2]).astype(np.float64)
    
    print(f"\nDaten: {data.shape[0]} Punkte, {data.shape[1]} Dimensionen")
    print(f"  - Cluster 1: 30 Punkte um [2, 2]")
    print(f"  - Cluster 2: 30 Punkte um [8, 8]")
    
    # Trainiere das Modell
    print("\nTrainiere SDOclust-Modell...")
    params = SDOclustParams(k=20, x=5, rho=0.2, chi=4, zeta=0.5, min_cluster_size=2)
    sdoclust.initialize(params)
    sdoclust.learn(data)
    print(f"✓ Fertig! {sdoclust.n_clusters()} Cluster gefunden")
    
    # Teste Clustering auf Trainingsdaten
    print("\nClustering auf Trainingsdaten:")
    labels = []
    for point in data:
        point_2d = point.reshape(1, -1)
        label = sdoclust.predict(point_2d)
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
        label = sdoclust.predict(point_2d)
        print(f"  {description:25}: Label = {label}")
    
    # Outlier-Scores
    print("\nOutlier-Scores für Test-Punkte:")
    for point, description in test_points:
        point_2d = np.array([point], dtype=np.float64)
        score = sdoclust.predict_outlier_score(point_2d)
        print(f"  {description:25}: Score = {score:.4f}")
    
    print("\n" + "=" * 70 + "\n")


def example_sdostream():
    """Beispiel für SDOstream (Sparse Data Observers Streaming)"""
    print("=" * 70)
    print("Beispiel 3: SDOstream (Sparse Data Observers Streaming)")
    print("=" * 70)
    
    # Erstelle SDOstream-Instanz
    sdostream = SDOstream()
    
    # Initialisiere mit Daten
    np.random.seed(42)
    init_data = np.random.randn(10, 2) * 1.0 + np.array([3.0, 3.0])
    init_data = init_data.astype(np.float64)
    
    print(f"\nInitialisierungsdaten: {init_data.shape[0]} Punkte")
    
    params = SDOstreamParams(k=10, x=5, t_fading=10.0, t_sampling=10.0)
    sdostream.initialize(params, data=init_data)
    print(f"✓ Modell initialisiert mit {sdostream.sdo.x} Observern")
    print(f"  Fading-Parameter f = exp(-1/T_fading) = {np.exp(-1.0/params.t_fading):.4f}")
    print(f"  Sampling-Rate T_sampling = {params.t_sampling:.2f}")
    
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
    
    for i, (point, label) in enumerate(streaming_points, 1):
        point_2d = np.array([point], dtype=np.float64)
        
        # Score vor Verarbeitung
        score_before = sdostream.predict(point_2d)
        
        # Verarbeite Punkt (Modell passt sich an)
        sdostream.learn(point_2d)
        
        # Score nach Verarbeitung
        score_after = sdostream.predict(point_2d)
        
        print(f"  {i}. {label:8}: [{point[0]:5.1f}, {point[1]:5.1f}] "
              f"→ Score: {score_before:.4f} → {score_after:.4f}")
    
    print(f"\nFinale Anzahl Observer: {sdostream.sdo.x}")
    
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
    
    test_point = np.array([[15.0, 15.0]], dtype=np.float64)
    
    print(f"\nDaten: {data.shape[0]} Punkte")
    print(f"Test-Punkt (Outlier): [{test_point[0,0]}, {test_point[0,1]}]")
    
    # SDO
    print("\n1. SDO (Outlier Detection):")
    sdo = SDO()
    sdo_params = SDOParams(k=15, x=5, rho=0.2)
    sdo.initialize(sdo_params)
    sdo.learn(data)
    sdo_score = sdo.predict(test_point)
    print(f"   Score: {sdo_score:.4f}")
    print(f"   Aktive Observer: {sdo.x}")
    
    # SDOclust
    print("\n2. SDOclust (Clustering + Outlier Detection):")
    sdoclust = SDOclust()
    sdoclust_params = SDOclustParams(k=15, x=5, rho=0.2, chi=4, zeta=0.5, min_cluster_size=2)
    sdoclust.initialize(sdoclust_params)
    sdoclust.learn(data)
    sdoclust_label = sdoclust.predict(test_point)
    sdoclust_score = sdoclust.predict_outlier_score(test_point)
    print(f"   Cluster-Label: {sdoclust_label}")
    print(f"   Outlier-Score: {sdoclust_score:.4f}")
    print(f"   Anzahl Cluster: {sdoclust.n_clusters()}")
    
    # SDOstream
    print("\n3. SDOstream (Streaming Outlier Detection):")
    sdostream = SDOstream()
    sdostream_params = SDOstreamParams(k=15, x=5, t_fading=10.0, t_sampling=10.0)
    sdostream.initialize(sdostream_params, data=data)
    
    # Simuliere Streaming: Verarbeite einige Punkte
    for point in data[::5]:  # Jeden 5. Punkt
        point_2d = point.reshape(1, -1)
        sdostream.learn(point_2d)
    
    sdostream_score = sdostream.predict(test_point)
    print(f"   Score: {sdostream_score:.4f}")
    print(f"   Anzahl Observer: {sdostream.sdo.x}")
    print(f"   Fading-Parameter f = {np.exp(-1.0/sdostream_params.t):.4f}")
    
    print("\n" + "=" * 70 + "\n")


def example_sklearn_integration():
    """Beispiel für sklearn-Integration"""
    print("=" * 70)
    print("Beispiel 5: Scikit-learn-ähnliche API")
    print("=" * 70)
    
    try:
        from sdo_sklearn import SDOOutlierDetector
        from sdoclust_sklearn import SDOclustClusterer
        from sdostream_sklearn import SDOstreamOutlierDetector
        
        # SDO mit sklearn-API
        print("\n1. SDO mit sklearn-API:")
        detector = SDOOutlierDetector(k=15, x=5, rho=0.2)
        
        np.random.seed(42)
        X = np.random.randn(50, 2) * 1.5 + np.array([3.0, 3.0])
        X = X.astype(np.float64)
        
        scores = detector.fit_predict(X)
        print(f"   Trainiert mit {len(X)} Punkten")
        print(f"   Top 3 Outlier-Scores: {np.sort(scores)[::-1][:3]}")
        
        # SDOstream mit sklearn-API
        print("\n2. SDOstream mit sklearn-API:")
        stream_detector = SDOstreamOutlierDetector(k=10, x=5, t_fading=10.0, t_sampling=10.0)
        stream_detector.fit(X[:20])  # Initialisiere mit ersten 20 Punkten
        
        # Streaming
        for point in X[20:30]:
            point_2d = point.reshape(1, -1)
            score = stream_detector.predict(point_2d)
            stream_detector.partial_fit(point_2d)
            print(f"   Punkt [{point[0]:5.1f}, {point[1]:5.1f}]: Score = {score[0]:.4f}")
        
        print("\n✓ sklearn-Integration erfolgreich")
        
    except ImportError as e:
        print(f"\n⚠ sklearn-Wrapper nicht verfügbar: {e}")
        print("   Installiere die sklearn-Wrapper-Dateien für diese Funktionalität")
    
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
        example_sklearn_integration()
        
        print("=" * 70)
        print("✓ Alle Beispiele erfolgreich abgeschlossen!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Fehler beim Ausführen der Beispiele: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
