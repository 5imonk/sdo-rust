#!/usr/bin/env python3
"""
Vollständiges Testskript für das SDO (Sparse Data Observers) Python-Modul
"""

import sys
import os

# Add paths for sdo module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append('/home/simon/sdo/.venv/lib/python3.12/site-packages')

try:
    from sdo import SDO
except ImportError:
    print("Fehler: Das 'sdo' Modul konnte nicht importiert werden.")
    print("Bitte installieren Sie das Modul mit 'maturin develop' oder 'pip install .'")
    sys.exit(1)

import numpy as np

def test_basic_usage():
    """Grundlegende Verwendung des SDO-Algorithmus"""
    print("=" * 60)
    print("Test 1: Grundlegende Verwendung")
    print("=" * 60)
    
    # Erstelle SDO-Instanz mit Parametern
    sdo = SDO(k=8, x=3, rho=0.2)
    
    # Trainingsdaten (2D-Daten mit normalen Punkten und einem Outlier)
    data = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
        [5.0, 6.0],
        [6.0, 7.0],
        [7.0, 8.0],
        [8.0, 9.0],
        [9.0, 10.0],
        [1.5, 2.5],
        [2.5, 3.5],
        [3.5, 4.5],
        [4.5, 5.5],
        [5.5, 6.5],
        [10.0, 11.0],  # Outlier
    ], dtype=np.float64)
    
    print(f"Trainingsdaten: {data.shape[0]} Punkte, {data.shape[1]} Dimensionen")
    print("Lerne SDO-Modell...")
    
    # Lerne das Modell
    sdo.learn(data)
    print(f"✓ Modell trainiert")
    print(f"  - Anzahl aktiver Observer: {sdo.x}")
    
    # Aktive Observer abrufen
    observers = sdo.get_active_observers()
    print(f"  - Shape der aktiven Observer: {observers.shape}")
    
    # Teste auf normalen Punkten
    normal_points = np.array([
        [4.0, 5.0],
        [5.0, 6.0],
        [2.0, 3.0],
    ], dtype=np.float64)
    
    print("\nOutlier-Scores für normale Punkte:")
    for point in normal_points:
        point_2d = point.reshape(1, -1)
        score = sdo.predict(point_2d)
        print(f"  Punkt [{point[0]:.1f}, {point[1]:.1f}]: Score = {score:.4f}")
    
    # Teste auf Outliern
    outlier_points = np.array([
        [20.0, 21.0],
        [100.0, 100.0],
        [-10.0, -10.0],
    ], dtype=np.float64)
    
    print("\nOutlier-Scores für Outlier:")
    for point in outlier_points:
        point_2d = point.reshape(1, -1)
        score = sdo.predict(point_2d)
        print(f"  Punkt [{point[0]:.1f}, {point[1]:.1f}]: Score = {score:.4f}")
    
    print("\n✓ Test 1 erfolgreich abgeschlossen\n")


def test_larger_dataset():
    """Test mit größerem Datensatz"""
    print("=" * 60)
    print("Test 2: Größerer Datensatz")
    print("=" * 60)
    
    # Generiere synthetische Daten
    np.random.seed(42)
    
    # Normale Daten (Cluster um [5, 5])
    normal_data = np.random.randn(100, 2) * 2 + np.array([5.0, 5.0])
    
    # Outlier
    outlier_data = np.array([
        [20.0, 20.0],
        [-10.0, -10.0],
        [15.0, -5.0],
        [0.0, 20.0],
        [-5.0, 15.0],
    ])
    
    all_data = np.vstack([normal_data, outlier_data]).astype(np.float64)
    
    print(f"Datensatz: {all_data.shape[0]} Punkte, {all_data.shape[1]} Dimensionen")
    print(f"  - Normale Punkte: {len(normal_data)}")
    print(f"  - Outlier: {len(outlier_data)}")
    
    # Lerne mit größerem Datensatz
    sdo = SDO(k=20, x=5, rho=0.3)
    sdo.learn(all_data)
    print(f"✓ Modell trainiert mit {sdo.x} aktiven Observern")
    
    # Berechne Scores für alle Punkte
    print("\nBerechne Outlier-Scores...")
    scores = []
    for point in all_data:
        point_2d = point.reshape(1, -1)
        score = sdo.predict(point_2d)
        scores.append(score)
    
    scores = np.array(scores)
    
    # Finde Top-Outlier
    print("\nTop 10 Outlier (höchste Scores):")
    top_indices = np.argsort(scores)[::-1][:10]
    for i, idx in enumerate(top_indices, 1):
        point = all_data[idx]
        score = scores[idx]
        is_outlier = idx >= len(normal_data)
        marker = "✓" if is_outlier else "✗"
        print(f"  {i}. {marker} Punkt [{point[0]:6.2f}, {point[1]:6.2f}]: Score = {score:.4f}")
    
    # Statistik
    normal_scores = scores[:len(normal_data)]
    outlier_scores = scores[len(normal_data):]
    
    print(f"\nStatistik:")
    print(f"  Normale Punkte - Mean: {normal_scores.mean():.4f}, Max: {normal_scores.max():.4f}")
    print(f"  Outlier - Mean: {outlier_scores.mean():.4f}, Min: {outlier_scores.min():.4f}")
    
    print("\n✓ Test 2 erfolgreich abgeschlossen\n")


def test_different_parameters():
    """Test mit verschiedenen Parametern"""
    print("=" * 60)
    print("Test 3: Verschiedene Parameter")
    print("=" * 60)
    
    # Generiere Daten
    np.random.seed(123)
    normal_data = np.random.randn(50, 2) * 1.5 + np.array([3.0, 3.0])
    outlier_data = np.array([[15.0, 15.0], [-5.0, -5.0]])
    all_data = np.vstack([normal_data, outlier_data]).astype(np.float64)
    
    # Verschiedene Parameter-Kombinationen
    param_sets = [
        {"k": 10, "x": 3, "rho": 0.2, "name": "Konservativ"},
        {"k": 20, "x": 5, "rho": 0.3, "name": "Moderat"},
        {"k": 30, "x": 7, "rho": 0.4, "name": "Aggressiv"},
    ]
    
    test_point = np.array([[15.0, 15.0]], dtype=np.float64)  # Bekannter Outlier
    
    print(f"Testpunkt (Outlier): [{test_point[0,0]}, {test_point[0,1]}]")
    print("\nVergleich verschiedener Parameter:")
    
    for params in param_sets:
        sdo = SDO(k=params["k"], x=params["x"], rho=params["rho"])
        sdo.learn(all_data)
        score = sdo.predict(test_point)
        
        print(f"  {params['name']:12} (k={params['k']:2}, x={params['x']}, rho={params['rho']:.1f}): "
              f"Score = {score:.4f}, Observer = {sdo.x}")
    
    print("\n✓ Test 3 erfolgreich abgeschlossen\n")


def test_edge_cases():
    """Test von Edge Cases"""
    print("=" * 60)
    print("Test 4: Edge Cases")
    print("=" * 60)
    
    # Test 1: Leere Daten
    print("Test 4.1: Leere Daten")
    empty_data = np.array([[]], dtype=np.float64).reshape(0, 2)
    try:
        sdo = SDO(k=5, x=3, rho=0.2)
        sdo.learn(empty_data)
        print("  ✓ Leere Daten werden korrekt behandelt")
    except Exception as e:
        print(f"  ✗ Fehler: {e}")
    
    # Test 2: Einzelner Datenpunkt
    print("\nTest 4.2: Einzelner Datenpunkt")
    single_point = np.array([[1.0, 2.0]], dtype=np.float64)
    sdo = SDO(k=1, x=1, rho=0.0)
    sdo.learn(single_point)
    score = sdo.predict(single_point)
    print(f"  ✓ Einzelner Punkt: Score = {score:.4f}")
    
    # Test 3: Sehr wenige Daten
    print("\nTest 4.3: Sehr wenige Daten")
    few_data = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float64)
    sdo = SDO(k=2, x=1, rho=0.1)
    sdo.learn(few_data)
    test_point = np.array([[10.0, 10.0]], dtype=np.float64)
    score = sdo.predict(test_point)
    print(f"  ✓ Wenige Daten: Score = {score:.4f}")
    
    # Test 4: 1D-Daten (wird zu 2D konvertiert)
    print("\nTest 4.4: 1D-Daten")
    data_1d = np.array([[1.0], [2.0], [3.0], [10.0]], dtype=np.float64)
    sdo = SDO(k=2, x=2, rho=0.2)
    sdo.learn(data_1d)
    point_1d = np.array([[5.0]], dtype=np.float64)
    score = sdo.predict(point_1d)
    print(f"  ✓ 1D-Daten: Score = {score:.4f}")
    
    print("\n✓ Test 4 erfolgreich abgeschlossen\n")


def test_performance():
    """Performance-Test"""
    print("=" * 60)
    print("Test 5: Performance")
    print("=" * 60)
    
    import time
    
    # Verschiedene Datensatzgrößen
    sizes = [100, 500, 1000, 5000]
    
    for size in sizes:
        # Generiere Daten
        np.random.seed(42)
        data = np.random.randn(size, 2).astype(np.float64)
        
        # Training
        start = time.time()
        sdo = SDO(k=min(50, size//10), x=5, rho=0.2)
        sdo.learn(data)
        train_time = time.time() - start
        
        # Prediction
        test_points = np.random.randn(100, 2).astype(np.float64)
        start = time.time()
        for point in test_points:
            point_2d = point.reshape(1, -1)
            sdo.predict(point_2d)
        predict_time = time.time() - start
        
        print(f"  Größe {size:5d}: Training = {train_time*1000:6.2f}ms, "
              f"100 Predictions = {predict_time*1000:6.2f}ms "
              f"({predict_time*10:6.2f}ms pro Prediction)")
    
    print("\n✓ Test 5 erfolgreich abgeschlossen\n")


def test_3d_data():
    """Test mit 3D-Daten"""
    print("=" * 60)
    print("Test 6: 3D-Daten")
    print("=" * 60)
    
    # Generiere 3D-Daten
    np.random.seed(42)
    normal_data = np.random.randn(50, 3) * 1.0 + np.array([0.0, 0.0, 0.0])
    outlier_data = np.array([
        [10.0, 10.0, 10.0],
        [-10.0, -10.0, -10.0],
    ])
    all_data = np.vstack([normal_data, outlier_data]).astype(np.float64)
    
    print(f"3D-Daten: {all_data.shape[0]} Punkte, {all_data.shape[1]} Dimensionen")
    
    sdo = SDO(k=15, x=5, rho=0.2)
    sdo.learn(all_data)
    print(f"✓ Modell trainiert")
    
    # Teste verschiedene Punkte
    test_points = [
        ([0.0, 0.0, 0.0], "Normal"),
        ([10.0, 10.0, 10.0], "Outlier"),
        ([5.0, 5.0, 5.0], "Zwischen"),
    ]
    
    print("\nTest-Punkte:")
    for point, label in test_points:
        point_2d = np.array([point], dtype=np.float64)
        score = sdo.predict(point_2d)
        print(f"  {label:8}: [{point[0]:5.1f}, {point[1]:5.1f}, {point[2]:5.1f}] "
              f"→ Score = {score:.4f}")
    
    print("\n✓ Test 6 erfolgreich abgeschlossen\n")


def main():
    """Hauptfunktion - führt alle Tests aus"""
    print("\n" + "=" * 60)
    print("SDO (Sparse Data Observers) - Vollständiger Test")
    print("=" * 60 + "\n")
    
    try:
        test_basic_usage()
        test_larger_dataset()
        test_different_parameters()
        test_edge_cases()
        test_performance()
        test_3d_data()
        
        print("=" * 60)
        print("✓ Alle Tests erfolgreich abgeschlossen!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Fehler beim Testen: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

