#!/usr/bin/env python3
"""
Einfaches Beispiel für die Verwendung von SDO
"""

import numpy as np
from sdo import SDO

# Erstelle SDO-Instanz
sdo = SDO()

# Beispiel-Daten
data = np.array([
    [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],
    [10.0, 11.0],  # Outlier
    [5.0, 6.0], [6.0, 7.0],
], dtype=np.float64)

# Trainiere das Modell
print("Trainiere SDO-Modell...")
sdo.learn(data, k=5, x=3, rho=0.2)
print(f"✓ Fertig! {sdo.x} aktive Observer")

# Teste einen Punkt
test_point = np.array([[10.0, 11.0]], dtype=np.float64)
score = sdo.predict(test_point)
print(f"\nOutlier-Score für [10.0, 11.0]: {score:.4f}")

# Teste einen normalen Punkt
normal_point = np.array([[3.0, 4.0]], dtype=np.float64)
score_normal = sdo.predict(normal_point)
print(f"Outlier-Score für [3.0, 4.0]: {score_normal:.4f}")

