# SDO Python Bindings

Python-Bindings für den Sparse Data Observers (SDO) Algorithmus und SDOclust (Clustering-Erweiterung), implementiert in Rust.

## Installation

1. Stelle sicher, dass ein Python virtualenv aktiviert ist:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Installiere maturin (falls noch nicht installiert):
```bash
pip install maturin
```

3. Baue und installiere das Modul:
```bash
maturin develop
```

## Verwendung

### Direkte Verwendung

```python
import numpy as np
from sdo import SDO

# Erstelle SDO-Instanz
sdo = SDO()

# Trainingsdaten
data = np.array([
    [1.0, 2.0], [2.0, 3.0], [3.0, 4.0],
    [10.0, 11.0],  # Outlier
], dtype=np.float64)

# Trainiere das Modell
sdo.learn(data, k=5, x=3, rho=0.2)

# Berechne Outlier-Score
point = np.array([[10.0, 11.0]], dtype=np.float64)
score = sdo.predict(point)
print(f"Outlier-Score: {score}")
```

### Scikit-learn-ähnliche API

```python
import numpy as np
from sdo_sklearn import SDOOutlierDetector

# Erstelle Detector
detector = SDOOutlierDetector(k=10, x=5, rho=0.2)

# Trainiere und berechne Scores
X = np.array([[1, 2], [2, 3], [10, 11]], dtype=np.float64)
scores = detector.fit_predict(X)

# Teste neue Daten
new_scores = detector.predict(np.array([[5, 6]], dtype=np.float64))
```

## Parameter

### SDO Parameter

- **k**: Anzahl der zu samplenden Observer (Standard: 10)
- **x**: Anzahl der nächsten Nachbarn für Observations (Standard: 5)
- **rho**: Fraktion der Observer, die als inaktiv entfernt werden, 0.0-1.0 (Standard: 0.2)

### SDOclust Parameter

- **k**: Anzahl der zu samplenden Observer (Standard: 10)
- **x**: Anzahl der nächsten Nachbarn für Observations und Predictions (Standard: 5)
- **rho**: Fraktion der Observer, die als inaktiv entfernt werden, 0.0-1.0 (Standard: 0.2)
- **chi** (χ): Anzahl der nächsten Observer für lokale Cutoff-Thresholds (Standard: 4)
- **zeta** (ζ): Mixing-Parameter für globale/lokale Thresholds, 0.0-1.0 (Standard: 0.5)
  - Höhere Werte betonen lokale Thresholds mehr
- **min_cluster_size** (e): Minimale Clustergröße (Standard: 2)
  - Cluster mit weniger Observer werden entfernt

## Beispiele

### Einfaches Beispiel
```bash
python example.py
```

### Vollständiger Test
```bash
python test_sdo.py
```

### Scikit-learn API Beispiel
```bash
python sdo_sklearn.py
```

### SDOclust Test
```bash
python test_sdoclust.py
```

### SDOclust Scikit-learn API Beispiel
```bash
python sdoclust_sklearn.py
```

## API Referenz

### SDO Klasse

#### Methoden

- `learn(data, k, x, rho)`: Trainiere das Modell
  - `data`: NumPy-Array (n_samples, n_features) mit dtype=np.float64
  - `k`: Anzahl Observer
  - `x`: Anzahl Nachbarn
  - `rho`: Fraktion zu entfernender Observer

- `predict(point)`: Berechne Outlier-Score
  - `point`: NumPy-Array (1, n_features) mit dtype=np.float64
  - Returns: Outlier-Score (float)

- `get_active_observers()`: Gibt aktive Observer zurück
  - Returns: NumPy-Array (n_observers, n_features)

### SDOOutlierDetector Klasse

Scikit-learn-kompatible API mit folgenden Methoden:

- `fit(X)`: Trainiere das Modell
- `predict(X)`: Berechne Outlier-Scores
- `fit_predict(X)`: Trainiere und berechne Scores
- `score_samples(X)`: Alias für predict
- `decision_function(X)`: Alias für predict

## SDOclust - Clustering

SDOclust erweitert SDO um Clustering-Fähigkeiten. Es verwendet Connected Components Clustering (CCC) mit lokalen Cutoff-Thresholds.

### Direkte Verwendung

```python
import numpy as np
from sdo import SDOclust

# Erstelle SDOclust-Instanz
sdoclust = SDOclust()

# Trainingsdaten mit mehreren Clustern
np.random.seed(42)
cluster1 = np.random.randn(30, 2) * 0.5 + np.array([2.0, 2.0])
cluster2 = np.random.randn(30, 2) * 0.5 + np.array([8.0, 8.0])
data = np.vstack([cluster1, cluster2]).astype(np.float64)

# Trainiere das Modell
sdoclust.learn(data, k=20, x=5, rho=0.2, chi=4, zeta=0.5, min_cluster_size=2)

# Berechne Cluster-Label
point = np.array([[2.0, 2.0]], dtype=np.float64)
label = sdoclust.predict(point)
print(f"Cluster-Label: {label}")

# Anzahl der Cluster
print(f"Anzahl Cluster: {sdoclust.n_clusters()}")
```

### Scikit-learn-ähnliche API

```python
import numpy as np
from sdoclust_sklearn import SDOclustClusterer

# Erstelle Clusterer
clusterer = SDOclustClusterer(k=20, x=5, chi=4, zeta=0.5, min_cluster_size=2)

# Trainiere und berechne Labels
X = np.array([[1, 2], [2, 3], [8, 9], [9, 10]], dtype=np.float64)
labels = clusterer.fit_predict(X)

# Teste neue Daten
new_labels = clusterer.predict(np.array([[5, 6]], dtype=np.float64))
```

## API Referenz

### SDOclust Klasse

#### Methoden

- `learn(data, k, x, rho, chi, zeta, min_cluster_size)`: Trainiere das Modell und führe Clustering durch
  - `data`: NumPy-Array (n_samples, n_features) mit dtype=np.float64
  - `k`: Anzahl Observer
  - `x`: Anzahl Nachbarn
  - `rho`: Fraktion zu entfernender Observer
  - `chi`: Anzahl nächster Observer für lokale Thresholds
  - `zeta`: Mixing-Parameter (0.0-1.0)
  - `min_cluster_size`: Minimale Clustergröße

- `predict(point)`: Berechne Cluster-Label
  - `point`: NumPy-Array (1, n_features) mit dtype=np.float64
  - Returns: Cluster-Label (int, -1 = Outlier)

- `n_clusters()`: Gibt die Anzahl der gefundenen Cluster zurück
  - Returns: Anzahl Cluster (int)

- `get_observer_labels()`: Gibt die Labels der Observer zurück
  - Returns: Liste von Labels (int, -1 = entfernt)

- `get_active_observers()`: Gibt aktive Observer zurück
  - Returns: NumPy-Array (n_observers, n_features)

### SDOclustClusterer Klasse

Scikit-learn-kompatible API mit folgenden Methoden:

- `fit(X)`: Trainiere das Modell
- `predict(X)`: Berechne Cluster-Labels
- `fit_predict(X)`: Trainiere und berechne Labels
- `get_observer_labels()`: Gibt Observer-Labels zurück
- `get_active_observers()`: Gibt aktive Observer zurück

## Hinweise

- Alle NumPy-Arrays müssen `dtype=np.float64` haben
- `predict()` erwartet ein 2D-Array mit einer Zeile: `point.reshape(1, -1)`
- **SDO**: Höhere Scores bedeuten höhere Wahrscheinlichkeit, dass der Punkt ein Outlier ist
- **SDOclust**: Labels >= 0 sind Cluster-Labels, -1 bedeutet Outlier/kein Cluster
- SDOclust kann nicht-konvexe Cluster identifizieren (z.B. Spiralen)

