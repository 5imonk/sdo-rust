# SDOclust – Tests und Visualisierung

Von **Projektroot** aus:

- **Tests:**  
  `python python/sdoclust/test_sdoclust.py`  
  (oder `python3` / `.venv/bin/python`, je nach Umgebung)
- **Visualisierung:**  
  `python python/sdoclust/visualize_sdoclust.py [--arff <datei.arff>] [--out-dir python/sdoclust/out]`  
  Drei Panels: Ground Truth, Vorhersagen, Observer-Set (Modell). Grafik in `python/sdoclust/out/` (oder `--out-dir`).

Voraussetzung: `maturin develop` im Projektroot, damit `sdo` importierbar ist. Optional: virtuelle Umgebung mit `scikit-learn`, `matplotlib` (und für ARFF: `scipy`).
