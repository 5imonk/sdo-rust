#!/usr/bin/env python3
"""Visualize SDOclust: Ground Truth, Predictions, Observer set (2D)."""

import sys
import os
import argparse

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _THIS_DIR)  # so "from common import ..." works

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import adjusted_rand_score

try:
    from sdo import SDOclust
except ImportError as e:
    print(f"Fehler: sdo-Modul nicht gefunden: {e}")
    print("Bitte im Projektroot 'maturin develop' ausführen.")
    sys.exit(1)

from common import get_observers_and_labels, plot_clustering_with_observers


def load_synthetic(seed=42):
    """Three 2D clusters, MinMax-scaled (reduced variance for less overlap)."""
    np.random.seed(seed)
    n = 40
    std = 0.06
    c1 = np.random.randn(n, 2).astype(np.float64) * std + np.array([0.25, 0.25])
    c2 = np.random.randn(n, 2).astype(np.float64) * std + np.array([0.75, 0.75])
    c3 = np.random.randn(n, 2).astype(np.float64) * std + np.array([0.5, 0.3])
    x = np.vstack([c1, c2, c3])
    y_true = np.repeat([0, 1, 2], n)
    x = MinMaxScaler().fit_transform(x)
    return x, y_true


def load_arff(path):
    """Load ARFF; return x (float), y_true (int), -1 for outliers if present."""
    from scipy.io import arff
    import pandas as pd
    with open(path, "r") as f:
        arff_data = arff.loadarff(f)
    df = pd.DataFrame(arff_data[0])
    if "class" not in df.columns:
        x = MinMaxScaler().fit_transform(df.astype(np.float64).to_numpy())
        return x, np.zeros(len(df), dtype=np.int32)
    y_raw = df["class"]
    if y_raw.dtype == object:
        y_raw = y_raw.map(
            lambda v: v.decode("utf-8").strip() if isinstance(v, bytes) else v
        )
    y_true = np.array(pd.Categorical(y_raw).codes, dtype=np.int32, copy=True)
    if "-1" in df["class"].astype(str).values:
        y_true[df["class"].astype(str) == "-1"] = -1
    df = df.drop(columns=["class"], errors="ignore")
    x = MinMaxScaler().fit_transform(df.astype(np.float64).to_numpy())
    return x, y_true


def main():
    parser = argparse.ArgumentParser(description="SDOclust: Test + Visualisierung (Observer-Set).")
    parser.add_argument("--arff", default="", help="Optional: ARFF-Datei (sonst synthetische 2D-Daten).")
    parser.add_argument("--out-dir", default=None, help="Verzeichnis für Grafik (default: python/sdoclust/out).")
    parser.add_argument("--no-plot", action="store_true", help="Keine Grafik speichern/anzeigen.")
    args = parser.parse_args()

    if args.arff and os.path.isfile(args.arff):
        x, y_true = load_arff(args.arff)
        print(f"Daten: {args.arff}, {x.shape[0]} Punkte, {x.shape[1]} Dimensionen")
    else:
        x, y_true = load_synthetic()
        print("Daten: synthetisch (3 Cluster, 2D)")

    if x.shape[1] < 2:
        print("Visualisierung nur für mind. 2 Dimensionen; nur ARI wird ausgegeben.")

    model = SDOclust(k=30, x=5, rho=0.2, chi=4, zeta=0.5, min_cluster_size=2)
    model.learn(x)
    y_pred = np.array([model.predict(x[i : i + 1, :], False)[0] for i in range(len(x))])
    obs_points, obs_labels = get_observers_and_labels(model)

    ari = adjusted_rand_score(y_true, y_pred)
    print(f"  n_clusters: {model.n_clusters()}")
    print(f"  Aktive Observer: {len(obs_labels)}")
    print(f"  ARI: {ari:.4f}")

    if not args.no_plot and x.shape[1] >= 2:
        out_dir = args.out_dir or os.path.join(os.path.dirname(__file__), "out")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "sdoclust_observers.png")
        plot_clustering_with_observers(
            x, y_true, y_pred, obs_points, obs_labels,
            title_prefix="SDOclust – ",
            out_path=out_path,
            suptitle="SDOclust: Daten, Vorhersagen, Observer-Set (Modell)",
        )

    print("Fertig.")


if __name__ == "__main__":
    main()
