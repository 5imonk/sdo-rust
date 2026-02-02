#!/usr/bin/env python3
"""
Test und Visualisierung für SDOclust und SDOstreamclust.
Zeigt Daten, Vorhersagen und das Observer-Set (Modell für Label-Vorhersage).
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import adjusted_rand_score

try:
    from sdo import SDOclust, SDOstreamclust
except ImportError as e:
    print(f"Fehler: sdo-Modul nicht gefunden: {e}")
    print("Bitte mit 'maturin develop' im Projektroot bauen.")
    sys.exit(1)


def get_observers_and_labels(model):
    """
    Einheitlicher Zugriff auf Observer-Positionen und -Labels.
    Funktioniert für SDOclust und SDOstreamclust.
    """
    if hasattr(model, "get_active_observers") and hasattr(model, "get_observer_labels"):
        obs_array = model.get_active_observers()
        if obs_array is None or obs_array.size == 0:
            return np.array([]).reshape(0, 2), np.array([], dtype=np.int32)
        obs_points = np.asarray(obs_array)
        if obs_points.ndim == 1:
            obs_points = obs_points.reshape(1, -1)
        labels = np.array(model.get_observer_labels(), dtype=np.int32)
        return obs_points, labels
    return np.array([]).reshape(0, 2), np.array([], dtype=np.int32)


def plot_clustering_with_observers(
    x, y_true, y_pred, obs_points, obs_labels,
    title_prefix="", out_path=None, suptitle=None
):
    """
    Zeigt Daten (Ground Truth), Vorhersagen und Observer-Set in 3 Subplots.
    Nur für 2D-Daten (erste zwei Spalten).
    """
    x_plot = x[:, :2] if x.shape[1] >= 2 else x
    n_plots = 3 if obs_points.size > 0 else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    # Farben für Cluster (inkl. -1 = Outlier schwarz)
    all_labels = np.concatenate([
        y_pred[y_pred >= 0],
        (obs_labels[obs_labels >= 0] if len(obs_labels) else np.array([0])),
    ])
    n_colors = max(1, len(np.unique(all_labels)))
    cmap = plt.get_cmap("tab10", max(n_colors, 10))

    # 1) Ground Truth
    ax = axes[0]
    for lab in np.unique(y_true):
        if lab == -1:
            ax.scatter(
                x_plot[y_true == -1, 0], x_plot[y_true == -1, 1],
                c="black", s=20, marker="x", label="Outlier"
            )
        else:
            ax.scatter(
                x_plot[y_true == lab, 0], x_plot[y_true == lab, 1],
                c=[cmap(lab % n_colors)], s=15, label=f"GT {lab}"
            )
    ax.set_title(f"{title_prefix}Ground Truth")
    ax.set_xlabel("f0")
    ax.set_ylabel("f1")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # 2) Vorhersagen
    ax = axes[1]
    for lab in np.unique(y_pred):
        if lab == -1:
            ax.scatter(
                x_plot[y_pred == -1, 0], x_plot[y_pred == -1, 1],
                c="black", s=20, marker="x", label="Outlier"
            )
        else:
            ax.scatter(
                x_plot[y_pred == lab, 0], x_plot[y_pred == lab, 1],
                c=[cmap(lab % n_colors)], s=15, label=f"Pred {lab}"
            )
    ax.set_title(f"{title_prefix}Vorhersagen")
    ax.set_xlabel("f0")
    ax.set_ylabel("f1")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # 3) Observer-Set (Modell für Label-Vorhersage)
    if n_plots == 3 and obs_points.size > 0:
        ax = axes[2]
        obs_2d = obs_points[:, :2] if obs_points.shape[1] >= 2 else obs_points
        for lab in np.unique(obs_labels):
            if lab == -1:
                ax.scatter(
                    obs_2d[obs_labels == -1, 0], obs_2d[obs_labels == -1, 1],
                    c="black", s=40, marker="x", label="Outlier", zorder=3
                )
            else:
                ax.scatter(
                    obs_2d[obs_labels == lab, 0], obs_2d[obs_labels == lab, 1],
                    c=[cmap(lab % n_colors)], s=40, edgecolors="black", linewidths=0.5,
                    label=f"Observer {lab}", zorder=3
                )
        ax.set_title(f"{title_prefix}Observer-Set (Modell)")
        ax.set_xlabel("f0")
        ax.set_ylabel("f1")
        ax.legend(loc="best", fontsize=8)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120)
        print(f"  Grafik gespeichert: {out_path}")
    else:
        plt.show()
    plt.close(fig)


def run_sdoclust_test(visualize=True, out_dir="."):
    """SDOclust: Batch-Lernen, dann Vorhersage + Observer-Visualisierung."""
    print("=" * 60)
    print("SDOclust – Test & Visualisierung")
    print("=" * 60)

    np.random.seed(42)
    n_per_cluster = 40
    c1 = np.random.randn(n_per_cluster, 2) * 0.12 + np.array([0.25, 0.25])
    c2 = np.random.randn(n_per_cluster, 2) * 0.12 + np.array([0.75, 0.75])
    c3 = np.random.randn(n_per_cluster, 2) * 0.12 + np.array([0.5, 0.3])
    x = np.vstack([c1, c2, c3]).astype(np.float64)
    y_true = np.repeat([0, 1, 2], n_per_cluster)

    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    model = SDOclust(k=30, x=5, rho=0.2, chi=4, zeta=0.5, min_cluster_size=2)
    model.learn(x)
    print(f"  n_clusters: {model.n_clusters()}")

    y_pred = []
    for i in range(len(x)):
        point = x[i : i + 1, :]
        label, _ = model.predict(point, False)
        y_pred.append(label)
    y_pred = np.array(y_pred)

    obs_points, obs_labels = get_observers_and_labels(model)
    print(f"  Aktive Observer: {len(obs_labels)}")

    ari = adjusted_rand_score(y_true, y_pred)
    print(f"  ARI: {ari:.4f}")

    if visualize and x.shape[1] >= 2:
        out_path = os.path.join(out_dir, "sdoclust_observers.png") if out_dir else None
        plot_clustering_with_observers(
            x, y_true, y_pred, obs_points, obs_labels,
            title_prefix="SDOclust – ",
            out_path=out_path,
            suptitle="SDOclust: Daten, Vorhersagen, Observer-Set (Modell)",
        )
    print("  SDOclust-Test OK.\n")
    return ari


def run_sdostreamclust_test(visualize=True, out_dir=".", use_arff=None):
    """SDOstreamclust: Streaming mit oder ohne ARFF; am Ende Observer anzeigen."""
    print("=" * 60)
    print("SDOstreamclust – Test & Visualisierung")
    print("=" * 60)

    if use_arff and os.path.isfile(use_arff):
        from scipy.io import arff
        import pandas as pd
        with open(use_arff, "r") as f:
            arff_data = arff.loadarff(f)
        df = pd.DataFrame(arff_data[0])
        if "class" in df.columns:
            y_raw = df["class"]
            if y_raw.dtype == object:
                y_raw = y_raw.map(lambda v: v.decode("utf-8").strip() if isinstance(v, bytes) else v)
            y_true = np.array(pd.Categorical(y_raw).codes, dtype=np.int32, copy=True)
            # -1 für Outlier falls vorhanden
            if "-1" in df["class"].astype(str).values:
                y_true[df["class"].astype(str) == "-1"] = -1
        else:
            y_true = np.zeros(len(df), dtype=np.int32)
        df = df.drop(columns=["class"], errors="ignore")
        x = MinMaxScaler().fit_transform(df.to_numpy().astype(np.float64))
        n_dim = x.shape[1]
    else:
        np.random.seed(43)
        n_dim = 2
        n_stream = 150
        t = np.arange(n_stream, dtype=np.float64)
        # Einfacher Drift: zuerst Cluster 0, dann 1, dann gemischt
        cluster_id = np.where(t < 50, 0, np.where(t < 100, 1, 2))
        x = np.random.randn(n_stream, n_dim).astype(np.float64) * 0.08
        x += np.array([[0.3, 0.3], [0.7, 0.7], [0.5, 0.5]])[cluster_id]
        x = np.clip(x, 0, 1).astype(np.float64)
        y_true = cluster_id

    k = 40
    model = SDOstreamclust(
        k=k,
        x=5,
        t_fading=30.0,
        chi_min=1,
        chi_prop=0.1,
        zeta=0.6,
        min_cluster_size=2,
        dimension=n_dim,
    )
    predictions = []
    for i in range(len(x)):
        point = x[i : i + 1, :].astype(np.float64)
        time_arr = np.array([float(i)], dtype=np.float64)
        label, score = model.learn(point, time=time_arr)
        predictions.append(label)
    y_pred = np.array(predictions)

    obs_points, obs_labels = get_observers_and_labels(model)
    print(f"  Aktive Observer: {len(obs_labels)}")

    ari = adjusted_rand_score(y_true, y_pred)
    print(f"  ARI: {ari:.4f}")

    if visualize and x.shape[1] >= 2:
        out_path = os.path.join(out_dir, "sdostreamclust_observers.png") if out_dir else None
        plot_clustering_with_observers(
            x, y_true, y_pred, obs_points, obs_labels,
            title_prefix="SDOstreamclust – ",
            out_path=out_path,
            suptitle="SDOstreamclust: Daten, Vorhersagen, Observer-Set (Modell)",
        )
    print("  SDOstreamclust-Test OK.\n")
    return ari


def main():
    parser = argparse.ArgumentParser(
        description="SDOclust und SDOstreamclust testen und Observer-Set visualisieren.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Keine Grafiken anzeigen/speichern",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Verzeichnis für gespeicherte Grafiken (default: aktuelles Verzeichnis)",
    )
    parser.add_argument(
        "--arff",
        default="",
        help="Optional: ARFF-Datei für SDOstreamclust (z.B. evaluation_tests/data/example/concept_drift.arff)",
    )
    args = parser.parse_args()

    do_plot = not args.no_plot
    out_dir = args.out_dir
    arff_path = args.arff.strip() or None

    run_sdoclust_test(visualize=do_plot, out_dir=out_dir)
    run_sdostreamclust_test(
        visualize=do_plot,
        out_dir=out_dir,
        use_arff=arff_path,
    )
    print("=" * 60)
    print("Alle Tests und Visualisierungen abgeschlossen.")
    print("=" * 60)


if __name__ == "__main__":
    main()
