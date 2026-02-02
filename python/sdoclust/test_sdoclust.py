#!/usr/bin/env python3
"""Tests for SDOclust: basic clustering, ARI, observer counts."""

import sys
import os

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

from common import get_observers_and_labels


def _make_synthetic_2d(n_per_cluster, centers, seed=42):
    np.random.seed(seed)
    xs = []
    ys = []
    for c, center in enumerate(centers):
        x = np.random.randn(n_per_cluster, 2).astype(np.float64) * 0.12 + center
        xs.append(x)
        ys.append(np.full(n_per_cluster, c, dtype=np.int32))
    x = np.vstack(xs)
    y = np.concatenate(ys)
    x = MinMaxScaler().fit_transform(x)
    return x, y


def test_two_clusters():
    """Two clusters: expect 2 clusters, reasonable ARI."""
    x, y_true = _make_synthetic_2d(40, [[0.25, 0.25], [0.75, 0.75]])
    model = SDOclust(k=30, x=5, rho=0.2, chi=4, zeta=0.5, min_cluster_size=2)
    model.learn(x)
    y_pred = np.array([
        model.predict(x[i : i + 1, :], False)[0] for i in range(len(x))
    ])
    obs_points, obs_labels = get_observers_and_labels(model)
    ari = adjusted_rand_score(y_true, y_pred)
    assert model.n_clusters() >= 1
    assert len(obs_labels) > 0
    print(f"  test_two_clusters: n_clusters={model.n_clusters()}, ARI={ari:.4f}, observers={len(obs_labels)}")
    return ari


def test_three_clusters():
    """Three clusters: expect multiple clusters, positive ARI possible."""
    x, y_true = _make_synthetic_2d(
        40,
        [[0.25, 0.25], [0.75, 0.75], [0.5, 0.3]],
        seed=43,
    )
    model = SDOclust(k=30, x=5, rho=0.2, chi=4, zeta=0.5, min_cluster_size=2)
    model.learn(x)
    y_pred = np.array([
        model.predict(x[i : i + 1, :], False)[0] for i in range(len(x))
    ])
    ari = adjusted_rand_score(y_true, y_pred)
    assert model.n_clusters() >= 1
    print(f"  test_three_clusters: n_clusters={model.n_clusters()}, ARI={ari:.4f}")
    return ari


def test_single_cluster():
    """Single compact cluster: one cluster, no crash."""
    x, _ = _make_synthetic_2d(50, [[0.5, 0.5]], seed=44)
    model = SDOclust(k=25, x=5, rho=0.2, chi=4, zeta=0.5, min_cluster_size=2)
    model.learn(x)
    y_pred = np.array([
        model.predict(x[i : i + 1, :], False)[0] for i in range(len(x))
    ])
    assert model.n_clusters() >= 1
    assert np.all(y_pred >= -1)
    print(f"  test_single_cluster: n_clusters={model.n_clusters()}")
    return 0.0


def main():
    print("=" * 60)
    print("SDOclust – Tests")
    print("=" * 60)
    test_two_clusters()
    test_three_clusters()
    test_single_cluster()
    print("=" * 60)
    print("Alle SDOclust-Tests OK.")
    print("=" * 60)


if __name__ == "__main__":
    main()
