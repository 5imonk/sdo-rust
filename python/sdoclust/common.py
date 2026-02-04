"""Shared helpers for SDOclust test and visualization."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Doppelkodierung: Farbe + Marker pro Klasse (gut unterscheidbar, auch bei 3 Klassen)
CLASS_STYLE = [
    ("#1f77b4", "o"),   # blau, Kreis
    ("#ff7f0e", "s"),   # orange, Quadrat
    ("#2ca02c", "^"),   # grün, Dreieck oben
    ("#d62728", "v"),   # rot, Dreieck unten
    ("#9467bd", "D"),   # lila, Diamant
    ("#8c564b", "P"),   # braun, Plus (5-seitig)
]


def get_observers_and_labels(model, with_final_threshold_radii=False):
    """Observer positions and labels from an SDOclust model.
    If with_final_threshold_radii=True and model has get_active_observers_with_final_thresholds,
    returns (obs_points, labels, radii) with radii for visualization circles; else radii is None."""
    radii = None
    if hasattr(model, "get_active_observers") and hasattr(model, "get_observer_labels"):
        if with_final_threshold_radii and hasattr(model, "get_active_observers_with_final_thresholds"):
            obs_array, labels_list, radii_list = model.get_active_observers_with_final_thresholds()
            if obs_array is not None and obs_array.size > 0:
                obs_points = np.asarray(obs_array)
                if obs_points.ndim == 1:
                    obs_points = obs_points.reshape(1, -1)
                labels = np.array(labels_list, dtype=np.int32)
                radii = np.array(radii_list, dtype=np.float64)
                return obs_points, labels, radii
        obs_array = model.get_active_observers()
        if obs_array is None or obs_array.size == 0:
            return np.array([]).reshape(0, 2), np.array([], dtype=np.int32), None
        obs_points = np.asarray(obs_array)
        if obs_points.ndim == 1:
            obs_points = obs_points.reshape(1, -1)
        labels = np.array(model.get_observer_labels(), dtype=np.int32)
        return obs_points, labels, radii
    return np.array([]).reshape(0, 2), np.array([], dtype=np.int32), None


def plot_clustering_with_observers(
    x, y_true, y_pred, obs_points, obs_labels,
    title_prefix="", out_path=None, suptitle=None,
    obs_final_threshold_radii=None,
):
    """Three panels: Ground Truth, Predictions, Observer set. 2D only.
    If obs_final_threshold_radii is given (same length as obs_points), each observer
    is drawn with a transparent circle of that radius (final threshold) behind the dot."""
    x_plot = x[:, :2] if x.shape[1] >= 2 else x
    n_plots = 3 if obs_points.size > 0 else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    def style_for_label(lab):
        if lab < 0:
            return "black", "x"
        idx = lab % len(CLASS_STYLE)
        return CLASS_STYLE[idx][0], CLASS_STYLE[idx][1]

    # Ground Truth
    ax = axes[0]
    for lab in np.unique(y_true):
        color, marker = style_for_label(lab)
        if lab == -1:
            ax.scatter(
                x_plot[y_true == -1, 0], x_plot[y_true == -1, 1],
                c=color, s=20, marker=marker, label="Outlier"
            )
        else:
            ax.scatter(
                x_plot[y_true == lab, 0], x_plot[y_true == lab, 1],
                c=color, s=15, marker=marker, label=f"GT {lab}"
            )
    ax.set_title(f"{title_prefix}Ground Truth")
    ax.set_xlabel("f0")
    ax.set_ylabel("f1")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Predictions
    ax = axes[1]
    for lab in np.unique(y_pred):
        color, marker = style_for_label(lab)
        if lab == -1:
            ax.scatter(
                x_plot[y_pred == -1, 0], x_plot[y_pred == -1, 1],
                c=color, s=20, marker=marker, label="Outlier"
            )
        else:
            ax.scatter(
                x_plot[y_pred == lab, 0], x_plot[y_pred == lab, 1],
                c=color, s=15, marker=marker, label=f"Pred {lab}"
            )
    ax.set_title(f"{title_prefix}Vorhersagen")
    ax.set_xlabel("f0")
    ax.set_ylabel("f1")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Observer set (optional: circles = final threshold radius, then dots)
    if n_plots == 3 and obs_points.size > 0:
        ax = axes[2]
        obs_2d = obs_points[:, :2] if obs_points.shape[1] >= 2 else obs_points
        # Draw transparent circles first (final threshold radius) if available
        if obs_final_threshold_radii is not None and len(obs_final_threshold_radii) == len(obs_2d):
            for i in range(len(obs_2d)):
                lab = obs_labels[i]
                color, _ = style_for_label(lab)
                r = float(obs_final_threshold_radii[i])
                circ = Circle(obs_2d[i], r, facecolor=color, edgecolor="none", alpha=0.2, zorder=1)
                ax.add_patch(circ)
        # Then dots on top
        for lab in np.unique(obs_labels):
            color, marker = style_for_label(lab)
            if lab == -1:
                ax.scatter(
                    obs_2d[obs_labels == -1, 0], obs_2d[obs_labels == -1, 1],
                    c=color, s=40, marker=marker, label="Outlier", zorder=3
                )
            else:
                ax.scatter(
                    obs_2d[obs_labels == lab, 0], obs_2d[obs_labels == lab, 1],
                    c=color, s=40, marker=marker, edgecolors="black", linewidths=0.5,
                    label=f"Observer {lab}", zorder=3
                )
        ax.set_title(f"{title_prefix}Observer-Set (Modell)")
        ax.set_xlabel("f0")
        ax.set_ylabel("f1")
        ax.legend(loc="best", fontsize=8)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120)
        print(f"  Grafik gespeichert: {out_path}")
    else:
        plt.show()
    plt.close(fig)


# --- Connectivity debug (Distanzmatrix → Connectivity → Python-CC vs. Rust-CC) ---

def _build_final_thresholds(observers_data, active_indices, zeta, global_threshold):
    """Finale Schwellwerte pro aktivem Observer (Position = Index in active_indices)."""
    index_to_local = {obs[5]: obs[4] for obs in observers_data}
    return [
        zeta * index_to_local[idx] + (1.0 - zeta) * global_threshold
        for idx in active_indices
    ]


def _build_distance_matrix_active(active_indices, distance_matrix_dict):
    """Vollständige Distanzmatrix nur für aktive Observer (n x n). Position i = active_indices[i]."""
    n = len(active_indices)
    idx_to_pos = {int(idx): i for i, idx in enumerate(active_indices)}
    D = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(D, 0.0)
    for i, idx_i in enumerate(active_indices):
        idx_i = int(idx_i)
        if idx_i not in distance_matrix_dict:
            continue
        for (j_idx, d) in distance_matrix_dict[idx_i]:
            j_idx = int(j_idx)
            if j_idx in idx_to_pos:
                j = idx_to_pos[j_idx]
                D[i, j] = D[j, i] = float(d)
    return D


def _build_connectivity_matrix(D, final_thresholds):
    """Connectivity-Matrix: Kante (i,j) gdw. d_ij < final_i und d_ij < final_j (wie in Rust)."""
    n = D.shape[0]
    C = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d_ij = D[i, j]
            if d_ij < final_thresholds[i] and d_ij < final_thresholds[j]:
                C[i, j] = 1
    return C


def _connected_components_from_adjacency(C):
    """Connected Components per BFS aus Adjazenzmatrix C (Positionen 0..n-1)."""
    n = C.shape[0]
    visited = [False] * n
    components = []
    for start in range(n):
        if visited[start]:
            continue
        comp = []
        stack = [start]
        visited[start] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in range(n):
                if (C[u, v] or C[v, u]) and not visited[v]:
                    visited[v] = True
                    stack.append(v)
        if comp:
            components.append(comp)
    return components


def run_connectivity_debug(model):
    """
    Holt Distanzmatrix + finale Schwellwerte, baut Connectivity-Matrix,
    berechnet Connected Components in Python und vergleicht mit Rust (get_connected_components_debug).
    Gibt (match, components_py, components_rust) zurück und gibt Ergebnis auf stdout aus.
    """
    if not hasattr(model, "get_all_observer_data_for_testing") or not hasattr(
        model, "get_connected_components_debug"
    ):
        print("  Connectivity-Debug: Modell unterstützt get_all_observer_data_for_testing oder get_connected_components_debug nicht.")
        return None, None, None

    result = model.get_all_observer_data_for_testing()
    observers_data_list, global_threshold, active_indices_list, distance_matrix_dict = result
    observers_data = [(obs[0], obs[1], obs[2], obs[3], obs[4], obs[5]) for obs in observers_data_list]
    active_indices = [int(x) for x in active_indices_list]
    distance_matrix_dict = {
        int(k): [(int(n[0]), float(n[1])) for n in v]
        for k, v in distance_matrix_dict.items()
    }
    zeta = float(model.zeta)
    global_threshold = float(global_threshold)

    final_thresholds = _build_final_thresholds(
        observers_data, active_indices, zeta, global_threshold
    )
    D = _build_distance_matrix_active(active_indices, distance_matrix_dict)
    C = _build_connectivity_matrix(D, final_thresholds)
    components_by_position = _connected_components_from_adjacency(C)
    components_py_all = [[active_indices[p] for p in comp] for comp in components_by_position]

    # Rust entfernt kleine Cluster (min_cluster_size). Für fairen Vergleich: Python genauso filtern.
    min_cluster_size = int(getattr(model, "min_cluster_size", 1))
    components_py = [c for c in components_py_all if len(c) >= min_cluster_size]
    components_rust = model.get_connected_components_debug()
    components_rust = [list(c) for c in components_rust]

    sets_py = {frozenset(c) for c in components_py}
    sets_rust = {frozenset(c) for c in components_rust}
    match = sets_py == sets_rust

    print("\n  --- Connectivity-Debug (Distanz → Connectivity → Python-CC vs. Rust-CC) ---")
    print(f"  Aktive Observer: {len(active_indices)}")
    print(f"  min_cluster_size: {min_cluster_size}")
    print(f"  Komponenten (Python, ungefiltert): {len(components_py_all)}; gefiltert: {len(components_py)}; (Rust): {len(components_rust)}")
    if components_py_all:
        sizes_all = sorted(len(c) for c in components_py_all)
        print(f"  Python sizes (ungefiltert): {sizes_all}")
    if components_py:
        sizes_f = sorted(len(c) for c in components_py)
        print(f"  Python sizes (gefiltert): {sizes_f}")
    if components_rust:
        sizes_r = sorted(len(c) for c in components_rust)
        print(f"  Rust sizes: {sizes_r}")
    if match:
        print("  Ergebnis: Python- und Rust-Connected-Components stimmen überein.")
    else:
        print("  Ergebnis: Abweichung – Python- und Rust-Connected-Components stimmen NICHT überein.")
    print("  ---")
    return match, components_py, components_rust


# --- Distance/threshold debug (Rust-Distanzmatrix + Thresholds vs. Python-Recalc) ---

def _pairwise_distances(X, metric="euclidean", minkowski_p=None):
    """Pairwise distances for X (n,d)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={X.shape}")

    if metric == "euclidean":
        diff = X[:, None, :] - X[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1))
    if metric == "manhattan":
        diff = np.abs(X[:, None, :] - X[None, :, :])
        return np.sum(diff, axis=-1)
    if metric == "chebyshev":
        diff = np.abs(X[:, None, :] - X[None, :, :])
        return np.max(diff, axis=-1)
    if metric == "minkowski":
        if minkowski_p is None:
            raise ValueError("minkowski_p must be provided for minkowski metric")
        p = float(minkowski_p)
        diff = np.abs(X[:, None, :] - X[None, :, :]) ** p
        return np.sum(diff, axis=-1) ** (1.0 / p)
    raise ValueError(f"unknown metric: {metric}")


def debug_distance_and_thresholds(
    model,
    metric="euclidean",
    minkowski_p=None,
    atol=1e-12,
    rtol=1e-9,
    verbose=True,
):
    """
    Prüft:
    - Distanzmatrix (Rust exportiert, aktiv×aktiv) vs. Python Pairwise-Recalc
    - lokale Thresholds h_ω (Rust stored in Observer.local_threshold) vs. Python chi-th NN (aktiv)
    - global_threshold (Rust stored via model.get_global_threshold() falls vorhanden) vs. Python mean(h_ω)
    """
    if not hasattr(model, "get_all_observer_data_for_testing"):
        raise RuntimeError("Model has no get_all_observer_data_for_testing()")

    observers_data_list, global_threshold_export, active_indices_list, distance_matrix_dict = (
        model.get_all_observer_data_for_testing()
    )

    # observers_data rows: (data, observations, age, is_active, local_threshold, index)
    observers_data = [
        (obs[0], float(obs[4]), int(obs[5])) for obs in observers_data_list
    ]
    idx_to_data = {idx: np.asarray(data, dtype=np.float64) for (data, _h, idx) in observers_data}
    idx_to_local_rust = {idx: float(h) for (_data, h, idx) in observers_data}

    active_indices = [int(x) for x in active_indices_list]
    n = len(active_indices)
    idx_to_pos = {idx: i for i, idx in enumerate(active_indices)}

    # Rust distances (active×active)
    distance_matrix_dict = {
        int(k): [(int(nbr[0]), float(nbr[1])) for nbr in v]
        for k, v in distance_matrix_dict.items()
    }
    D_rust = np.full((n, n), np.nan, dtype=np.float64)
    np.fill_diagonal(D_rust, 0.0)
    for idx_i in active_indices:
        i = idx_to_pos[idx_i]
        for (idx_j, d) in distance_matrix_dict.get(idx_i, []):
            if idx_j in idx_to_pos:
                j = idx_to_pos[idx_j]
                D_rust[i, j] = D_rust[j, i] = float(d)

    # Python distances (recalc)
    X = np.vstack([idx_to_data[idx] for idx in active_indices])
    D_py = _pairwise_distances(X, metric=metric, minkowski_p=minkowski_p)
    np.fill_diagonal(D_py, 0.0)

    # Distance comparison
    mask = ~np.isnan(D_rust)
    abs_diff = np.abs(D_rust[mask] - D_py[mask])
    max_abs = float(abs_diff.max()) if abs_diff.size else 0.0
    ok_dist = np.allclose(D_rust[mask], D_py[mask], atol=atol, rtol=rtol) if abs_diff.size else True

    # Local thresholds: Rust stored
    chi = int(getattr(model, "chi"))
    local_rust = np.array([idx_to_local_rust[idx] for idx in active_indices], dtype=np.float64)

    # Local thresholds: Python recompute = chi-th nearest neighbor among active (exclude self)
    local_py = np.zeros(n, dtype=np.float64)
    for i in range(n):
        row = np.sort(D_py[i, :])
        row = row[row > 0.0]  # remove self
        if row.size == 0:
            local_py[i] = np.inf
        else:
            k = min(chi, row.size)
            local_py[i] = row[k - 1]

    diff_local = np.abs(local_rust - local_py)
    max_local = float(np.nanmax(diff_local)) if diff_local.size else 0.0
    ok_local = np.allclose(local_rust, local_py, atol=atol, rtol=rtol)

    # Global threshold: prefer Rust stored value if available
    global_rust_stored = None
    if hasattr(model, "get_global_threshold"):
        global_rust_stored = float(model.get_global_threshold())
    global_export = float(global_threshold_export)
    global_py = float(np.mean(local_py[np.isfinite(local_py)])) if np.any(np.isfinite(local_py)) else float("inf")

    # Report
    if verbose:
        print("\n  --- Distance/Threshold Debug (Rust vs. Python) ---")
        print(f"  n_active={n}, metric={metric}, chi={chi}")
        print(f"  Distance: ok={ok_dist}, max_abs_diff={max_abs:.3e} (atol={atol}, rtol={rtol})")
        if abs_diff.size:
            worst_flat = int(np.argmax(abs_diff))
            ii, jj = np.argwhere(mask)[worst_flat]
            print(f"    worst pair: pos({ii},{jj}) idx({active_indices[ii]},{active_indices[jj]}) "
                  f"D_rust={D_rust[ii,jj]:.15g} D_py={D_py[ii,jj]:.15g}")
        missing = int(np.isnan(D_rust).sum())
        if missing:
            print(f"  Distance: WARNING missing entries in D_rust: {missing} of {n*n}")

        print(f"  Local thresholds: ok={ok_local}, max_abs_diff={max_local:.3e}")
        if diff_local.size:
            wi = int(np.nanargmax(diff_local))
            print(f"    worst i: pos({wi}) idx({active_indices[wi]}) h_rust={local_rust[wi]:.15g} h_py={local_py[wi]:.15g}")

        if global_rust_stored is not None:
            print(f"  Global threshold: rust_stored={global_rust_stored:.15g}, python_mean={global_py:.15g}, abs_diff={abs(global_rust_stored-global_py):.3e}")
            print(f"  Global threshold (export/recomputed): export={global_export:.15g}, abs_diff(export-python)={abs(global_export-global_py):.3e}")
        else:
            print(f"  Global threshold: export={global_export:.15g}, python_mean={global_py:.15g}, abs_diff={abs(global_export-global_py):.3e}")

        if global_rust_stored is not None:
            zeta = float(getattr(model, "zeta"))
            final_rust = zeta * local_rust + (1.0 - zeta) * global_rust_stored
            final_py = zeta * local_py + (1.0 - zeta) * global_py
            max_final = float(np.nanmax(np.abs(final_rust - final_py))) if n else 0.0
            print(f"  Final thresholds: max_abs_diff={max_final:.3e} (zeta={zeta})")
        print("  ---")

    return {
        "ok_dist": bool(ok_dist),
        "max_abs_dist": max_abs,
        "ok_local": bool(ok_local),
        "max_abs_local": max_local,
        "global_rust_stored": global_rust_stored,
        "global_export": global_export,
        "global_py": global_py,
    }
