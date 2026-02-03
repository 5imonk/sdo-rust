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
