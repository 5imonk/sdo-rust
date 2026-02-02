"""Shared helpers for SDOstreamclust tests and visualization. Re-exports from sdoclust.common."""

import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PYTHON_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

from sdoclust.common import get_observers_and_labels, plot_clustering_with_observers

__all__ = ["get_observers_and_labels", "plot_clustering_with_observers"]
