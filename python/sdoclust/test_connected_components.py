#!/usr/bin/env python3
"""
Test-Skript zur Überprüfung der Connected Components Berechnung.
Vergleicht Rust-Implementierung mit Python-Nachbau.
"""

import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _THIS_DIR)

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict, deque

try:
    from sdo import SDOclust
except ImportError as e:
    print(f"Fehler: sdo-Modul nicht gefunden: {e}")
    print("Bitte im Projektroot 'maturin develop' ausführen.")
    sys.exit(1)


def compute_euclidean_distance(p1, p2):
    """Berechnet euklidische Distanz zwischen zwei Punkten."""
    return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))


def compute_local_threshold_python(observer_idx, observers_data, active_indices, distance_matrix, chi):
    """Berechnet lokalen Threshold für einen Observer in Python."""
    if observer_idx not in distance_matrix:
        return float('inf')
    
    distances = distance_matrix[observer_idx]
    # Sortiere nach Distanz
    sorted_distances = sorted(distances, key=lambda x: x[1])
    
    # Zähle aktive Observer bis chi erreicht ist
    found = 0
    for neighbor_idx, dist in sorted_distances:
        if neighbor_idx in active_indices:
            found += 1
            if found == chi:
                return dist
    
    # Wenn weniger als chi gefunden, gib letzte aktive Distanz zurück
    if found > 0:
        for neighbor_idx, dist in reversed(sorted_distances):
            if neighbor_idx in active_indices:
                return dist
    
    return float('inf')


def dfs_python(start_idx, observers_data, active_indices, distance_matrix, zeta, global_threshold, visited):
    """DFS für Connected Components in Python (nach Rust-Logik)."""
    component = set()
    stack = [start_idx]
    
    visited.add(start_idx)
    component.add(start_idx)
    
    # Finde Observer-Daten
    observer_data_map = {obs[5]: obs for obs in observers_data}
    
    while stack:
        current_idx = stack.pop()
        
        # Hole lokalen Threshold für aktuellen Observer
        h_omega_current = observer_data_map[current_idx][4]  # local_threshold
        final_threshold_current = zeta * h_omega_current + (1.0 - zeta) * global_threshold
        
        # Hole Nachbarn aus Distanzmatrix
        if current_idx not in distance_matrix:
            continue
            
        neighbors = distance_matrix[current_idx]
        
        # Iteriere über alle Nachbarn, die näher als final_threshold_current sind
        for neighbor_idx, dist in neighbors:
            if neighbor_idx in visited or neighbor_idx not in active_indices:
                continue
            
            # Hole lokalen Threshold für Nachbarn
            h_omega_neighbor = observer_data_map[neighbor_idx][4]  # local_threshold
            final_threshold_neighbor = zeta * h_omega_neighbor + (1.0 - zeta) * global_threshold
            
            # Zwei Observer sind verbunden wenn d(ν,ω) < h'_ω UND d(ν,ω) < h'_ν
            if dist < final_threshold_neighbor:
                visited.add(neighbor_idx)
                component.add(neighbor_idx)
                stack.append(neighbor_idx)
    
    return component


def find_connected_components_python(observers_data, active_indices, distance_matrix, zeta, global_threshold):
    """Findet alle Connected Components in Python (nach Rust-Logik)."""
    connected_components = []
    visited = set()
    
    for start_idx in active_indices:
        if start_idx in visited:
            continue
        
        component = dfs_python(start_idx, observers_data, active_indices, distance_matrix, zeta, global_threshold, visited)
        if component:
            connected_components.append(component)
    
    return connected_components


def test_connected_components():
    """Test Connected Components Berechnung."""
    print("=" * 60)
    print("Connected Components Test")
    print("=" * 60)
    
    # Erstelle Test-Daten
    np.random.seed(42)
    n_per_cluster = 40
    centers = [[0.25, 0.25], [0.75, 0.75]]
    
    xs = []
    for center in centers:
        x = np.random.randn(n_per_cluster, 2).astype(np.float64) * 0.12 + center
        xs.append(x)
    x = np.vstack(xs)
    x = MinMaxScaler().fit_transform(x)
    
    print(f"Daten: {x.shape[0]} Punkte, {x.shape[1]} Dimensionen")
    
    # Trainiere SDOclust
    k = 30
    x_param = 5
    rho = 0.2
    chi = 4
    zeta = 0.5
    min_cluster_size = 2
    
    model = SDOclust(k=k, x=x_param, rho=rho, chi=chi, zeta=zeta, min_cluster_size=min_cluster_size)
    model.learn(x)
    
    print(f"Modell trainiert: k={k}, x={x_param}, rho={rho}, chi={chi}, zeta={zeta}, min_cluster_size={min_cluster_size}")
    print(f"Anzahl Cluster (Rust): {model.n_clusters()}")
    
    # Extrahiere Observer-Daten
    result = model.get_all_observer_data_for_testing()
    observers_data_list, global_threshold, active_indices_list, distance_matrix_dict = result
    
    # Konvertiere zu Python-Listen/Dict
    observers_data = [(obs[0], obs[1], obs[2], obs[3], obs[4], obs[5]) for obs in observers_data_list]
    active_indices = list(active_indices_list)
    distance_matrix = {int(k): [(int(n[0]), float(n[1])) for n in v] for k, v in distance_matrix_dict.items()}
    
    print(f"\nObserver-Daten extrahiert:")
    print(f"  Anzahl Observer: {len(observers_data)}")
    print(f"  Aktive Observer: {len(active_indices)}")
    print(f"  Global Threshold: {global_threshold:.6f}")
    print(f"  Distanzmatrix-Einträge: {len(distance_matrix)}")
    
    # Zeige einige Observer-Daten
    print(f"\nErste 5 aktive Observer:")
    for i, idx in enumerate(active_indices[:5]):
        obs = next(obs for obs in observers_data if obs[5] == idx)
        print(f"  Observer {idx}: data={obs[0][:2]}, observations={obs[1]:.2f}, local_threshold={obs[4]:.6f}, is_active={obs[3]}")
    
    # Berechne Connected Components in Python
    print(f"\nBerechne Connected Components in Python...")
    python_components = find_connected_components_python(
        observers_data, active_indices, distance_matrix, zeta, global_threshold
    )
    
    print(f"Python Connected Components: {len(python_components)}")
    for i, comp in enumerate(python_components):
        print(f"  Component {i}: {len(comp)} Observer: {sorted(list(comp))[:10]}{'...' if len(comp) > 10 else ''}")
    
    # Entferne kleine Cluster
    python_components_filtered = [comp for comp in python_components if len(comp) >= min_cluster_size]
    print(f"\nNach Filterung (min_cluster_size={min_cluster_size}): {len(python_components_filtered)} Components")
    
    # Zeige Details
    print(f"\nDetaillierte Analyse:")
    print(f"  Anzahl aktive Observer: {len(active_indices)}")
    print(f"  Anzahl Components (vor Filterung): {len(python_components)}")
    print(f"  Anzahl Components (nach Filterung): {len(python_components_filtered)}")
    
    # Zeige lokale Thresholds
    print(f"\nLokale Thresholds (erste 10 aktive Observer):")
    for idx in active_indices[:10]:
        obs = next(obs for obs in observers_data if obs[5] == idx)
        final_threshold = zeta * obs[4] + (1.0 - zeta) * global_threshold
        print(f"  Observer {idx}: h_omega={obs[4]:.6f}, final_threshold={final_threshold:.6f}")
    
    # Zeige Distanzmatrix-Details
    print(f"\nDistanzmatrix-Details (erste 3 aktive Observer):")
    for idx in active_indices[:3]:
        if idx in distance_matrix:
            neighbors = distance_matrix[idx]
            print(f"  Observer {idx}: {len(neighbors)} Nachbarn")
            if neighbors:
                print(f"    Erste 5: {neighbors[:5]}")
        else:
            print(f"  Observer {idx}: Keine Distanzen")
    
    print("\n" + "=" * 60)
    print("Test abgeschlossen")
    print("=" * 60)
    
    return python_components_filtered


if __name__ == "__main__":
    test_connected_components()
