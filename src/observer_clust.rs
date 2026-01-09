use std::cmp::Ordering;
use std::collections::HashMap;

use crate::observer::ObserverSet;
use crate::utils::compute_distance;

/// Clustering-Erweiterungen für ObserverSet
impl ObserverSet {
    /// Führt Clustering auf den aktiven Observers durch
    /// chi: Anzahl der nächsten Observer für lokale Thresholds
    /// zeta: Mixing-Parameter für globale/lokale Thresholds
    /// min_cluster_size: minimale Clustergröße
    /// Verwendet distance_metric und minkowski_p aus ObserverSet
    pub fn learn_cluster(&mut self, chi: usize, zeta: f64, min_cluster_size: usize) {
        // Hole aktive Observer
        let active_observers: Vec<Vec<f64>> = self
            .iter_observers(true)
            .map(|obs| obs.data.clone())
            .collect();

        let n = active_observers.len();
        if n == 0 {
            return;
        }

        // Hole aktive Observer-Indizes
        let active_indices: Vec<usize> = self.iter_observers(true).map(|obs| obs.index).collect();

        if active_indices.len() != n {
            return; // Mismatch
        }

        // Schritt 1: Berechne lokale Cutoff-Thresholds h_ω
        let mut local_thresholds = vec![0.0; n];

        // Verwende Tree für Nearest Neighbor Search wenn möglich
        {
            self.ensure_spatial_tree(false);
            let tree_opt = self.get_spatial_tree(false);
            if let Some(ref tree_ref) = tree_opt {
                if tree_ref.is_some() {
                    // Verwende Tree für Nearest Neighbor Search
                    for (idx, observer) in active_observers.iter().enumerate() {
                        let chi_actual = (chi + 1).min(n); // +1 weil wir den Observer selbst ausschließen
                        let distances =
                            self.search_k_nearest_distances(observer, chi_actual, false);
                        if distances.len() >= chi {
                            local_thresholds[idx] = distances[chi - 1];
                        } else if !distances.is_empty() {
                            local_thresholds[idx] = distances[distances.len() - 1];
                        } else {
                            local_thresholds[idx] = f64::INFINITY;
                        }
                    }
                } else {
                    // Fallback zu Brute-Force
                    self.compute_local_thresholds_brute_force(
                        &active_observers,
                        &mut local_thresholds,
                        chi,
                    );
                }
            } else {
                // Fallback zu Brute-Force
                self.compute_local_thresholds_brute_force(
                    &active_observers,
                    &mut local_thresholds,
                    chi,
                );
            }
        } // tree_opt wird hier freigegeben

        // Schritt 2: Berechne globalen Density-Threshold h
        let global_threshold: f64 =
            local_thresholds.iter().sum::<f64>() / local_thresholds.len() as f64;

        // Schritt 3: Berechne finale Thresholds mit Mixture-Modell: h'_ω = ζ·h_ω + (1-ζ)·h
        let final_thresholds: Vec<f64> = local_thresholds
            .iter()
            .map(|&h_omega| zeta * h_omega + (1.0 - zeta) * global_threshold)
            .collect();

        // Schritt 4 & 5: Finde Connected Components mit DFS und weise Labels zu
        let mut visited = vec![false; n];
        let mut current_label = 0i32;

        // DFS für jeden unbesuchten Observer
        for start_idx in 0..n {
            if !visited[start_idx] {
                let mut stack = vec![start_idx];
                visited[start_idx] = true;
                // Setze Label direkt im Observer
                if let Some(&observer_idx) = active_indices.get(start_idx) {
                    self.update_label(observer_idx, Some(current_label));
                }

                while let Some(current_idx) = stack.pop() {
                    // Finde alle verbundenen Nachbarn
                    for neighbor_idx in 0..n {
                        if neighbor_idx != current_idx && !visited[neighbor_idx] {
                            // Berechne Distanz
                            let dist = compute_distance(
                                &active_observers[current_idx],
                                &active_observers[neighbor_idx],
                                self.get_distance_metric(),
                                self.get_minkowski_p(),
                            );
                            // Zwei Observer sind verbunden wenn d(ν,ω) < h'_ω UND d(ν,ω) < h'_ν
                            if dist < final_thresholds[current_idx]
                                && dist < final_thresholds[neighbor_idx]
                            {
                                visited[neighbor_idx] = true;
                                // Setze Label direkt im Observer
                                if let Some(&observer_idx) = active_indices.get(neighbor_idx) {
                                    self.update_label(observer_idx, Some(current_label));
                                }
                                stack.push(neighbor_idx);
                            }
                        }
                    }
                }
                current_label += 1;
            }
        }

        // Schritt 6: Entferne kleine Cluster
        // Zähle Größe jedes Clusters
        let mut cluster_sizes: HashMap<i32, usize> = HashMap::new();
        for obs in self.iter_observers(true) {
            if let Some(label) = obs.label {
                if label >= 0 {
                    *cluster_sizes.entry(label).or_insert(0) += 1;
                }
            }
        }

        // Sammle Indizes von Observers, die entfernt werden sollen
        let mut indices_to_remove: Vec<usize> = Vec::new();
        for obs in self.iter_observers(true) {
            if let Some(label) = obs.label {
                if label >= 0 {
                    if let Some(&size) = cluster_sizes.get(&label) {
                        if size < min_cluster_size {
                            indices_to_remove.push(obs.index);
                        }
                    }
                }
            }
        }

        // Entferne Labels von Observers in zu kleinen Clustern
        for idx in indices_to_remove {
            self.update_label(idx, None);
        }

        // Baue Tree für aktive Observer
        self.ensure_spatial_tree(true);
    }

    /// Berechnet lokale Thresholds mit Brute-Force (Fallback)
    fn compute_local_thresholds_brute_force(
        &self,
        active_observers: &[Vec<f64>],
        local_thresholds: &mut [f64],
        chi: usize,
    ) {
        for (idx, observer) in active_observers.iter().enumerate() {
            let mut distances: Vec<f64> = active_observers
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != idx)
                .map(|(_, other)| {
                    compute_distance(
                        observer,
                        other,
                        self.get_distance_metric(),
                        self.get_minkowski_p(),
                    )
                })
                .collect();

            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

            let chi_actual = chi.min(distances.len());
            if chi_actual > 0 {
                local_thresholds[idx] = distances[chi_actual - 1];
            } else {
                local_thresholds[idx] = f64::INFINITY;
            }
        }
    }
}
