use std::collections::{HashMap, HashSet};

use crate::obset::ObserverSet;

/// Clustering-Erweiterungen für ObserverSet
impl ObserverSet {
    /// Führt Clustering auf den aktiven Observers durch
    /// chi: Anzahl der nächsten Observer für lokale Thresholds
    /// zeta: Mixing-Parameter für globale/lokale Thresholds
    /// min_cluster_size: minimale Clustergröße
    /// write_labels: Wenn true, werden Labels in obs.label geschrieben; sonst nur temporär für interne Verwendung
    /// Gibt die Cluster-Map zurück: HashMap<label, HashSet<observer_indices>>
    /// Verwendet distance_metric und minkowski_p aus ObserverSet
    pub fn learn_cluster(
        &mut self,
        chi: usize,
        zeta: f64,
        min_cluster_size: usize,
        write_labels: bool,
    ) -> HashMap<i32, HashSet<usize>> {
        // Hole aktive Observer
        let active_observers: Vec<Vec<f64>> = self
            .iter_observers(true)
            .map(|obs| obs.data.clone())
            .collect();

        let n = active_observers.len();
        if n == 0 {
            return HashMap::new();
        }

        // Hole aktive Observer-Indizes
        let active_indices: Vec<usize> = self.iter_observers(true).map(|obs| obs.index).collect();

        if active_indices.len() != n {
            return HashMap::new(); // Mismatch
        }

        // Schritt 1: Berechne lokale Cutoff-Thresholds h_ω
        // Verwende gecachte Distanzlisten für effiziente Berechnung
        if self.distance_lists.is_empty() {
            // Wenn keine Distanzlisten vorhanden, baue sie neu auf
            self.rebuild_distance_lists();
        }

        // Sammle lokale Thresholds für globale Berechnung mit Cache
        let active_indices: Vec<usize> = self.iter_observers(true).map(|obs| obs.index).collect();

        // Validiere Cache vor Batch-Berechnung
        self.validate_threshold_cache();

        let local_thresholds: Vec<f64> = active_indices
            .iter()
            .map(|&idx| self.compute_local_threshold_cached(idx, chi))
            .collect();

        // Schritt 2: Berechne globalen Density-Threshold h
        let global_threshold: f64 = if !local_thresholds.is_empty() {
            local_thresholds.iter().sum::<f64>() / local_thresholds.len() as f64
        } else {
            f64::INFINITY
        };

        // Schritt 3: Entferne alle Labels von allen Observern
        let all_indices: Vec<usize> = self.iter_observers(false).map(|obs| obs.index).collect();
        let all_indices_clone = all_indices.clone();
        for index in all_indices {
            self.update_label(index, None);
        }

        // Schritt 4 & 5: Finde Connected Components mit DFS und weise Labels zu
        // Iteriere über alle aktiven Observer (nach Index)
        let active_indices: Vec<usize> = self.iter_observers(true).map(|obs| obs.index).collect();

        let mut current_label = 0i32;

        // DFS für jeden unbesuchten aktiven Observer
        for &start_index in &active_indices {
            // Prüfe ob Observer bereits ein Label hat (besucht)
            let start_observer = self.observers_by_index.get(&start_index).unwrap();
            if start_observer.label.is_some() {
                continue; // Bereits besucht
            }

            // Starte neue Connected Component
            let mut stack = vec![start_index];
            self.update_label(start_index, Some(current_label));

            while let Some(current_index) = stack.pop() {
                // Berechne finalen Threshold für den aktuellen Observer
                let final_threshold_current =
                    self.compute_final_threshold(current_index, zeta, chi, global_threshold);

                // Sammle potenzielle Nachbarn aus der Distanzliste (nur die, die näher als final_threshold_current sind)
                // Optimiert: Binary Search + early termination
                let potential_neighbors: Vec<(usize, f64)> = {
                    if let Some(distance_list) = self.distance_lists.get(&current_index) {
                        let start_pos =
                            distance_list.find_threshold_position(final_threshold_current);
                        distance_list.distances[start_pos..]
                            .iter()
                            .filter(|(neighbor_idx, dist)| {
                                *dist < final_threshold_current
                                    && *neighbor_idx != current_index
                                    && self.is_active(*neighbor_idx)
                            })
                            .map(|(idx, dist)| (*idx, *dist))
                            .collect()
                    } else {
                        Vec::new()
                    }
                };

                // Verarbeite potenzielle Nachbarn
                for (neighbor_index, dist) in potential_neighbors {
                    // Prüfe ob Nachbar bereits besucht (hat Label)
                    let neighbor_observer = self.observers_by_index.get(&neighbor_index).unwrap();
                    if neighbor_observer.label.is_some() {
                        continue; // Bereits besucht
                    }

                    // Berechne finalen Threshold für den Nachbar
                    let final_threshold_neighbor =
                        self.compute_final_threshold(neighbor_index, zeta, chi, global_threshold);

                    // Zwei Observer sind verbunden wenn d(ν,ω) < h'_ω UND d(ν,ω) < h'_ν
                    if dist < final_threshold_neighbor {
                        self.update_label(neighbor_index, Some(current_label));
                        stack.push(neighbor_index);
                    }
                }
            }
            current_label += 1;
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

        // Baue Cluster-Map aus Labels
        let mut cluster_map: HashMap<i32, HashSet<usize>> = HashMap::new();
        for obs in self.iter_observers(true) {
            if let Some(label) = obs.label {
                if label >= 0 {
                    cluster_map.entry(label).or_default().insert(obs.index);
                }
            }
        }

        // Wenn write_labels=false, entferne Labels wieder (waren nur temporär)
        if !write_labels {
            for index in all_indices_clone {
                self.update_label(index, None);
            }
        }

        cluster_map
    }

    /// Setze Cache als gültig (nach vollständiger Aktualisierung)
    pub fn validate_threshold_cache(&mut self) {
        self.cache_valid = true;
    }

    /// Berechne lokalen Threshold mit Cache-Unterstützung
    pub fn compute_local_threshold_cached(&mut self, index: usize, chi: usize) -> f64 {
        let cache_key = (index, chi);

        // Prüfe Cache
        if self.cache_valid {
            if let Some(&cached_threshold) = self.local_threshold_cache.get(&cache_key) {
                return cached_threshold;
            }
        }

        // Berechne und cache
        let threshold = self.compute_local_threshold_impl(index, chi);
        self.local_threshold_cache.insert(cache_key, threshold);
        threshold
    }

    /// Implementierung der lokalen Threshold-Berechnung
    pub(crate) fn compute_local_threshold_impl(&self, index: usize, chi: usize) -> f64 {
        if let Some(list) = self.distance_lists.get(&index) {
            let mut found = 0;

            // Iteriere über sortierte Distanzen bis chi aktive Observer gefunden
            for (target_idx, dist) in &list.distances {
                if self.is_active(*target_idx) {
                    found += 1;
                    if found == chi {
                        return *dist;
                    }
                }
            }

            // Wenn weniger als chi aktive Observer gefunden, letzte Distanz verwenden
            if found > 0 {
                // Finde letzte aktive Distanz
                for (target_idx, dist) in list.distances.iter().rev() {
                    if self.is_active(*target_idx) {
                        return *dist;
                    }
                }
            }
        }

        f64::INFINITY
    }

    /// Berechnet den finalen Threshold für einen Observer mit Mixture-Modell: h'_ω = ζ·h_ω + (1-ζ)·h
    fn compute_final_threshold(
        &self,
        index: usize,
        zeta: f64,
        chi: usize,
        global_threshold: f64,
    ) -> f64 {
        let local_threshold = self.compute_local_threshold_impl(index, chi);
        zeta * local_threshold + (1.0 - zeta) * global_threshold
    }
}
