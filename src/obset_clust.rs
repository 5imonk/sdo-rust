use std::collections::{HashMap, HashSet};

use crate::obset::ObserverSet;

/// Clustering-Erweiterungen für ObserverSet
impl ObserverSet {
    /// DFS-Helper für Connected Components
    /// Findet alle Observer in derselben Connected Component wie start_index
    fn dfs(
        &self,
        start_index: usize,
        zeta: f64,
        global_threshold: f64,
        active_set: &HashSet<usize>,
        visited_indices: &mut HashSet<usize>,
    ) -> HashSet<usize> {
        let mut connected_component = HashSet::new();
        let mut stack = vec![start_index];

        visited_indices.insert(start_index);
        connected_component.insert(start_index);

        while let Some(current_index) = stack.pop() {
            // Hole h_omega für den aktuellen Observer
            let h_omega_current = self.get(current_index).unwrap().local_threshold;
            let final_threshold_current = zeta * h_omega_current + (1.0 - zeta) * global_threshold;

            // Sammle potenzielle Nachbarn aus der Distanzliste
            if let Some(distance_list) = self.distance_matrix.get(current_index) {
                let end_pos = distance_list.find_threshold_position(final_threshold_current);

                // Wir müssen über alle Nachbarn iterieren, die näher als final_threshold_current sind
                let neighbors: Vec<(usize, f64)> = distance_list.distances[..end_pos].to_vec();

                for (neighbor_index, dist) in neighbors {
                    // Strict Top-N Semantik: nur Indizes aus active_set zulassen
                    if visited_indices.contains(&neighbor_index)
                        || !active_set.contains(&neighbor_index)
                    {
                        continue;
                    }

                    // Hole h_omega für den Nachbarn
                    let h_omega_neighbor = self.get(neighbor_index).unwrap().local_threshold;
                    let final_threshold_neighbor =
                        zeta * h_omega_neighbor + (1.0 - zeta) * global_threshold;

                    // Zwei Observer sind verbunden wenn d(ν,ω) < h'_ω UND d(ν,ω) < h'_ν
                    if dist < final_threshold_neighbor {
                        visited_indices.insert(neighbor_index);
                        connected_component.insert(neighbor_index);
                        stack.push(neighbor_index);
                    }
                }
            }
        }

        connected_component
    }

    /// Findet alle Connected Components unter den aktiven Observern
    pub fn find_connected_components(&mut self, zeta: f64) -> Vec<HashSet<usize>> {
        // Strict Top-N aktive Observer (genau num_active via iter_observers(true))
        let active_indices: Vec<usize> = self.iter_observers(true).map(|obs| obs.index).collect();
        let active_set: HashSet<usize> = active_indices.iter().copied().collect();
        let mut connected_components = Vec::new();
        let mut visited_indices: HashSet<usize> = HashSet::new();
        let global_threshold = self.global_threshold;

        for &start_index in &active_indices {
            if visited_indices.contains(&start_index) {
                continue; // Bereits besucht
            }

            let component = self.dfs(
                start_index,
                zeta,
                global_threshold,
                &active_set,
                &mut visited_indices,
            );
            if !component.is_empty() {
                connected_components.push(component);
            }
        }

        connected_components
    }

    /// Entfernt kleine Cluster aus der Liste
    fn remove_small_clusters(
        connected_components: &mut Vec<HashSet<usize>>,
        min_cluster_size: usize,
    ) {
        connected_components.retain(|component| component.len() >= min_cluster_size);
    }

    /// Gibt gefundene Connected Components zurück (nach remove_small_clusters), nur für Debug.
    /// Jede Komponente ist eine Liste von Observer-Indizes.
    pub fn get_connected_components_for_debug(
        &mut self,
        zeta: f64,
        min_cluster_size: usize,
    ) -> Vec<Vec<usize>> {
        let mut components = self.find_connected_components(zeta);
        Self::remove_small_clusters(&mut components, min_cluster_size);
        components
            .into_iter()
            .map(|set| set.into_iter().collect())
            .collect()
    }

    /// Findet den Cluster mit dem maximalen Score basierend auf historischen Label-Observations
    /// Berücksichtigt nur noch nicht verarbeitete Cluster und noch nicht verwendete Labels
    /// Gibt Option<(cluster_index, max_score, candidate_label)> zurück
    fn get_max_cluster_score(
        &self,
        connected_components: &[HashSet<usize>],
        used_labels: &HashSet<usize>,
        processed_connected_components: &HashSet<usize>,
    ) -> Option<(usize, f64, usize)> {
        connected_components
            .iter()
            .enumerate()
            .filter(|(cluster_idx, _)| !processed_connected_components.contains(cluster_idx))
            .filter_map(|(cluster_idx, cluster_set)| {
                // Berechne normalisierte Cluster-Scores für alle Observer im Cluster
                let label_scores = self.get_normalized_cluster_scores(
                    &cluster_set.iter().cloned().collect::<Vec<_>>(),
                );

                // Filtere nur noch nicht verwendete Labels
                let available_label_scores: HashMap<usize, f64> = label_scores
                    .into_iter()
                    .filter(|(label, _)| !used_labels.contains(label))
                    .collect();

                // Finde maximalen Score und entsprechendes Label aus verfügbaren Labels
                let (candidate_label, max_score) = available_label_scores
                    .iter()
                    .max_by(|(_, &a), (_, &b)| {
                        a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(&label, &score)| (label, score))
                    .unwrap_or((0usize, 0.0));

                Some((cluster_idx, max_score, candidate_label))
            })
            .max_by(|(_, score_a, _), (_, score_b, _)| {
                score_a
                    .partial_cmp(score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Weist Labels zu Clustern zu basierend auf historischen Cluster-Beobachtungen (Algorithmus 3.5)
    /// Iterative Label-Zuweisung: In jeder Iteration wird der Cluster mit maximalem Score gefunden und zugewiesen.
    /// Kaltstart: Sind alle label_observations leer, bekommt jede Connected Component ein eindeutiges Label (0, 1, 2, …).
    /// Gibt eine HashMap zurück: cluster_index -> label (usize)
    pub fn label_connected_components(
        &mut self,
        connected_components: &[HashSet<usize>],
    ) -> HashMap<usize, usize> {
        let mut cluster_labels: HashMap<usize, usize> = HashMap::new();

        // Kaltstart: Alle Observer haben leere label_observations → jeder Component ein eindeutiges Label
        let any_has_labels = connected_components.iter().any(|cluster_set| {
            cluster_set.iter().any(|&obs_idx| {
                self.get(obs_idx)
                    .map(|o| !o.label_observations.is_empty())
                    .unwrap_or(false)
            })
        });
        if !any_has_labels {
            for (cluster_idx, _) in connected_components.iter().enumerate() {
                cluster_labels.insert(cluster_idx, cluster_idx);
            }
            self.last_label = connected_components.len().checked_sub(1).unwrap_or(0);
            return cluster_labels;
        }

        let mut used_labels: HashSet<usize> = HashSet::new();
        let mut processed_connected_components: HashSet<usize> = HashSet::new();
        let mut next_novel_label = self.last_label + 1;

        // Wiederhole bis alle Cluster verarbeitet sind
        while processed_connected_components.len() < connected_components.len() {
            if let Some((cluster_idx, score, candidate_label)) = self.get_max_cluster_score(
                connected_components,
                &used_labels,
                &processed_connected_components,
            ) {
                if score == 0.0 {
                    let novel_label = next_novel_label;
                    cluster_labels.insert(cluster_idx, novel_label);
                    used_labels.insert(novel_label);
                    processed_connected_components.insert(cluster_idx);
                    next_novel_label += 1;
                } else {
                    cluster_labels.insert(cluster_idx, candidate_label);
                    used_labels.insert(candidate_label);
                    processed_connected_components.insert(cluster_idx);
                }
            } else {
                for (cluster_idx, _) in connected_components.iter().enumerate() {
                    if !processed_connected_components.contains(&cluster_idx) {
                        let novel_label = next_novel_label;
                        cluster_labels.insert(cluster_idx, novel_label);
                        used_labels.insert(novel_label);
                        processed_connected_components.insert(cluster_idx);
                        next_novel_label += 1;
                    }
                }
                break;
            }
        }

        self.last_label = next_novel_label.saturating_sub(1);
        cluster_labels
    }

    /// Führt vollständiges Clustering durch: Thresholds setzen, Connected Components finden,
    /// Labels zuweisen und Label-Observations aktualisieren
    /// chi: Anzahl der nächsten Observer für lokale Thresholds
    /// zeta: Mixing-Parameter für globale/lokale Thresholds
    /// min_cluster_size: minimale Clustergröße
    /// fading: Optional Fading-Parameter für Streaming (f = exp(-T^-1))
    /// current_time: Optional aktuelle Zeit für Streaming
    /// Gibt die Cluster-Map zurück: HashMap<label, HashSet<observer_indices>>
    pub fn learn_clustering(
        &mut self,
        chi: usize,
        zeta: f64,
        min_cluster_size: usize,
        _fading: Option<f64>,
        _current_time: Option<f64>,
    ) {
        // Schritt 1: Setze Thresholds für alle aktiven Observer
        self.set_thresholds(chi);

        // Schritt 2: Finde Connected Components mit Helper-Methode
        let mut connected_components = self.find_connected_components(zeta);

        // Schritt 3: Entferne kleine Cluster mit Helper-Methode
        connected_components.retain(|component| component.len() >= min_cluster_size);

        // Schritt 4: Weise Labels zu basierend auf historischen Label-Observations
        let cluster_labels = self.label_connected_components(&connected_components);

        // Schritt 5: Aktualisiere Label-Observations (nur Zähler)
        self.update_label_observations_with_clusters(&connected_components, &cluster_labels);
    }

    /// Berechnet normalisierte Cluster-Scores für gegebene Observer-Indizes
    /// Gibt HashMap zurück: label -> normalisierter Score
    /// Normalisiert jede label_observations HashMap vor der Summierung
    pub fn get_normalized_cluster_scores(&self, observer_indices: &[usize]) -> HashMap<usize, f64> {
        let mut label_scores: HashMap<usize, f64> = HashMap::new();

        for &obs_idx in observer_indices {
            if let Some(observer) = self.get(obs_idx) {
                let normalized_scores = observer.get_normalized_label_observations();
                for (label, &normalized_value) in &normalized_scores {
                    *label_scores.entry(*label).or_insert(0.0) += normalized_value;
                }
            }
        }

        label_scores
    }

    /// Berechnet und setzt lokale und globale Thresholds für alle aktiven Observer
    pub fn set_thresholds(&mut self, chi: usize) {
        if self.distance_matrix.is_empty() {
            self.rebuild_distance_lists();
        }

        // Strict Top-N aktive Observer (genau num_active via iter_observers(true))
        let active_indices: Vec<usize> = self.iter_observers(true).map(|obs| obs.index).collect();
        let active_set: HashSet<usize> = active_indices.iter().copied().collect();
        let mut local_thresholds = Vec::with_capacity(active_indices.len());

        for &idx in &active_indices {
            let h_omega = self.compute_local_threshold_impl(idx, chi, &active_set);
            self.update_local_threshold(idx, h_omega);
            local_thresholds.push(h_omega);
        }

        self.global_threshold = if !local_thresholds.is_empty() {
            local_thresholds.iter().sum::<f64>() / local_thresholds.len() as f64
        } else {
            f64::INFINITY
        };
    }

    /// Implementierung der lokalen Threshold-Berechnung
    pub(crate) fn compute_local_threshold_impl(
        &self,
        index: usize,
        chi: usize,
        active_set: &HashSet<usize>,
    ) -> f64 {
        if let Some(list) = self.distance_matrix.get(index) {
            let mut found = 0;

            // Iteriere über sortierte Distanzen bis chi aktive Observer gefunden
            for (target_idx, dist) in &list.distances {
                if active_set.contains(target_idx) {
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
                    if active_set.contains(target_idx) {
                        return *dist;
                    }
                }
            }
        }

        f64::INFINITY
    }

    /// Aktualisiert Cluster-Beobachtungen Lω: nur Zähler, kein Fading/Zeit.
    /// Für jeden Observer in einem Cluster: Lcω ← Lcω + 1 für das zugehörige Label.
    pub fn update_label_observations_with_clusters(
        &mut self,
        clusters: &Vec<HashSet<usize>>,
        cluster_labels: &HashMap<usize, usize>,
    ) {
        if clusters.is_empty() || cluster_labels.is_empty() {
            return;
        }

        let mut label_updates: Vec<(usize, HashMap<usize, f64>, f64)> = Vec::new();

        for (cluster_idx, cluster_set) in clusters.iter().enumerate() {
            if let Some(&label) = cluster_labels.get(&cluster_idx) {
                for &obs_idx in cluster_set {
                    if let Some(observer) = self.get(obs_idx) {
                        let mut new_observations = observer.label_observations.clone();
                        *new_observations.entry(label).or_insert(0.0) += 1.0;
                        label_updates.push((obs_idx, new_observations, observer.label_time));
                    }
                }
            }
        }

        for (index, new_observations, label_time) in label_updates {
            self.update_label_observations(index, new_observations, label_time);
        }
    }
}
