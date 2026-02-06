use std::collections::{HashMap, HashSet};

use crate::obset::ObserverSet;

/// Clustering-Erweiterungen für ObserverSet
impl ObserverSet {
    /// Wie learn_clustering, aber verwendet batch_age statt +1.0 für label_observations
    pub fn learn_clustering_time(
        &mut self,
        chi: usize,
        zeta: f64,
        min_cluster_size: usize,
        fading: f64,
        batch_start_time: f64,
        batch_end_time: f64,
        batch_age: f64,
    ) {
        // Schritt 1: Setze Thresholds für alle aktiven Observer
        self.set_thresholds(chi);

        // Schritt 2: Finde Connected Components mit Helper-Methode
        let mut connected_components = self.find_connected_components(zeta);

        // Schritt 3: Entferne kleine Cluster
        connected_components.retain(|component| component.len() >= min_cluster_size);

        // Schritt 4: Weise Labels zu basierend auf historischen Label-Observations
        let cluster_labels = self.label_connected_components(&connected_components);

        // Schritt 5: Aktualisiere Label-Observations (mit Fading und batch_age)
        self.update_label_observations_time(
            &connected_components,
            &cluster_labels,
            fading,
            batch_start_time,
            batch_end_time,
            batch_age,
        );
    }

    /// Aktualisiert Label-Observations mit batch_age und korrektem Fading.
    /// 1. Fade alle bestehenden label_observation-Werte: v * fading^(batch_end_time - observer.label_time).
    /// 2. Addiere batch_age (gefadet) zum beobachteten Label: batch_age * fading^(batch_start_time - observer.label_time),
    ///    oder füge das Label in die HashMap ein falls noch nicht vorhanden.
    fn update_label_observations_time(
        &mut self,
        clusters: &Vec<HashSet<usize>>,
        cluster_labels: &HashMap<usize, usize>,
        fading: f64,
        batch_start_time: f64,
        batch_end_time: f64,
        batch_age: f64,
    ) {
        if clusters.is_empty() || cluster_labels.is_empty() {
            return;
        }

        let mut label_updates: Vec<(usize, HashMap<usize, f64>, f64)> = Vec::new();

        for (cluster_idx, cluster_set) in clusters.iter().enumerate() {
            if let Some(&label) = cluster_labels.get(&cluster_idx) {
                for &obs_idx in cluster_set {
                    if let Some(observer) = self.get(obs_idx) {
                        // 1. Fade all existing label_observation values: v * fading^(batch_end_time - label_time)
                        let time_diff_to_end = batch_end_time - observer.label_time;
                        let fade_factor = fading.powf(time_diff_to_end);
                        let mut new_observations: HashMap<usize, f64> = observer
                            .label_observations
                            .iter()
                            .map(|(&l, &v)| (l, v * fade_factor))
                            .collect();

                        // 2. Add batch_age (faded) to the observed label, or add label to hashmap
                        let time_diff_start_to_observer = batch_start_time - observer.label_time;
                        let faded_batch_age = batch_age * fading.powf(time_diff_start_to_observer);
                        *new_observations.entry(label).or_insert(0.0) += faded_batch_age;

                        label_updates.push((obs_idx, new_observations, batch_end_time));
                    }
                }
            }
        }

        // Führe alle Updates aus
        for (index, new_observations, label_time) in label_updates {
            self.update_label_observations(index, new_observations, label_time);
        }
    }
}
