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
    /// 1. Für jedes bestehende Label (außer dem beobachteten): Fade mit observations * fading.powf(batch_end_time - label_entry.time)
    ///    und speichere als neue observations mit time = batch_end_time
    /// 2. Für das beobachtete Label: Berechne gefadeten Wert zur batch_start_time, addiere batch_age,
    ///    speichere als neue observations mit time = batch_end_time
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

        for (cluster_idx, cluster_set) in clusters.iter().enumerate() {
            if let Some(&observed_label) = cluster_labels.get(&cluster_idx) {
                for &obs_idx in cluster_set {
                    // Hole rohe label_observations für Updates
                    if let Some(label_map) = self.get_label_observations_raw(obs_idx) {
                        // Erstelle neue HashMap für aktualisierte LabelObservationEntry
                        use crate::obset::LabelObservationEntry;
                        let mut updated_map: HashMap<usize, LabelObservationEntry> = HashMap::new();
                        
                        // 1. Für jedes bestehende Label: Fade zur batch_end_time
                        for (&label, entry) in label_map.iter() {
                            if label == observed_label {
                                // Für das beobachtete Label: Berechne gefadeten Wert zur batch_start_time, addiere batch_age
                                let time_diff_to_start = batch_start_time - entry.time;
                                let faded_value = entry.observations * fading.powf(time_diff_to_start);
                                let new_observations = faded_value + batch_age;
                                updated_map.insert(label, LabelObservationEntry {
                                    observations: new_observations,
                                    time: batch_end_time,
                                });
                            } else {
                                // Für andere Labels: Fade zur batch_end_time
                                let time_diff_to_end = batch_end_time - entry.time;
                                let faded_value = entry.observations * fading.powf(time_diff_to_end);
                                updated_map.insert(label, LabelObservationEntry {
                                    observations: faded_value,
                                    time: batch_end_time,
                                });
                            }
                        }
                        
                        // Falls das beobachtete Label noch nicht existiert, füge es hinzu
                        if !updated_map.contains_key(&observed_label) {
                            // Berechne gefadeten batch_age von batch_start_time
                            let faded_batch_age = batch_age * fading.powf(batch_start_time - 0.0);
                            updated_map.insert(observed_label, LabelObservationEntry {
                                observations: faded_batch_age,
                                time: batch_end_time,
                            });
                        }
                        
                        // Aktualisiere mit neuer HashMap
                        self.update_label_observations(obs_idx, updated_map);
                    } else {
                        // Erstelle neue label_observations HashMap
                        use crate::obset::LabelObservationEntry;
                        let mut new_map: HashMap<usize, LabelObservationEntry> = HashMap::new();
                        // Berechne gefadeten batch_age von batch_start_time
                        let faded_batch_age = batch_age * fading.powf(batch_start_time - 0.0);
                        new_map.insert(observed_label, LabelObservationEntry {
                            observations: faded_batch_age,
                            time: batch_end_time,
                        });
                        self.update_label_observations(obs_idx, new_map);
                    }
                }
            }
        }
    }
}
