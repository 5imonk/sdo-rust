use std::collections::{HashMap, HashSet};

use crate::observer::ObserverSet;

/// Streaming-Clustering-Erweiterungen für ObserverSet
/// Enthält Funktionen für Cluster-Formation, Labeling und Lω Updates
impl ObserverSet {
    /// Weist Labels zu Clustern zu basierend auf historischen Cluster-Beobachtungen (Algorithmus 3.5)
    /// Gibt eine HashMap zurück: cluster_index -> label (i32)
    pub fn label_clusters(&self, clusters: &[HashSet<usize>]) -> HashMap<usize, i32> {
        let mut cluster_labels: HashMap<usize, i32> = HashMap::new();
        let mut available_labels: HashSet<i32> = HashSet::new();
        let mut next_novel_label = 0i32;

        // Berechne Scores für jeden Cluster
        let mut cluster_scores: Vec<(usize, f64, i32)> = clusters
            .iter()
            .enumerate()
            .map(|(cluster_idx, cluster_set)| {
                // Berechne normalisierte Lω Vektoren für alle Observer im Cluster
                let mut label_scores: HashMap<i32, f64> = HashMap::new();

                for &obs_idx in cluster_set {
                    if let Some(observer_arc) = self.observers_by_index.get(&obs_idx) {
                        let l_omega = &observer_arc.cluster_observations;
                        if l_omega.is_empty() {
                            continue;
                        }

                        // Normalisiere Lω (Summe = 1)
                        let sum: f64 = l_omega.iter().sum();
                        if sum > 0.0 {
                            for (label_idx, &value) in l_omega.iter().enumerate() {
                                let label = label_idx as i32;
                                *label_scores.entry(label).or_insert(0.0) += value / sum;
                            }
                        }
                    }
                }

                // Finde maximalen Score und entsprechendes Label
                let (candidate_label, max_score) = label_scores
                    .iter()
                    .max_by(|(_, &a), (_, &b)| {
                        a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(&label, &score)| (label, score))
                    .unwrap_or((0, 0.0));

                (cluster_idx, max_score, candidate_label)
            })
            .collect();

        // Sortiere nach Score (höchster zuerst) - Priority Queue
        cluster_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Weise Labels zu
        for (cluster_idx, score, candidate_label) in cluster_scores {
            if score == 0.0 {
                // Neuer Cluster (novel class)
                let novel_label = next_novel_label;
                cluster_labels.insert(cluster_idx, novel_label);
                available_labels.insert(novel_label);
                next_novel_label += 1;
            } else if available_labels.contains(&candidate_label) {
                // Label ist verfügbar
                cluster_labels.insert(cluster_idx, candidate_label);
                available_labels.remove(&candidate_label);
            } else {
                // Label bereits vergeben - recalculate (vereinfacht: weise novel label zu)
                let novel_label = next_novel_label;
                cluster_labels.insert(cluster_idx, novel_label);
                available_labels.insert(novel_label);
                next_novel_label += 1;
            }
        }

        cluster_labels
    }

    /// Aktualisiert Cluster-Beobachtungen Lω: Fading und/oder Cluster-Updates
    /// fading: Fading-Parameter f = exp(-T^-1)
    /// current_time: Aktuelle Zeit ti
    /// clusters: Liste von Cluster-Sets (kann leer sein, wenn nur Fading angewendet wird)
    /// cluster_labels: HashMap von cluster_index -> label (kann leer sein, wenn nur Fading angewendet wird)
    /// Verwendet intern last_cluster_time aus ObserverSet für Fading-Berechnung
    pub fn update_cluster_observations_with_fading_and_clusters(
        &mut self,
        fading: f64,
        current_time: f64,
        clusters: &[HashSet<usize>],
        cluster_labels: &HashMap<usize, i32>,
    ) {
        let time_diff = current_time - self.last_cluster_time;
        let apply_fading = time_diff > 0.0;

        // Schritt 1: Wende Fading an (wenn time_diff > 0)
        if apply_fading {
            let updates: Vec<(usize, Vec<f64>)> = {
                self.iter_observers(false)
                    .map(|observer| {
                        let observer_time_diff = current_time - observer.time;
                        let fading_factor = fading.powf(observer_time_diff);

                        // Wende Fading auf alle Cluster-Beobachtungen an
                        let faded_observations: Vec<f64> = observer
                            .cluster_observations
                            .iter()
                            .map(|&val| fading_factor * val)
                            .collect();

                        (observer.index, faded_observations)
                    })
                    .collect()
            };

            // Aktualisiere jeden Observer mit gefadeten Beobachtungen
            for (index, faded_observations) in updates {
                self.update_cluster_observations(index, faded_observations);
            }
        }

        // Schritt 2: Update Lω für Observer in Clustern: Lcω ← Lcω + 1
        if !clusters.is_empty() && !cluster_labels.is_empty() {
            for (cluster_idx, cluster_set) in clusters.iter().enumerate() {
                if let Some(&label) = cluster_labels.get(&cluster_idx) {
                    let label_idx = label as usize;

                    // Erweitere cluster_observations Vektoren falls nötig
                    for &obs_idx in cluster_set {
                        if let Some(observer) = self.get(obs_idx) {
                            let mut new_observations = observer.cluster_observations.clone();

                            // Stelle sicher, dass der Vektor groß genug ist
                            while new_observations.len() <= label_idx {
                                new_observations.push(0.0);
                            }

                            // Increment: Lcω ← Lcω + 1
                            new_observations[label_idx] += 1.0;

                            self.update_cluster_observations(obs_idx, new_observations);
                        }
                    }
                }
            }
        }

        // Update last_cluster_time nach erfolgreichem Update
        if apply_fading || !clusters.is_empty() {
            self.last_cluster_time = current_time;
        }
    }
}
