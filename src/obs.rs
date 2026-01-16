use std::cmp::Ordering;
use std::collections::HashMap;

/// Geordnete Distanzliste für einen Observer
/// Speichert Distanzen zu anderen Observern als (index, distance) Paare, sortiert nach distance
#[derive(Clone, Debug)]
pub(crate) struct OrderedDistanceList {
    /// Liste von (target_index, distance) Paaren, sortiert nach distance (aufsteigend)
    pub(crate) distances: Vec<(usize, f64)>,
}

impl OrderedDistanceList {
    /// Erstellt eine neue leere Distanzliste
    pub(crate) fn new() -> Self {
        Self {
            distances: Vec::new(),
        }
    }

    /// Fügt eine Distanz hinzu und behält die Sortierung bei
    /// O(n) - könnte mit Binary Search optimiert werden, aber für kleine n ist linear besser
    pub(crate) fn insert(&mut self, target_index: usize, distance: f64) {
        // Entferne alte Einträge für diesen target_index
        self.distances.retain(|(idx, _)| *idx != target_index);

        // Füge neuen Eintrag hinzu und sortiere
        self.distances.push((target_index, distance));
        self.distances
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    }

    /// Entfernt einen Eintrag für den gegebenen target_index
    pub(crate) fn remove(&mut self, target_index: usize) {
        self.distances.retain(|(idx, _)| *idx != target_index);
    }

    /// Gibt alle Distanzen zurück (nur für Debugging)
    #[allow(dead_code)]
    fn get_all_distances(&self) -> &[(usize, f64)] {
        &self.distances
    }
}

/// Observer-Struktur mit Daten, Beobachtungen und Index
#[derive(Clone, Debug)]
pub struct Observer {
    pub data: Vec<f64>,
    pub observations: f64,
    pub time: f64,
    /// last time the observer was updated
    pub age: f64,
    pub index: usize,
    pub label: Option<i32>,
    /// Cluster observations Lω ∈ R^|C| - historische Cluster-Zugehörigkeiten
    pub cluster_observations: Vec<f64>,
}

impl Observer {
    /// Gibt den normalisierten Cluster-Score für diesen Observer zurück
    /// Normalisiert den Lω Vektor so dass die Summe = 1 (leerer Vektor -> leere HashMap)
    pub fn get_normalized_cluster_score(&self) -> HashMap<i32, f64> {
        let mut normalized_scores: HashMap<i32, f64> = HashMap::new();

        if !self.cluster_observations.is_empty() {
            let sum: f64 = self.cluster_observations.iter().sum();
            if sum > 0.0 {
                for (label_idx, &value) in self.cluster_observations.iter().enumerate() {
                    let label = label_idx as i32;
                    let normalized_value = value / sum;
                    normalized_scores.insert(label, normalized_value);
                }
            }
        }

        normalized_scores
    }
}

// Helper struct for comparing floats in collections
#[derive(Debug, Clone, Copy)]
pub(crate) struct OrderedFloat(pub(crate) f64);

impl PartialEq for OrderedFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

// Composite key for sorting by observations (descending)
// Key: (OrderedFloat(observations), index) - sorted descending by observations, then by index
#[derive(Clone, Debug, Copy)]
pub(crate) struct ObservationKey {
    pub(crate) observations: OrderedFloat,
    pub(crate) index: usize,
}

impl PartialEq for ObservationKey {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl Eq for ObservationKey {}

impl PartialOrd for ObservationKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ObservationKey {
    fn cmp(&self, other: &Self) -> Ordering {
        // Primary: observations (descending) - reverse the comparison
        self.observations
            .cmp(&other.observations)
            .reverse()
            // Secondary: index (ascending) as tie-breaker
            .then(self.index.cmp(&other.index))
    }
}

// Composite key for sorting by normalized score (ascending - worst first)
// Key: (OrderedFloat(normalized_score), index) - sorted ascending by score, then by index
#[derive(Clone, Debug, Copy)]
pub(crate) struct NormalizedScoreKey {
    pub(crate) score: OrderedFloat,
    pub(crate) index: usize,
}

impl PartialEq for NormalizedScoreKey {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl Eq for NormalizedScoreKey {}

impl PartialOrd for NormalizedScoreKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NormalizedScoreKey {
    fn cmp(&self, other: &Self) -> Ordering {
        // Primary: normalized score (ascending - worst first)
        self.score
            .cmp(&other.score)
            // Secondary: index (ascending) as tie-breaker
            .then(self.index.cmp(&other.index))
    }
}
