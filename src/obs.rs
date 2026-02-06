use std::cmp::Ordering;
use std::collections::HashMap;

/// Observer-Struktur mit Daten, Beobachtungen und Index
#[derive(Clone, Debug)]
pub struct Observer {
    pub data: Vec<f64>,
    pub observations: f64,
    pub time: f64,
    /// last time the observer was updated
    pub age: f64,
    pub index: usize,
    /// Local threshold h_ω
    pub local_threshold: f64,
    /// Label observations Lω ∈ R^|C| - historische Cluster-Zugehörigkeiten
    pub label_observations: HashMap<usize, f64>,
    /// last time label_observations were updated
    pub label_time: f64,
}

impl Observer {
    pub fn get_label(&self) -> Option<usize> {
        self.label_observations
            .iter()
            .max_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&label, _)| label)
    }
    /// Gibt den normalisierten Cluster-Score für diesen Observer zurück
    /// Normalisiert den Lω Vektor so dass die Summe = 1 (leerer Vektor -> leere HashMap)
    pub fn get_normalized_label_observations(&self) -> HashMap<usize, f64> {
        let mut normalized_scores: HashMap<usize, f64> = HashMap::new();

        if !self.label_observations.is_empty() {
            let sum: f64 = self.label_observations.values().sum();
            if sum > 0.0 {
                for (&label, &value) in self.label_observations.iter() {
                    let normalized_value = value / sum;
                    normalized_scores.insert(label, normalized_value);
                }
            }
        }

        normalized_scores
    }
}

/// Neighbor information with index, distance, and active status
/// Used for unified k-nearest neighbor search operations
#[derive(Debug, Clone)]
pub struct NeighborInfo {
    pub index: usize,
    pub distance: f64,
    pub is_active: bool,
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
