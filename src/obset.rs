use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use crate::distance_matrix::DistanceMatrix;
use crate::obs::{NeighborInfo, NormalizedScoreKey, ObservationKey, Observer, OrderedFloat};
use crate::utils::{compute_distance, DistanceMetric};

/// Efficient ObserverSet with dual indexing for O(log n) operations
/// Uses Brute-Force for k-NN operations (Tree support disabled)
/// Uses Arc<Observer> for shared ownership and direct access without HashMap lookups
#[derive(Clone)]
pub struct ObserverSet {
    // Primary storage: O(1) access by index, Arc for shared ownership
    pub(crate) observers_by_index: HashMap<usize, Arc<Observer>>,

    // Secondary index: sorted by observations (descending)
    // BTreeSet gives us O(log n) for min/max and sorted iteration, no redundant value
    pub(crate) indices_by_obs: BTreeSet<ObservationKey>,

    // Tertiary index: sorted by normalized score (observations/age, ascending)
    // O(log n) finding of worst observer
    pub(crate) indices_by_score: BTreeSet<NormalizedScoreKey>,

    // Parameters for distance computation
    distance_metric: DistanceMetric,
    minkowski_p: Option<f64>,

    // Number of active observers
    num_active: usize,

    // Cached index of the last active observer (lowest observations score among active observers)
    // None if num_active == 0 or there are fewer than num_active observers
    last_active_observer: Option<usize>,

    // Sparse distance matrix for local threshold / clustering (lazy initialization)
    pub(crate) distance_matrix: DistanceMatrix,
    pub(crate) global_threshold: f64,
    pub(crate) last_label: usize,
}

impl ObserverSet {
    pub fn new(distance_metric: DistanceMetric, minkowski_p: Option<f64>) -> Self {
        Self {
            observers_by_index: HashMap::new(),
            indices_by_obs: BTreeSet::new(),
            indices_by_score: BTreeSet::new(),
            distance_metric: distance_metric,
            minkowski_p: minkowski_p,
            num_active: 0,
            last_active_observer: None,
            distance_matrix: DistanceMatrix::new(distance_metric, minkowski_p),
            global_threshold: f64::INFINITY,
            last_label: 0,
        }
    }

    /// Set num_active
    pub fn set_num_active(&mut self, num_active: usize) {
        if num_active == self.num_active {
            return;
        }
        self.num_active = num_active;
        // Update last_active_observer cache
        self.update_last_active_observer();
    }

    /// Aktualisiert den Cache für last_active_observer
    /// Sollte aufgerufen werden, wenn sich die Observer-Liste oder num_active ändert
    fn update_last_active_observer(&mut self) {
        if self.num_active == 0 {
            self.last_active_observer = None;
            return;
        }

        // Finde den Observer an Position num_active - 1 (0-indexiert)
        // Das ist der Observer mit dem niedrigsten observations-Score unter den aktiven
        self.last_active_observer = self
            .indices_by_obs
            .iter()
            .nth(self.num_active - 1)
            .map(|key| key.index);
    }

    /// Rebuild the full distance matrix (e.g. for initial clustering).
    pub(crate) fn rebuild_distance_lists(&mut self) {
        self.distance_matrix.rebuild(&self.observers_by_index);
    }
    /// Prüft, ob ein Observer aktiv ist (gehört zu den Top num_active Observern nach observations)
    /// Ein Observer ist aktiv, wenn seine observations >= observations des last_active_observer sind
    /// O(1) - verwendet gecachten last_active_observer
    pub(crate) fn is_active(&self, index: usize) -> bool {
        if self.num_active == 0 {
            return false; // Keine aktiven Observer definiert
        }

        // Hole den Observer
        let observer = match self.observers_by_index.get(&index) {
            Some(arc) => arc.as_ref(),
            None => return false, // Observer existiert nicht
        };

        // Verwende gecachten last_active_observer
        match self.last_active_observer {
            Some(last_active_idx) => {
                // Hole den last_active_observer
                let last_active_arc = match self.observers_by_index.get(&last_active_idx) {
                    Some(arc) => arc.as_ref(),
                    None => {
                        // Cache ist veraltet, alle Observer sind aktiv
                        return true;
                    }
                };
                // Observer ist aktiv, wenn seine observations >= observations des last_active_observer
                observer.observations >= last_active_arc.observations
            }
            None => {
                // Wenn es weniger als num_active Observer gibt, sind alle aktiv
                true
            }
        }
    }

    /// Insert a new observer - O(log n)
    pub fn insert(&mut self, observer: Observer) {
        let index = observer.index;

        // Wenn ein Observer mit diesem Index bereits existiert, raise an error
        if self.observers_by_index.contains_key(&index) {
            panic!("Observer with index {} already exists", index);
        }

        // Create keys for secondary indices (no cloning of observer data)
        let obs_key = ObservationKey {
            observations: OrderedFloat(observer.observations),
            index,
        };
        let normalized_score = if observer.age > 0.0 {
            observer.observations / observer.age
        } else {
            f64::INFINITY
        };
        let score_key = NormalizedScoreKey {
            score: OrderedFloat(normalized_score),
            index,
        };

        // Insert into all structures with Arc for shared ownership
        let observer_arc = Arc::new(observer);
        self.observers_by_index.insert(index, observer_arc);
        self.indices_by_obs.insert(obs_key);
        self.indices_by_score.insert(score_key);

        // Update last_active_observer cache, da sich die Observer-Liste geändert hat
        self.update_last_active_observer();

        // Update distance lists für alle Observer
        let new_observer = self
            .observers_by_index
            .get(&index)
            .map(Arc::as_ref)
            .expect("just inserted");
        self.distance_matrix
            .insert(new_observer, &self.observers_by_index);
    }

    /// Get observer by index - O(1)
    pub fn get(&self, index: usize) -> Option<&Observer> {
        self.observers_by_index.get(&index).map(|arc| arc.as_ref())
    }

    /// Update observer's observations and age - O(log n)
    pub fn update_observer(&mut self, index: usize, new_observations: f64, new_age: f64) -> bool {
        // Get the current observer Arc
        let observer_arc = match self.observers_by_index.get(&index) {
            Some(arc) => arc.clone(),
            None => return false,
        };

        // Remove old entries from secondary indices using old values
        let old_obs_key = ObservationKey {
            observations: crate::obs::OrderedFloat(observer_arc.observations),
            index,
        };
        let old_normalized_score = if observer_arc.age > 0.0 {
            observer_arc.observations / observer_arc.age
        } else {
            f64::INFINITY
        };
        let old_score_key = NormalizedScoreKey {
            score: crate::obs::OrderedFloat(old_normalized_score),
            index,
        };

        self.indices_by_obs.remove(&old_obs_key);
        self.indices_by_score.remove(&old_score_key);

        // Update the observer - try to update in place if we have exclusive access
        let updated_observer = {
            // Get mutable reference to the Arc in the HashMap
            let arc_mut = self.observers_by_index.get_mut(&index).unwrap();
            if let Some(mut_observer) = Arc::get_mut(arc_mut) {
                // Exclusive access - update in place (no clone!)
                mut_observer.observations = new_observations;
                mut_observer.age = new_age;
                Arc::clone(arc_mut) // Clone the Arc reference, not the Observer
            } else {
                // Shared - create new Arc with updated values
                Arc::new(Observer {
                    data: observer_arc.data.clone(),
                    observations: new_observations,
                    time: observer_arc.time,
                    age: new_age,
                    index: observer_arc.index,
                    local_threshold: observer_arc.local_threshold,
                    label_observations: observer_arc.label_observations.clone(),
                    label_time: observer_arc.label_time,
                })
            }
        };

        // Update HashMap with new Arc
        self.observers_by_index.insert(index, updated_observer);

        // Re-insert with updated values
        let new_obs_key = ObservationKey {
            observations: crate::obs::OrderedFloat(new_observations),
            index,
        };
        let new_normalized_score = if new_age > 0.0 {
            new_observations / new_age
        } else {
            f64::INFINITY
        };
        let new_score_key = NormalizedScoreKey {
            score: crate::obs::OrderedFloat(new_normalized_score),
            index,
        };

        self.indices_by_obs.insert(new_obs_key);
        self.indices_by_score.insert(new_score_key);

        // Update last_active_observer cache, da sich observations geändert haben
        self.update_last_active_observer();

        true
    }

    /// Get top N observers by observations - O(N)
    /// Clones observers - use get_active_arcs() for better performance
    pub fn get_observers(&self, active: bool) -> Vec<Observer> {
        self.indices_by_obs
            .iter()
            .take(if active {
                self.num_active
            } else {
                self.observers_by_index.len()
            })
            .filter_map(|key| {
                self.observers_by_index
                    .get(&key.index)
                    .map(|arc| (**arc).clone())
            })
            .collect()
    }

    /// Get iterator over active observers (top N by observations) - O(1) to create, O(N) to iterate
    /// More efficient than get_active when you only need to iterate without cloning
    pub fn iter_observers(&self, active: bool) -> impl Iterator<Item = &Observer> {
        // Collect indices first, then map to Arc dereferences
        let indices: Vec<usize> = self
            .indices_by_obs
            .iter()
            .take(if active {
                self.num_active
            } else {
                self.observers_by_index.len()
            })
            .map(|key| key.index)
            .collect();
        indices
            .into_iter()
            .filter_map(move |index| self.observers_by_index.get(&index).map(|arc| arc.as_ref()))
    }

    /// Remove an observer by index - O(log n)
    pub fn remove(&mut self, index: usize) -> Option<Observer> {
        // Wenn ein Observer mit diesem Index nicht existiert, raise an error
        if !self.observers_by_index.contains_key(&index) {
            panic!("Observer with index {} does not exist", index);
        }

        // Entferne ALLE Einträge mit diesem index aus sekundären Indizes
        // (nicht nur den mit aktuellen observations, da sich diese geändert haben könnten)
        let keys_to_remove_obs: Vec<ObservationKey> = self
            .indices_by_obs
            .iter()
            .filter(|key| key.index == index)
            .cloned()
            .collect();
        for key in keys_to_remove_obs {
            self.indices_by_obs.remove(&key);
        }

        let keys_to_remove_score: Vec<NormalizedScoreKey> = self
            .indices_by_score
            .iter()
            .filter(|key| key.index == index)
            .cloned()
            .collect();
        for key in keys_to_remove_score {
            self.indices_by_score.remove(&key);
        }

        // Remove from primary structure
        let observer_arc = self.observers_by_index.remove(&index)?;

        // Update last_active_observer cache, da sich die Observer-Liste geändert hat
        self.update_last_active_observer();

        // Update distance lists: Entferne diesen Observer aus allen Distanzlisten
        self.distance_matrix.remove(index);

        // Return owned Observer (dereference Arc)
        Some((*observer_arc).clone())
    }

    /// Replace an observer - O(log n)
    pub fn replace(&mut self, old_index: usize, new_observer: Observer) -> bool {
        // Remove old observer
        if self.remove(old_index).is_none() {
            return false;
        }

        // Insert new observer
        self.insert(new_observer);
        true
    }

    /// Get number of observers - O(1)
    pub fn len(&self) -> usize {
        self.observers_by_index.len()
    }

    /// Check if empty - O(1)
    pub fn is_empty(&self) -> bool {
        self.observers_by_index.is_empty()
    }

    /// Get number of active observers - O(1)
    pub fn get_num_active(&self) -> usize {
        self.num_active
    }

    /// Get iterator over all observers (unsorted)
    #[allow(dead_code)] // Für zukünftige Verwendung
    pub fn iter(&self) -> impl Iterator<Item = &Observer> {
        self.observers_by_index.values().map(|arc| arc.as_ref())
    }

    /// Find the worst observer by normalized score - O(1)
    /// By default k = 1
    pub fn find_k_worst(&self, k: Option<usize>) -> Vec<usize> {
        if let Some(k) = k {
            if k < 1 {
                panic!("k must be greater than 0");
            }
        }
        self.indices_by_score
            .iter()
            .take(k.unwrap_or(1))
            .map(|key| key.index)
            .collect()
    }

    /// Update only observations - O(log n)
    pub fn update_observations(&mut self, index: usize, new_observations: f64) -> bool {
        if let Some(observer) = self.observers_by_index.get(&index) {
            let current_age = observer.age;
            self.update_observer(index, new_observations, current_age)
        } else {
            false
        }
    }

    /// Update observer label observations HashMap - O(1)
    pub fn update_label_observations(
        &mut self,
        index: usize,
        label_observations: HashMap<usize, f64>,
        label_time: f64,
    ) -> bool {
        if let Some(arc) = self.observers_by_index.get_mut(&index) {
            if let Some(mut_observer) = Arc::get_mut(arc) {
                // Exclusive access - update in place (no clone!)
                mut_observer.label_observations = label_observations;
                mut_observer.label_time = label_time;
                true
            } else {
                // Shared - create new Arc with updated label_observations
                let observer_arc = self.observers_by_index.get(&index).unwrap().clone();
                let updated_observer = Arc::new(Observer {
                    data: observer_arc.data.clone(),
                    observations: observer_arc.observations,
                    time: observer_arc.time,
                    age: observer_arc.age,
                    index: observer_arc.index,
                    local_threshold: observer_arc.local_threshold,
                    label_observations,
                    label_time,
                });
                self.observers_by_index.insert(index, updated_observer);
                true
            }
        } else {
            false
        }
    }

    /// Update observer local threshold - O(1)
    pub fn update_local_threshold(&mut self, index: usize, local_threshold: f64) -> bool {
        if let Some(arc) = self.observers_by_index.get_mut(&index) {
            if let Some(mut_observer) = Arc::get_mut(arc) {
                // Exclusive access - update in place (no clone!)
                mut_observer.local_threshold = local_threshold;
                true
            } else {
                // Shared - create new Arc with updated local_threshold
                let observer_arc = self.observers_by_index.get(&index).unwrap().clone();
                let updated_observer = Arc::new(Observer {
                    data: observer_arc.data.clone(),
                    observations: observer_arc.observations,
                    time: observer_arc.time,
                    age: observer_arc.age,
                    index: observer_arc.index,
                    local_threshold,
                    label_observations: observer_arc.label_observations.clone(),
                    label_time: observer_arc.label_time,
                });
                self.observers_by_index.insert(index, updated_observer);
                true
            }
        } else {
            false
        }
    }

    /// Returns (active_neighbors, all_neighbors) where:
    /// - active_neighbors: k-nearest active neighbors only
    /// - all_neighbors: k-nearest neighbors regardless of active status, only if learn is set and true, otherwise None
    pub fn search_neighbors_unified(
        &self,
        query_point: &[f64],
        k: usize,
        learn: Option<bool>,
        k_learn: Option<usize>,
    ) -> (Vec<NeighborInfo>, Option<Vec<NeighborInfo>>) {
        let mut nearest_active = Vec::with_capacity(k);
        // Determine k for nearest_all: use k_learn if provided, otherwise fall back to k
        let k_all = k_learn.unwrap_or(0) + k;
        let mut nearest_all = if let Some(flag) = learn {
            if flag {
                Some(Vec::with_capacity(k_all))
            } else {
                None
            }
        } else {
            None
        };

        // Helper function to update k-nearest vectors with worst candidate replacement
        let update_k_nearest =
            |candidates: &mut Vec<NeighborInfo>, neighbor_info: NeighborInfo, k_limit: usize| {
                if candidates.len() < k_limit {
                    candidates.push(neighbor_info);
                } else {
                    // Find worst candidate (max distance)
                    let worst_idx = candidates
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.distance.partial_cmp(&b.distance).unwrap())
                        .map(|(idx, _)| idx)
                        .unwrap();

                    // Replace if new neighbor is closer than current worst
                    if neighbor_info.distance < candidates[worst_idx].distance {
                        candidates[worst_idx] = neighbor_info;
                    }
                }
            };

        // Single pass through sorted observers
        for (position, obs_key) in self.indices_by_obs.iter().enumerate() {
            let observer = match self.observers_by_index.get(&obs_key.index) {
                Some(obs) => obs,
                None => continue,
            };

            // Determine active status from position in sorted order
            let is_active = position < self.num_active;

            // Break if learn is false or not set and this one is inactive
            if ((learn.is_some() && !learn.unwrap()) || learn.is_none()) && !is_active {
                break;
            }

            // Compute distance once
            let distance = compute_distance(
                &observer.data,
                query_point,
                self.distance_metric,
                self.minkowski_p,
            );

            let neighbor_info = NeighborInfo {
                index: obs_key.index,
                distance,
                is_active,
            };

            // Update nearest (all observers if learn is Some(true))
            if let Some(true) = learn {
                if let Some(ref mut all_vec) = nearest_all {
                    update_k_nearest(all_vec, neighbor_info.clone(), k_all);
                }
            }

            // Update nearest_active (only active observers)
            if is_active {
                update_k_nearest(&mut nearest_active, neighbor_info, k);
            }
        }

        (nearest_active, nearest_all)
    }

    /// Distanz von einem Punkt zu einem Observer (für Wiederverwendung in SDOstream Step 4)
    pub(crate) fn distance_from_point(&self, point: &[f64], observer_index: usize) -> Option<f64> {
        let observer = self.observers_by_index.get(&observer_index)?;
        Some(compute_distance(
            &observer.data,
            point,
            self.distance_metric,
            self.minkowski_p,
        ))
    }

    /// Finde k nächste Nachbarn für einen Observer unter Verwendung der Distanzliste
    /// O(k) da Distanzliste bereits sortiert ist
    pub fn get_k_nearest_neighbors(&self, observer_index: usize, k: usize) -> Vec<(usize, f64)> {
        if let Some(distance_list) = self.distance_matrix.get(observer_index) {
            let end = k.min(distance_list.distances.len());
            distance_list.distances[..end].to_vec()
        } else {
            Vec::new()
        }
    }

    /// Finde alle Nachbarn innerhalb eines Thresholds unter Verwendung der Distanzliste
    /// O(log n + m) wobei m die Anzahl der Nachbarn innerhalb des Thresholds ist
    pub fn get_neighbors_within_threshold(
        &self,
        observer_index: usize,
        threshold: f64,
    ) -> Vec<(usize, f64)> {
        if let Some(distance_list) = self.distance_matrix.get(observer_index) {
            let end_pos = distance_list.find_threshold_position(threshold);
            distance_list.distances[..end_pos].to_vec()
        } else {
            Vec::new()
        }
    }

    /// Batch-Update für Distanzlisten wenn mehrere Observer gleichzeitig aktualisiert werden
    /// Vermeidet wiederholte Neuberechnungen
    pub fn batch_update_distance_lists(&mut self, updated_indices: &[usize]) {
        if updated_indices.is_empty() {
            return;
        }

        // Sammle aktuelle Daten für alle aktualisierten Observer
        let updated_data: HashMap<usize, Vec<f64>> = updated_indices
            .iter()
            .filter_map(|&idx| {
                self.observers_by_index
                    .get(&idx)
                    .map(|arc| (idx, arc.data.clone()))
            })
            .collect();

        // Aktualisiere Distanzen nur zwischen aktualisierten Observern
        for &i in updated_indices {
            for &j in updated_indices {
                if i < j {
                    // Vermeide Doppelarbeit
                    if let (Some(data_i), Some(data_j)) =
                        (updated_data.get(&i), updated_data.get(&j))
                    {
                        let distance = compute_distance(
                            data_i,
                            data_j,
                            self.distance_metric,
                            self.minkowski_p,
                        );

                        // Aktualisiere beide Richtungen
                        if let Some(list_i) = self.distance_matrix.get_mut(i) {
                            list_i.insert(j, distance);
                        }
                        if let Some(list_j) = self.distance_matrix.get_mut(j) {
                            list_j.insert(i, distance);
                        }
                    }
                }
            }
        }
    }

    /// Calculate Mahalanobis distance uniformity score for a subset of observers.
    /// Requires feature "mahalanobis". Returns a score where lower = more uniform (convex).
    #[cfg(feature = "mahalanobis")]
    pub fn mahalanobis_uniformity_score(&self, observer_indices: Option<&[usize]>) -> f64 {
        // Collect observer data based on indices
        let observer_data: Vec<Vec<f64>> = match observer_indices {
            Some(indices) => indices
                .iter()
                .filter_map(|&idx| self.get(idx).map(|obs| obs.data.clone()))
                .collect(),
            None => self
                .iter_observers(true)
                .map(|obs| obs.data.clone())
                .collect(),
        };

        if observer_data.len() < 2 {
            return 0.0; // No meaningful score for < 2 points
        }

        // Calculate mean vector
        let num_observers = observer_data.len();
        let num_features = observer_data[0].len();

        let mut mean = vec![0.0; num_features];
        for obs_data in &observer_data {
            for (j, &value) in obs_data.iter().enumerate() {
                mean[j] += value;
            }
        }
        for value in &mut mean {
            *value /= num_observers as f64;
        }

        // Calculate covariance matrix manually
        let mut cov_matrix = vec![vec![0.0; num_features]; num_features];
        for obs_data in &observer_data {
            for i in 0..num_features {
                for j in 0..num_features {
                    let diff_i = obs_data[i] - mean[i];
                    let diff_j = obs_data[j] - mean[j];
                    cov_matrix[i][j] += diff_i * diff_j;
                }
            }
        }
        for i in 0..num_features {
            for j in 0..num_features {
                cov_matrix[i][j] /= (num_observers - 1) as f64;
            }
        }

        // Try to compute inverse of covariance matrix using our helper function
        let inv_cov = match matrix_inverse_2x2_or_3x3(&cov_matrix) {
            Some(inv) => inv,
            None => {
                // For singular matrices, use diagonal approximation
                let mut inv_cov = vec![vec![0.0; num_features]; num_features];
                for i in 0..num_features {
                    if cov_matrix[i][i] > 1e-10 {
                        inv_cov[i][i] = 1.0 / cov_matrix[i][i];
                    } else {
                        inv_cov[i][i] = 1.0; // Regularization
                    }
                }
                inv_cov
            }
        };

        // Calculate Mahalanobis distances for each observer
        let mut distances = Vec::new();
        for obs_data in &observer_data {
            let diff: Vec<f64> = obs_data.iter().zip(&mean).map(|(x, m)| x - m).collect();

            // Calculate diff^T * inv_cov * diff
            let mut temp = vec![0.0; num_features];
            for i in 0..num_features {
                for j in 0..num_features {
                    temp[i] += inv_cov[i][j] * diff[j];
                }
            }

            let mut mahal_dist_sq = 0.0;
            for i in 0..num_features {
                mahal_dist_sq += diff[i] * temp[i];
            }

            distances.push(mahal_dist_sq.sqrt());
        }

        // Return mean distance as uniformity score
        distances.iter().sum::<f64>() / distances.len() as f64
    }

    /// Clear all observers - O(n)
    #[allow(dead_code)] // Für zukünftige Verwendung
    pub fn clear(&mut self) {
        self.observers_by_index.clear();
        self.indices_by_obs.clear();
        self.indices_by_score.clear();
    }
}

impl Default for ObserverSet {
    fn default() -> Self {
        Self::new(DistanceMetric::Euclidean, None)
    }
}

/// Simple matrix inverse for 2x2 or 3x3 matrices (used only for Mahalanobis).
#[cfg(feature = "mahalanobis")]
fn matrix_inverse_2x2_or_3x3(matrix: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();

    if n == 2 {
        let a = matrix[0][0];
        let b = matrix[0][1];
        let c = matrix[1][0];
        let d = matrix[1][1];
        let det = a * d - b * c;

        if det.abs() < 1e-10 {
            return None;
        }

        let inv_det = 1.0 / det;
        Some(vec![
            vec![d * inv_det, -b * inv_det],
            vec![-c * inv_det, a * inv_det],
        ])
    } else if n == 3 {
        // For 3x3, use simple cofactor method
        let det = matrix_determinant_3x3(matrix);

        if det.abs() < 1e-10 {
            return None;
        }

        let inv_det = 1.0 / det;
        let mut inv = vec![vec![0.0; 3]; 3];

        // Compute cofactors
        inv[0][0] = (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) * inv_det;
        inv[0][1] = (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) * inv_det;
        inv[0][2] = (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) * inv_det;

        inv[1][0] = (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) * inv_det;
        inv[1][1] = (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) * inv_det;
        inv[1][2] = (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) * inv_det;

        inv[2][0] = (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) * inv_det;
        inv[2][1] = (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) * inv_det;
        inv[2][2] = (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) * inv_det;

        Some(inv)
    } else {
        None
    }
}

/// Calculate determinant of 3x3 matrix (used only for Mahalanobis).
#[cfg(feature = "mahalanobis")]
fn matrix_determinant_3x3(matrix: &[Vec<f64>]) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}
