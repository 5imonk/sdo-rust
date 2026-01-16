use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::obs::{NormalizedScoreKey, ObservationKey, Observer, OrderedDistanceList, OrderedFloat};
use crate::utils::{compute_distance, DistanceMetric};

/// Efficient ObserverSet with dual indexing for O(log n) operations
/// Uses Brute-Force for k-NN operations (Tree support disabled)
/// Uses Arc<Observer> for shared ownership and direct access without HashMap lookups
pub struct ObserverSet {
    // Primary storage: O(1) access by index, Arc for shared ownership
    pub(crate) observers_by_index: HashMap<usize, Arc<Observer>>,

    // Secondary index: sorted by observations (descending) - stores only indices
    // Key: (observations, index) -> index
    // BTreeMap gives us O(log n) for min/max and sorted iteration
    pub(crate) indices_by_obs: BTreeMap<ObservationKey, usize>,

    // Tertiary index: sorted by normalized score (observations/age, ascending) - stores only indices
    // Key: (normalized_score, index) -> index
    // This allows O(log n) finding of worst observer
    pub(crate) indices_by_score: BTreeMap<NormalizedScoreKey, usize>,

    // Parameters for distance computation
    distance_metric: DistanceMetric,
    minkowski_p: Option<f64>,

    // Number of active observers
    num_active: usize,

    // Cached index of the last active observer (lowest observations score among active observers)
    // None if num_active == 0 or there are fewer than num_active observers
    last_active_observer: Option<usize>,

    // Geordnete Distanzlisten für jeden Observer (für effiziente lokale Threshold-Berechnung)
    // Key: observer_index -> OrderedDistanceList
    // Wird nur aktualisiert wenn clustering aktiv ist (lazy initialization)
    pub(crate) distance_lists: HashMap<usize, OrderedDistanceList>,

    // Letzte Zeit für Cluster-Beobachtungs-Fading (für SDOstreamclust)
    pub(crate) last_cluster_time: f64,
}

impl ObserverSet {
    pub fn new() -> Self {
        Self {
            observers_by_index: HashMap::new(),
            indices_by_obs: BTreeMap::new(),
            indices_by_score: BTreeMap::new(),
            distance_metric: DistanceMetric::Euclidean,
            minkowski_p: None,
            num_active: 0,
            last_active_observer: None,
            distance_lists: HashMap::new(),
            last_cluster_time: 0.0,
        }
    }

    /// Set distance parameters
    pub fn set_tree_params(&mut self, distance_metric: DistanceMetric, minkowski_p: Option<f64>) {
        self.distance_metric = distance_metric;
        self.minkowski_p = minkowski_p;
    }

    /// Get distance metric
    pub fn get_distance_metric(&self) -> DistanceMetric {
        self.distance_metric
    }

    /// Get minkowski_p parameter
    pub fn get_minkowski_p(&self) -> Option<f64> {
        self.minkowski_p
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
            .map(|(_key, &idx)| idx);
    }

    /// Aktualisiert Distanzlisten, wenn ein neuer Observer eingefügt wird
    fn update_distance_lists_on_insert(&mut self, new_index: usize, new_data: &[f64]) {
        // Für jeden existierenden Observer: Berechne Distanz zum neuen Observer und füge sie hinzu
        for (existing_index, existing_arc) in &self.observers_by_index {
            if *existing_index != new_index {
                let existing_data = &existing_arc.data;
                let distance = compute_distance(
                    existing_data,
                    new_data,
                    self.distance_metric,
                    self.minkowski_p,
                );

                // Füge Distanz zur Liste des existierenden Observers hinzu
                let list = self
                    .distance_lists
                    .entry(*existing_index)
                    .or_insert_with(OrderedDistanceList::new);
                list.insert(new_index, distance);
            }
        }

        // Erstelle neue Distanzliste für den neuen Observer
        let mut new_list = OrderedDistanceList::new();
        for (existing_index, existing_arc) in &self.observers_by_index {
            if *existing_index != new_index {
                let existing_data = &existing_arc.data;
                let distance = compute_distance(
                    new_data,
                    existing_data,
                    self.distance_metric,
                    self.minkowski_p,
                );
                new_list.insert(*existing_index, distance);
            }
        }
        self.distance_lists.insert(new_index, new_list);
    }

    /// Aktualisiert Distanzlisten, wenn ein Observer entfernt wird
    fn update_distance_lists_on_remove(&mut self, removed_index: usize) {
        // Entferne die Distanzliste des entfernten Observers
        self.distance_lists.remove(&removed_index);

        // Entferne Einträge für diesen Observer aus allen anderen Distanzlisten
        for list in self.distance_lists.values_mut() {
            list.remove(removed_index);
        }
    }

    /// Baut alle Distanzlisten neu auf (für initiale Berechnung oder wenn sich viele Observer geändert haben)
    pub(crate) fn rebuild_distance_lists(&mut self) {
        self.distance_lists.clear();

        let indices: Vec<usize> = self.observers_by_index.keys().cloned().collect();
        let data_map: HashMap<usize, &[f64]> = self
            .observers_by_index
            .iter()
            .map(|(idx, arc)| (*idx, arc.data.as_slice()))
            .collect();

        for &i in &indices {
            let mut list = OrderedDistanceList::new();
            let data_i = data_map[&i];

            for &j in &indices {
                if i != j {
                    let data_j = data_map[&j];
                    let distance =
                        compute_distance(data_i, data_j, self.distance_metric, self.minkowski_p);
                    list.insert(j, distance);
                }
            }
            self.distance_lists.insert(i, list);
        }
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
        let data = observer_arc.data.clone();
        self.observers_by_index.insert(index, observer_arc);
        self.indices_by_obs.insert(obs_key, index);
        self.indices_by_score.insert(score_key, index);

        // Update last_active_observer cache, da sich die Observer-Liste geändert hat
        self.update_last_active_observer();

        // Update distance lists für alle Observer
        self.update_distance_lists_on_insert(index, &data);
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
                    label: observer_arc.label,
                    cluster_observations: observer_arc.cluster_observations.clone(),
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

        self.indices_by_obs.insert(new_obs_key, index);
        self.indices_by_score.insert(new_score_key, index);

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
            .filter_map(|(_key, &index)| {
                self.observers_by_index
                    .get(&index)
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
            .map(|(_key, &index)| index)
            .collect();
        indices
            .into_iter()
            .filter_map(move |index| self.observers_by_index.get(&index).map(|arc| arc.as_ref()))
    }

    /// Remove an observer by index - O(log n)
    pub fn remove(&mut self, index: usize) -> Option<Observer> {
        // Get the observer Arc to remove (need values for keys)
        let observer_arc = self.observers_by_index.get(&index)?;

        // Create keys to remove from secondary indices
        let obs_key = ObservationKey {
            observations: crate::obs::OrderedFloat(observer_arc.observations),
            index,
        };
        let normalized_score = if observer_arc.age > 0.0 {
            observer_arc.observations / observer_arc.age
        } else {
            f64::INFINITY
        };
        let score_key = NormalizedScoreKey {
            score: OrderedFloat(normalized_score),
            index,
        };

        // Remove from all structures
        let observer_arc = self.observers_by_index.remove(&index)?;
        self.indices_by_obs.remove(&obs_key);
        self.indices_by_score.remove(&score_key);

        // Update last_active_observer cache, da sich die Observer-Liste geändert hat
        self.update_last_active_observer();

        // Update distance lists: Entferne diesen Observer aus allen Distanzlisten
        self.update_distance_lists_on_remove(index);

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

    /// Get iterator over all observers (unsorted)
    #[allow(dead_code)] // Für zukünftige Verwendung
    pub fn iter(&self) -> impl Iterator<Item = &Observer> {
        self.observers_by_index.values().map(|arc| arc.as_ref())
    }

    /// Find the worst observer by normalized score - O(1)
    /// By default k = 1
    pub fn find_k_worst_normalized_scores(&self, k: Option<usize>) -> Vec<(usize, f64)> {
        self.indices_by_score
            .iter()
            .take(k.unwrap_or(1))
            .map(|(key, &index)| (index, key.score.0))
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

    /// Update observer label - O(1)
    pub fn update_label(&mut self, index: usize, label: Option<i32>) -> bool {
        if let Some(arc) = self.observers_by_index.get_mut(&index) {
            if let Some(mut_observer) = Arc::get_mut(arc) {
                // Exclusive access - update in place (no clone!)
                mut_observer.label = label;
                true
            } else {
                // Shared - create new Arc with updated label
                let observer_arc = self.observers_by_index.get(&index).unwrap().clone();
                let updated_observer = Arc::new(Observer {
                    data: observer_arc.data.clone(),
                    observations: observer_arc.observations,
                    time: observer_arc.time,
                    age: observer_arc.age,
                    index: observer_arc.index,
                    label,
                    cluster_observations: observer_arc.cluster_observations.clone(),
                });
                self.observers_by_index.insert(index, updated_observer);
                true
            }
        } else {
            false
        }
    }

    /// Update cluster observations vector - O(1)
    pub fn update_cluster_observations(
        &mut self,
        index: usize,
        cluster_observations: Vec<f64>,
    ) -> bool {
        if let Some(arc) = self.observers_by_index.get_mut(&index) {
            if let Some(mut_observer) = Arc::get_mut(arc) {
                // Exclusive access - update in place (no clone!)
                mut_observer.cluster_observations = cluster_observations;
                true
            } else {
                // Shared - create new Arc with updated cluster_observations
                let observer_arc = self.observers_by_index.get(&index).unwrap().clone();
                let updated_observer = Arc::new(Observer {
                    data: observer_arc.data.clone(),
                    observations: observer_arc.observations,
                    time: observer_arc.time,
                    age: observer_arc.age,
                    index: observer_arc.index,
                    label: observer_arc.label,
                    cluster_observations,
                });
                self.observers_by_index.insert(index, updated_observer);
                true
            }
        } else {
            false
        }
    }

    /// Perform k-nearest neighbor search
    /// Always uses Brute-Force (Tree support disabled)
    pub fn search_k_nearest_distances(
        &self,
        query_point: &[f64],
        k: usize,
        active: bool,
    ) -> Vec<f64> {
        self.brute_force_k_nearest(query_point, k, active)
    }

    /// Brute-Force k-nearest neighbor search für nicht-euklidische Metriken
    fn brute_force_k_nearest(&self, query_point: &[f64], k: usize, active: bool) -> Vec<f64> {
        let mut distances: Vec<(usize, f64)> = self
            .iter_observers(active)
            .map(|obs| {
                (
                    obs.index,
                    compute_distance(
                        &obs.data,
                        query_point,
                        self.distance_metric,
                        self.minkowski_p,
                    ),
                )
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        distances
            .into_iter()
            .take(k)
            .map(|(_, dist)| dist)
            .collect()
    }

    /// Find k nearest observer indices (not just distances)
    /// Returns indices of the k nearest observers to the query point
    /// Always uses Brute-Force (Tree support disabled)
    pub fn search_k_nearest_indices(
        &self,
        query_point: &[f64],
        k: usize,
        active: bool,
    ) -> Vec<usize> {
        self.brute_force_k_nearest_indices(query_point, k, active)
    }

    /// Brute-Force k-nearest neighbor search für Indizes
    fn brute_force_k_nearest_indices(
        &self,
        query_point: &[f64],
        k: usize,
        active: bool,
    ) -> Vec<usize> {
        let mut observer_distances: Vec<(usize, f64)> = self
            .iter_observers(active)
            .map(|obs| {
                (
                    obs.index,
                    compute_distance(
                        &obs.data,
                        query_point,
                        self.distance_metric,
                        self.minkowski_p,
                    ),
                )
            })
            .collect();

        observer_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        observer_distances
            .into_iter()
            .take(k)
            .map(|(idx, _)| idx)
            .collect()
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
        Self::new()
    }
}
