use kiddo::float::kdtree::KdTree;
use kiddo::SquaredEuclidean;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::utils::{compute_distance, DistanceMetric};

/// Observer-Struktur mit Daten, Beobachtungen und Index
#[derive(Clone, Debug)]
pub struct Observer {
    pub data: Vec<f64>,
    pub observations: f64,
    pub time: f64,
    /// last time the observer was updated
    pub age: f64,
    pub index: usize,
}

// ObserverRTreeObject wurde entfernt - nicht mehr nötig mit kiddo

// Helper struct for comparing floats in collections
#[derive(Debug, Clone, Copy)]
struct OrderedFloat(f64);

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
struct ObservationKey {
    observations: OrderedFloat,
    index: usize,
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
struct NormalizedScoreKey {
    score: OrderedFloat,
    index: usize,
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

/// Räumliche Indexierung: kiddo KD-Tree nur für euklidische Distanz
/// Für andere Metriken wird kein Baum verwendet (Brute-Force)
/// kiddo ist für euklidische Distanzen optimiert und unterstützt dynamische Updates
/// Maximale Dimension: 100 (kiddo benötigt Dimension zur Compile-Zeit)
/// B = 64: Bucket-Größe für den KD-Tree (erhöht für Duplikate)
/// IDX = u32: Index-Typ für interne Strukturen
pub(crate) enum SpatialTreeObserver {
    /// kiddo KD-Tree für euklidische Distanz (bis zu 100 Dimensionen)
    Euclidean(KdTree<f64, usize, 100, 64, u32>),
    /// Kein Baum für nicht-euklidische Metriken (Brute-Force wird verwendet)
    NonEuclidean,
}

/// Efficient ObserverSet with dual indexing for O(log n) operations
/// Includes spatial tree for efficient k-NN operations
/// Uses Arc<Observer> for shared ownership and direct access without HashMap lookups
pub struct ObserverSet {
    // Primary storage: O(1) access by index, Arc for shared ownership
    observers_by_index: HashMap<usize, Arc<Observer>>,

    // Secondary index: sorted by observations (descending) - stores only indices
    // Key: (observations, index) -> index
    // BTreeMap gives us O(log n) for min/max and sorted iteration
    indices_by_obs: BTreeMap<ObservationKey, usize>,

    // Tertiary index: sorted by normalized score (observations/age, ascending) - stores only indices
    // Key: (normalized_score, index) -> index
    // This allows O(log n) finding of worst observer
    indices_by_score: BTreeMap<NormalizedScoreKey, usize>,

    // R*-Tree spatial index for efficient k-NN operations with incremental updates
    // Lazy-initialized and supports incremental insertions and removals
    spatial_tree: RefCell<Option<SpatialTreeObserver>>,

    // Parameters for R*-Tree construction
    distance_metric: DistanceMetric,
    minkowski_p: Option<f64>,
}

impl ObserverSet {
    pub fn new() -> Self {
        Self {
            observers_by_index: HashMap::new(),
            indices_by_obs: BTreeMap::new(),
            indices_by_score: BTreeMap::new(),
            spatial_tree: RefCell::new(None),
            distance_metric: DistanceMetric::Euclidean,
            minkowski_p: None,
        }
    }

    /// Set tree parameters and invalidate existing tree
    pub fn set_tree_params(&mut self, distance_metric: DistanceMetric, minkowski_p: Option<f64>) {
        self.distance_metric = distance_metric;
        self.minkowski_p = minkowski_p;
        // Invalidate tree when parameters change
        if let Ok(mut tree_opt) = self.spatial_tree.try_borrow_mut() {
            *tree_opt = None;
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

        // Incrementally add to kiddo KD-Tree if it exists (nur für euklidische Distanz)
        // Sonst wird der Baum lazy gebaut wenn nötig
        self.insert_into_tree(index, &data);
    }

    /// Get observer by index - O(1)
    #[allow(dead_code)] // Für zukünftige Verwendung
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
            observations: OrderedFloat(observer_arc.observations),
            index,
        };
        let old_normalized_score = if observer_arc.age > 0.0 {
            observer_arc.observations / observer_arc.age
        } else {
            f64::INFINITY
        };
        let old_score_key = NormalizedScoreKey {
            score: OrderedFloat(old_normalized_score),
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
                })
            }
        };

        // Update HashMap with new Arc
        self.observers_by_index.insert(index, updated_observer);

        // Re-insert with updated values
        let new_obs_key = ObservationKey {
            observations: OrderedFloat(new_observations),
            index,
        };
        let new_normalized_score = if new_age > 0.0 {
            new_observations / new_age
        } else {
            f64::INFINITY
        };
        let new_score_key = NormalizedScoreKey {
            score: OrderedFloat(new_normalized_score),
            index,
        };

        self.indices_by_obs.insert(new_obs_key, index);
        self.indices_by_score.insert(new_score_key, index);

        true
    }

    /// Get top N observers by observations - O(N)
    /// Clones observers - use get_active_arcs() for better performance
    pub fn get_active(&self, num_active: usize) -> Vec<Observer> {
        self.indices_by_obs
            .iter()
            .take(num_active)
            .filter_map(|(_key, &index)| {
                self.observers_by_index
                    .get(&index)
                    .map(|arc| (**arc).clone())
            })
            .collect()
    }

    /// Get iterator over active observers (top N by observations) - O(1) to create, O(N) to iterate
    /// More efficient than get_active when you only need to iterate without cloning
    #[allow(dead_code)] // Für zukünftige Verwendung
    pub fn iter_active(&self, num_active: usize) -> impl Iterator<Item = &Observer> {
        // Collect indices first, then map to Arc dereferences
        let indices: Vec<usize> = self
            .indices_by_obs
            .iter()
            .take(num_active)
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
            observations: OrderedFloat(observer_arc.observations),
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

        // Remove Observer from Spatial Tree
        self.remove_from_tree(index);

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
    pub fn find_worst_normalized_score(&self) -> Option<(usize, f64)> {
        self.indices_by_score
            .first_key_value() // Gets the smallest (worst) score
            .and_then(|(key, &index)| {
                self.observers_by_index
                    .get(&index)
                    .map(|_| (index, key.score.0))
            })
    }

    /// Update only observations - O(log n)
    #[allow(dead_code)] // Für zukünftige Verwendung
    pub fn update_observations(&mut self, index: usize, new_observations: f64) -> bool {
        if let Some(observer) = self.observers_by_index.get(&index) {
            let current_age = observer.age;
            self.update_observer(index, new_observations, current_age)
        } else {
            false
        }
    }

    /// Update only age - O(log n)
    #[allow(dead_code)] // Für zukünftige Verwendung
    pub fn update_age(&mut self, index: usize, new_age: f64) -> bool {
        if let Some(observer) = self.observers_by_index.get(&index) {
            let current_obs = observer.observations;
            self.update_observer(index, current_obs, new_age)
        } else {
            false
        }
    }

    /// Update observations and age - O(log n)
    #[allow(dead_code)] // Für zukünftige Verwendung
    pub fn update_observations_and_age(
        &mut self,
        index: usize,
        new_observations: f64,
        new_age: f64,
    ) -> bool {
        if self.observers_by_index.contains_key(&index) {
            self.update_observer(index, new_observations, new_age)
        } else {
            false
        }
    }

    /// Update observations, age, and time - O(log n)
    /// Wichtig für zeitbasierte Updates in SDOstream
    pub fn update_observer_with_time(
        &mut self,
        index: usize,
        new_observations: f64,
        new_age: f64,
        new_time: f64,
    ) -> bool {
        // Get the current observer Arc
        let observer_arc = match self.observers_by_index.get(&index) {
            Some(arc) => arc.clone(),
            None => return false,
        };

        // Remove old entries from secondary indices using old values
        let old_obs_key = ObservationKey {
            observations: OrderedFloat(observer_arc.observations),
            index,
        };
        let old_normalized_score = if observer_arc.age > 0.0 {
            observer_arc.observations / observer_arc.age
        } else {
            f64::INFINITY
        };
        let old_score_key = NormalizedScoreKey {
            score: OrderedFloat(old_normalized_score),
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
                mut_observer.time = new_time;
                Arc::clone(arc_mut) // Clone the Arc reference, not the Observer
            } else {
                // Shared - create new Arc with updated values
                Arc::new(Observer {
                    data: observer_arc.data.clone(),
                    observations: new_observations,
                    time: new_time,
                    age: new_age,
                    index: observer_arc.index,
                })
            }
        };

        // Update HashMap with new Arc
        self.observers_by_index.insert(index, updated_observer);

        // Re-insert with updated values
        let new_obs_key = ObservationKey {
            observations: OrderedFloat(new_observations),
            index,
        };
        let new_normalized_score = if new_age > 0.0 {
            new_observations / new_age
        } else {
            f64::INFINITY
        };
        let new_score_key = NormalizedScoreKey {
            score: OrderedFloat(new_normalized_score),
            index,
        };

        self.indices_by_obs.insert(new_obs_key, index);
        self.indices_by_score.insert(new_score_key, index);

        true
    }

    /// Get spatial tree (builds if necessary)
    #[allow(dead_code)] // Für zukünftige Verwendung
    pub fn get_spatial_tree(&self) -> Option<std::cell::Ref<'_, Option<SpatialTreeObserver>>> {
        Some(self.spatial_tree.borrow())
    }

    /// Stellt sicher, dass der Spatial Tree gebaut ist (lazy initialization)
    /// Nur für euklidische Distanz wird ein kiddo KD-Tree gebaut
    /// Diese Methode ist thread-safe durch RefCell, aber nicht re-entrant
    pub fn ensure_spatial_tree(&self) {
        if self.distance_metric != DistanceMetric::Euclidean {
            return; // Kein Baum für nicht-euklidische Metriken
        }

        // Prüfe zuerst, ob der Baum bereits existiert (immutable borrow)
        // Verwende try_borrow() um zu vermeiden, dass wir panicken wenn bereits geborrowt
        if let Ok(tree_opt) = self.spatial_tree.try_borrow() {
            if tree_opt.is_some() {
                return; // Baum existiert bereits
            }
        } else {
            // RefCell ist bereits geborrowt - das bedeutet, dass der Baum gerade gebaut wird
            // oder bereits existiert. In diesem Fall geben wir einfach zurück.
            return;
        }

        // Jetzt mutable borrow für den Build
        // Verwende try_borrow_mut() um zu vermeiden, dass wir panicken wenn bereits geborrowt
        if let Ok(mut tree_opt) = self.spatial_tree.try_borrow_mut() {
            if tree_opt.is_none() {
                // Baue kiddo KD-Tree aus allen aktuellen Observers (nur für Euclidean)
                let all_observers: Vec<Arc<Observer>> =
                    self.observers_by_index.values().cloned().collect();
                *tree_opt = Self::build_tree_from_observers(&all_observers, self.distance_metric);
            }
        }
        // Wenn try_borrow_mut() fehlschlägt, bedeutet das, dass der Baum gerade gebaut wird
        // oder bereits existiert. In diesem Fall geben wir einfach zurück.
    }

    /// Perform k-nearest neighbor search
    /// Für euklidische Distanz: verwendet kiddo KD-Tree
    /// Für andere Metriken: Brute-Force
    pub fn search_k_nearest(&self, query_point: &[f64], k: usize) -> Vec<f64> {
        match self.distance_metric {
            DistanceMetric::Euclidean => {
                // Verwende kiddo KD-Tree für euklidische Distanz
                // Stelle sicher, dass der Baum existiert
                self.ensure_spatial_tree();

                // Separate Borrow nach ensure_spatial_tree() (Borrow sollte zurückgegeben sein)
                let tree_opt = self.spatial_tree.borrow();
                if let Some(SpatialTreeObserver::Euclidean(ref kdtree)) = *tree_opt {
                    // Prüfe Dimension - kiddo unterstützt bis zu 100 Dimensionen
                    if query_point.len() > 100 {
                        // Fallback zu Brute-Force für sehr hohe Dimensionen
                        return self.brute_force_k_nearest(query_point, k);
                    }

                    // kiddo API: nearest_n benötigt &[f64; 100] - pad query_point
                    let mut padded_query = [0.0; 100];
                    for (i, &val) in query_point.iter().take(100).enumerate() {
                        padded_query[i] = val;
                    }

                    // kiddo API: nearest_n gibt Vec<NearestNeighbor> zurück
                    // NearestNeighbor hat .distance und .item (der Index)
                    // Verwende SquaredEuclidean für euklidische Distanz
                    let neighbors = kdtree.nearest_n::<SquaredEuclidean>(&padded_query, k);
                    // kiddo gibt quadrierte Distanz zurück, wir müssen die Wurzel ziehen
                    neighbors.iter().map(|n| n.distance.sqrt()).collect()
                } else {
                    self.brute_force_k_nearest(query_point, k)
                }
            }
            _ => {
                // Für andere Metriken: Brute-Force
                self.brute_force_k_nearest(query_point, k)
            }
        }
    }

    /// Brute-Force k-nearest neighbor search für nicht-euklidische Metriken
    fn brute_force_k_nearest(&self, query_point: &[f64], k: usize) -> Vec<f64> {
        let mut distances: Vec<(usize, f64)> = self
            .observers_by_index
            .iter()
            .map(|(&index, arc)| {
                let dist = compute_distance(
                    &arc.data,
                    query_point,
                    self.distance_metric,
                    self.minkowski_p,
                );
                (index, dist)
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
    /// Optimiert für euklidische Distanz mit kiddo KD-Tree
    pub fn search_k_nearest_indices(&self, query_point: &[f64], k: usize) -> Vec<usize> {
        match self.distance_metric {
            DistanceMetric::Euclidean => {
                // Verwende kiddo KD-Tree für euklidische Distanz
                self.ensure_spatial_tree();

                let tree_opt = self.spatial_tree.borrow();
                if let Some(SpatialTreeObserver::Euclidean(ref kdtree)) = *tree_opt {
                    // Prüfe Dimension - kiddo unterstützt bis zu 100 Dimensionen
                    if query_point.len() <= 100 {
                        // kiddo API: nearest_n benötigt &[f64; 100] - pad query_point
                        let mut padded_query = [0.0; 100];
                        for (i, &val) in query_point.iter().take(100).enumerate() {
                            padded_query[i] = val;
                        }

                        // kiddo API: nearest_n gibt Vec<NearestNeighbor> zurück
                        // NearestNeighbor hat .distance und .item (der Index)
                        let neighbors = kdtree.nearest_n::<SquaredEuclidean>(&padded_query, k);
                        return neighbors.iter().map(|n| n.item).collect();
                    }
                }
            }
            _ => {}
        }

        // Fallback: Brute-Force für nicht-euklidische Metriken oder sehr hohe Dimensionen
        let mut observer_distances: Vec<(usize, f64)> = self
            .observers_by_index
            .iter()
            .map(|(&index, arc)| {
                let dist = compute_distance(
                    &arc.data,
                    query_point,
                    self.distance_metric,
                    self.minkowski_p,
                );
                (index, dist)
            })
            .collect();

        observer_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        observer_distances
            .into_iter()
            .take(k)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Insert observer into kiddo KD-Tree incrementally (nur für euklidische Distanz)
    fn insert_into_tree(&mut self, index: usize, data: &[f64]) {
        if self.distance_metric != DistanceMetric::Euclidean {
            return; // Kein Baum für nicht-euklidische Metriken
        }

        // Prüfe Dimension - kiddo unterstützt bis zu 100 Dimensionen
        if data.len() > 100 {
            return; // Zu viele Dimensionen für kiddo
        }

        let mut tree_opt = self.spatial_tree.borrow_mut();
        if let Some(SpatialTreeObserver::Euclidean(ref mut kdtree)) = *tree_opt {
            // kiddo API: add(point, data) - data ist der Index
            // Punkt muss als Array mit fester Größe übergeben werden
            // Da wir variable Dimensionen haben, müssen wir das Array zur Laufzeit erstellen
            let point_array: Vec<f64> = data.iter().take(100).cloned().collect();
            // Pad auf 100 Dimensionen falls nötig
            let mut padded_point = [0.0; 100];
            for (i, &val) in point_array.iter().enumerate() {
                padded_point[i] = val;
            }
            kdtree.add(&padded_point, index);
        }
    }

    /// Remove observer from kiddo KD-Tree incrementally (nur für euklidische Distanz)
    fn remove_from_tree(&mut self, index: usize) {
        if self.distance_metric != DistanceMetric::Euclidean {
            return; // Kein Baum für nicht-euklidische Metriken
        }

        let mut tree_opt = self.spatial_tree.borrow_mut();
        if let Some(SpatialTreeObserver::Euclidean(ref mut kdtree)) = *tree_opt {
            // Hole den Observer, um den Punkt zu bekommen
            if let Some(observer_arc) = self.observers_by_index.get(&index) {
                let data = &observer_arc.data;
                if data.len() > 100 {
                    return; // Zu viele Dimensionen
                }

                // Erstelle Punkt-Array für kiddo
                let point_array: Vec<f64> = data.iter().take(100).cloned().collect();
                let mut padded_point = [0.0; 100];
                for (i, &val) in point_array.iter().enumerate() {
                    padded_point[i] = val;
                }

                // kiddo API: remove(point, item) entfernt den Punkt mit dem gegebenen Index
                kdtree.remove(&padded_point, index);
            }
        }
    }

    /// Build kiddo KD-Tree from observers (nur für euklidische Distanz)
    /// Für andere Metriken wird kein Baum gebaut (NonEuclidean)
    fn build_tree_from_observers(
        observers: &[Arc<Observer>],
        metric: DistanceMetric,
    ) -> Option<SpatialTreeObserver> {
        if observers.is_empty() {
            return None;
        }

        match metric {
            DistanceMetric::Euclidean => {
                let mut kdtree: KdTree<f64, usize, 100, 64, u32> = KdTree::new();

                for arc in observers {
                    let data = &arc.data;
                    // Prüfe Dimension
                    if data.len() > 100 {
                        // Überspringe Observer mit zu vielen Dimensionen
                        continue;
                    }

                    // Erstelle Punkt-Array für kiddo (pad auf 100 Dimensionen)
                    let point_array: Vec<f64> = data.iter().take(100).cloned().collect();
                    let mut padded_point = [0.0; 100];
                    for (i, &val) in point_array.iter().enumerate() {
                        padded_point[i] = val;
                    }

                    // kiddo API: add(point, data) - data ist der Index
                    kdtree.add(&padded_point, arc.index);
                }

                Some(SpatialTreeObserver::Euclidean(kdtree))
            }
            _ => {
                // Für andere Metriken: kein Baum, Brute-Force wird verwendet
                Some(SpatialTreeObserver::NonEuclidean)
            }
        }
    }

    /// Clear all observers - O(n)
    #[allow(dead_code)] // Für zukünftige Verwendung
    pub fn clear(&mut self) {
        self.observers_by_index.clear();
        self.indices_by_obs.clear();
        self.indices_by_score.clear();
        *self.spatial_tree.borrow_mut() = None;
    }
}

impl Default for ObserverSet {
    fn default() -> Self {
        Self::new()
    }
}

// Alte Observer-Wrapper wurden entfernt - wir verwenden jetzt ObserverIndex* Wrapper
// die indexbasiert arbeiten und keine Klonierung benötigen
