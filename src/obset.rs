use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};

use ahash::AHashMap;

use rayon::prelude::*;

use mtree::distance::{Distance, EuclideanDistance};
use mtree::node::ObjectNode;
use mtree::{MTree, Point};
use statrs::distribution::{DiscreteCDF, Hypergeometric};
use std::sync::Arc;

use crate::obs::{NeighborInfo, OrderedFloat};
use crate::utils::{compute_distance, DistanceMetric};

/// Custom-Distanz für nicht-euklidische Metriken (Manhattan, Chebyshev, Minkowski).
/// Wird im MTree und in distance_from_point genutzt, damit überall dieselbe Metrik gilt.
#[derive(Clone, Copy)]
struct MetricDistance(DistanceMetric, Option<f64>);

impl Distance<Point> for MetricDistance {
    type Output = f64;

    fn distance(&self, a: &Point, b: &Point) -> f64 {
        compute_distance(&a.0, &b.0, self.0, self.1)
    }

    fn clone_box(&self) -> Box<dyn Distance<Point, Output = f64> + Send + Sync> {
        Box::new(MetricDistance(self.0, self.1))
    }
}

/// Erstellt das MTree mit der konfigurierten Distanz: Euclidean → mtree, sonst MetricDistance.
fn make_mtree(distance_metric: DistanceMetric, minkowski_p: Option<f64>) -> MTree<Point, usize, f64> {
    if distance_metric == DistanceMetric::Euclidean {
        MTree::with_distance(EuclideanDistance)
    } else {
        MTree::with_distance(MetricDistance(distance_metric, minkowski_p))
    }
}

/// Entry für observations_list: sortiert nach gefadeten observations (descending), dann index
#[derive(Clone, Debug, Copy)]
struct ObservationEntry {
    observations: OrderedFloat,
    index: usize,
    time: f64,
    fading: Option<OrderedFloat>, // Fading-Parameter für gefadete Sortierung
}

/// Entry für label_observations: speichert unverfadete observations und time pro Label
#[derive(Clone, Debug)]
pub struct LabelObservationEntry {
    /// Unverfadeter Wert
    pub observations: f64,
    /// Zeitpunkt der letzten Aktualisierung
    pub time: f64,
}

impl PartialEq for ObservationEntry {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl Eq for ObservationEntry {}

impl PartialOrd for ObservationEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ObservationEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Verwende fading aus Entry oder 1.0 als Fallback
        let fading_a = self.fading.map(|f| f.0).unwrap_or(1.0);
        let fading_b = other.fading.map(|f| f.0).unwrap_or(1.0);

        // Verwende common_touched = max(a.time, b.time) wie in C++ Implementierung
        let common_touched = self.time.max(other.time);

        // Berechne gefadete Werte: observations * fading.powf(common_touched - time)
        let faded_a = self.observations.0 * fading_a.powf(common_touched - self.time);
        let faded_b = other.observations.0 * fading_b.powf(common_touched - other.time);

        // Sortiere nach gefadeten Werten (descending) - reverse comparison
        let ordering = faded_b.partial_cmp(&faded_a).unwrap_or(Ordering::Equal);

        // Tie-breaker: index (ascending)
        ordering.then(self.index.cmp(&other.index))
    }
}

/// Helper function to convert Vec<f64> to mtree::Point
fn vec_to_point(data: &[f64]) -> Point {
    Point::new(data.to_vec())
}

/// Calculate minimum sample size using hypergeometric distribution
/// Returns the minimum number of observers to sample to ensure with probability p_required
/// that at least x_target active observers are included
pub(crate) fn min_sample_size_hypergeometric(
    n_pop: u64,      // Total observers (k)
    k_success: u64,  // Active observers (num_active)
    x_target: u64,   // Required active neighbors (x)
    p_required: f64, // Required probability (p_safe)
) -> Option<u64> {
    // Edge cases
    if x_target == 0 {
        return Some(0);
    }
    if k_success < x_target {
        return None; // Not enough active observers
    }
    if n_pop < x_target {
        return None; // Not enough total observers
    }

    for n_sample in x_target..=n_pop {
        let hyper = match Hypergeometric::new(n_pop, k_success, n_sample) {
            Ok(h) => h,
            Err(_) => continue,
        };
        // P(X < x_target) = cdf(x_target - 1)
        let x_target_minus_one = if x_target > 0 { x_target - 1 } else { 0 };
        let prob = hyper.cdf(x_target_minus_one);
        if 1.0 - prob >= p_required {
            return Some(n_sample);
        }
    }
    None
}

/// Efficient ObserverSet with mtree for spatial search and flat data structures
pub struct ObserverSet {
    // MTree für räumliche Suche: Key = Point (mtree::Point mit Hash + Eq), Value = global unique index (usize)
    mtree: MTree<Point, usize, f64>,

    // Point-Lookup: Index -> Point (für mtree Updates und Datenzugriff)
    // Point enthält Vec<f64> als Tuple-Struct Point(pub Vec<f64>), daher können wir direkt darauf zugreifen
    point_by_index: AHashMap<usize, Point>,

    // Observations-Liste: sortiert nach observations (descending), durchsuchbar nach index
    observations_list: BTreeSet<ObservationEntry>,

    // Index-Lookup für observations_list (für O(log n) Updates)
    index_to_obs_entry: AHashMap<usize, ObservationEntry>,

    // Age-Map: Index -> age
    age_by_index: AHashMap<usize, f64>,

    // Label observations: Index -> HashMap<label, LabelObservationEntry>
    label_observations_by_index: AHashMap<usize, HashMap<usize, LabelObservationEntry>>,

    // Local thresholds: Index -> local_threshold
    local_threshold_by_index: AHashMap<usize, f64>,

    // Globale Parameter
    distance_metric: DistanceMetric,
    minkowski_p: Option<f64>,
    fading: Option<f64>, // Fading-Parameter f = exp(-T^-1) für gefadete Sortierung
    num_active: usize,
    global_threshold: f64,
    last_label: usize,
    last_active_observer: Option<usize>,
}

impl ObserverSet {
    pub fn new(
        distance_metric: DistanceMetric,
        minkowski_p: Option<f64>,
        fading: Option<f64>,
    ) -> Self {
        // MTree mit konfigurierter Distanz: Euclidean → mtree (inkl. SIMD), sonst MetricDistance
        let mtree = make_mtree(distance_metric, minkowski_p);

        Self {
            mtree,
            point_by_index: AHashMap::default(),
            observations_list: BTreeSet::new(),
            index_to_obs_entry: AHashMap::default(),
            age_by_index: AHashMap::default(),
            label_observations_by_index: AHashMap::default(),
            local_threshold_by_index: AHashMap::default(),
            distance_metric: distance_metric,
            minkowski_p: minkowski_p,
            fading: fading,
            num_active: 0,
            global_threshold: f64::INFINITY,
            last_label: 0,
            last_active_observer: None,
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
            .observations_list
            .iter()
            .nth(self.num_active - 1)
            .map(|entry| entry.index);
    }

    /// Prüft, ob ein Observer aktiv ist (gehört zu den Top num_active Observern nach observations)
    /// Ein Observer ist aktiv, wenn seine observations >= observations des last_active_observer sind
    /// O(1) - verwendet gecachten last_active_observer
    pub(crate) fn is_active(&self, index: usize) -> bool {
        if self.num_active == 0 {
            return false; // Keine aktiven Observer definiert
        }

        // Hole den Observer-Eintrag
        let entry = match self.index_to_obs_entry.get(&index) {
            Some(e) => e,
            None => return false, // Observer existiert nicht
        };

        // Verwende gecachten last_active_observer
        match self.last_active_observer {
            Some(last_active_idx) => {
                // Hole den last_active_observer Eintrag
                let last_active_entry = match self.index_to_obs_entry.get(&last_active_idx) {
                    Some(e) => e,
                    None => {
                        // Cache ist veraltet, alle Observer sind aktiv
                        return true;
                    }
                };
                // Observer ist aktiv, wenn seine observations >= observations des last_active_observer
                entry.observations >= last_active_entry.observations
            }
            None => {
                // Wenn es weniger als num_active Observer gibt, sind alle aktiv
                true
            }
        }
    }

    /// Insert a new observer - O(log n)
    pub fn insert(&mut self, index: usize, data: Vec<f64>, observations: f64, time: f64, age: f64) {
        // Wenn ein Observer mit diesem Index bereits existiert, raise an error
        if self.point_by_index.contains_key(&index) {
            panic!("Observer with index {} already exists", index);
        }

        // Konvertiere data zu Point
        let point = vec_to_point(&data);

        // Füge zu mtree hinzu
        self.mtree.insert(point.clone(), index);

        // Speichere Point (enthält bereits Vec<f64>)
        self.point_by_index.insert(index, point);

        // Erstelle ObservationEntry mit fading
        let obs_entry = ObservationEntry {
            observations: OrderedFloat(observations),
            index,
            time,
            fading: self.fading.map(OrderedFloat),
        };

        // Füge zu observations_list hinzu
        self.observations_list.insert(obs_entry);
        self.index_to_obs_entry.insert(index, obs_entry);

        // Füge zu age_by_index hinzu
        self.age_by_index.insert(index, age);

        // Initialisiere label_observations, local_threshold
        self.label_observations_by_index
            .insert(index, HashMap::new());
        self.local_threshold_by_index.insert(index, f64::INFINITY);

        // Update last_active_observer cache
        self.update_last_active_observer();
    }

    /// Get data by index - O(1)
    /// Returns reference to Vec<f64> from Point
    pub fn get_data(&self, index: usize) -> Option<&Vec<f64>> {
        self.point_by_index.get(&index).map(|p| &p.0)
    }

    /// Get observations by index - O(1)
    pub fn get_observations(&self, index: usize) -> Option<f64> {
        self.index_to_obs_entry
            .get(&index)
            .map(|e| e.observations.0)
    }

    /// Get age by index - O(1)
    pub fn get_age(&self, index: usize) -> Option<f64> {
        self.age_by_index.get(&index).copied()
    }

    /// Get label observations by index with fading applied - O(n) where n is number of labels
    /// Returns HashMap with faded values: observations * fading.powf(current_time - time)
    pub fn get_label_observations(
        &self,
        index: usize,
        current_time: f64,
    ) -> Option<HashMap<usize, f64>> {
        let raw = self.label_observations_by_index.get(&index)?;
        let fading = self.fading.unwrap_or(1.0); // Wenn kein fading, verwende 1.0 (kein Fading)

        Some(
            raw.iter()
                .map(|(&label, entry)| {
                    let time_diff = current_time - entry.time;
                    let faded_value = entry.observations * fading.powf(time_diff);
                    (label, faded_value)
                })
                .collect(),
        )
    }

    /// Get raw label observations by index (for updates) - O(1)
    /// Returns HashMap with LabelObservationEntry (unfaded observations and time)
    pub fn get_label_observations_raw(
        &self,
        index: usize,
    ) -> Option<&HashMap<usize, LabelObservationEntry>> {
        self.label_observations_by_index.get(&index)
    }

    /// Get local threshold by index - O(1)
    pub fn get_local_threshold(&self, index: usize) -> Option<f64> {
        self.local_threshold_by_index.get(&index).copied()
    }

    /// Get time by index - O(1)
    pub fn get_time(&self, index: usize) -> Option<f64> {
        self.index_to_obs_entry.get(&index).map(|e| e.time)
    }

    /// Update observer's observations and age - O(log n)
    pub fn update_observer(&mut self, index: usize, new_observations: f64, new_age: f64) -> bool {
        // Hole alte ObservationEntry
        let old_entry = match self.index_to_obs_entry.get(&index).copied() {
            Some(e) => e,
            None => return false,
        };

        // Entferne alte Entry aus observations_list
        self.observations_list.remove(&old_entry);

        // Hole alte time und fading
        let time = old_entry.time;
        let fading = old_entry.fading;

        // Erstelle neue ObservationEntry mit aktualisiertem fading (falls vorhanden)
        let new_entry = ObservationEntry {
            observations: OrderedFloat(new_observations),
            index,
            time,
            fading: fading.or(self.fading.map(OrderedFloat)),
        };

        // Füge neue Entry zu observations_list hinzu
        self.observations_list.insert(new_entry);
        self.index_to_obs_entry.insert(index, new_entry);

        // Aktualisiere age_by_index
        self.age_by_index.insert(index, new_age);

        // Update last_active_observer cache
        self.update_last_active_observer();

        true
    }

    /// Update observer's observations, age, and time - O(log n)
    pub fn update_observer_with_time(
        &mut self,
        index: usize,
        new_observations: f64,
        new_age: f64,
        new_time: f64,
    ) -> bool {
        // Hole alte ObservationEntry
        let old_entry = match self.index_to_obs_entry.get(&index).copied() {
            Some(e) => e,
            None => return false,
        };

        // Entferne alte Entry aus observations_list
        self.observations_list.remove(&old_entry);

        // Hole fading aus alter Entry oder verwende self.fading
        let fading = old_entry.fading.or(self.fading.map(OrderedFloat));

        // Erstelle neue ObservationEntry mit aktualisiertem fading
        let new_entry = ObservationEntry {
            observations: OrderedFloat(new_observations),
            index,
            time: new_time,
            fading,
        };

        // Füge neue Entry zu observations_list hinzu
        self.observations_list.insert(new_entry);
        self.index_to_obs_entry.insert(index, new_entry);

        // Aktualisiere age_by_index
        self.age_by_index.insert(index, new_age);

        // Update last_active_observer cache
        self.update_last_active_observer();

        true
    }

    /// Update observer data - O(n) wegen mtree Rebuild
    /// Note: mtree doesn't have remove, so we rebuild the tree
    /// For better performance, consider keeping track of updates and rebuilding periodically
    pub fn update_data(&mut self, index: usize, new_data: Vec<f64>) -> bool {
        if !self.point_by_index.contains_key(&index) {
            return false;
        }

        // Konvertiere neue Daten zu Point
        let new_point = vec_to_point(&new_data);

        // Aktualisiere point_by_index
        self.point_by_index.insert(index, new_point);

        // Rebuild mtree - sammle alle Einträge und baue neu auf (gleiche Distanz wie bei new)
        let mut new_mtree = make_mtree(self.distance_metric, self.minkowski_p);
        for (&idx, point) in &self.point_by_index {
            new_mtree.insert(point.clone(), idx);
        }
        self.mtree = new_mtree;

        true
    }

    /// Remove an observer by index - O(n) wegen mtree Rebuild
    /// Note: mtree doesn't have remove, so we rebuild the tree
    pub fn remove(&mut self, index: usize) -> Option<Vec<f64>> {
        // Wenn ein Observer mit diesem Index nicht existiert, return None
        if !self.point_by_index.contains_key(&index) {
            return None;
        }

        // Entferne aus point_by_index und extrahiere Vec<f64>
        let point = self.point_by_index.remove(&index)?;
        let data = point.0; // Extrahiere Vec<f64> aus Point

        // Entferne aus observations_list
        if let Some(entry) = self.index_to_obs_entry.get(&index).copied() {
            self.observations_list.remove(&entry);
            self.index_to_obs_entry.remove(&index);
        }

        // Entferne aus age_by_index
        self.age_by_index.remove(&index);

        // Entferne aus label_observations_by_index, local_threshold_by_index
        self.label_observations_by_index.remove(&index);
        self.local_threshold_by_index.remove(&index);

        // Rebuild mtree ohne diesen Eintrag (gleiche Distanz wie bei new)
        let mut new_mtree = make_mtree(self.distance_metric, self.minkowski_p);
        for (&idx, point) in &self.point_by_index {
            new_mtree.insert(point.clone(), idx);
        }
        self.mtree = new_mtree;

        // Update last_active_observer cache
        self.update_last_active_observer();

        Some(data)
    }

    /// Replace an observer - O(log n)
    pub fn replace(
        &mut self,
        old_index: usize,
        new_index: usize,
        new_data: Vec<f64>,
        new_observations: f64,
        new_time: f64,
        new_age: f64,
    ) -> bool {
        // Remove old observer
        if self.remove(old_index).is_none() {
            return false;
        }

        // Insert new observer
        self.insert(new_index, new_data, new_observations, new_time, new_age);
        true
    }

    /// Get number of observers - O(1)
    pub fn len(&self) -> usize {
        self.point_by_index.len()
    }

    /// Check if empty - O(1)
    pub fn is_empty(&self) -> bool {
        self.point_by_index.is_empty()
    }

    /// Get number of active observers - O(1)
    pub fn get_num_active(&self) -> usize {
        self.num_active
    }

    /// Get iterator over observers (sorted by observations)
    pub fn iter_observers(
        &self,
        active: bool,
    ) -> impl Iterator<Item = (usize, &Vec<f64>, f64, f64, f64)> {
        let limit = if active {
            self.num_active
        } else {
            self.observations_list.len()
        };

        self.observations_list
            .iter()
            .take(limit)
            .filter_map(move |entry| {
                let point = self.point_by_index.get(&entry.index)?;
                let age = self.age_by_index.get(&entry.index)?;
                Some((
                    entry.index,
                    &point.0,
                    entry.observations.0,
                    entry.time,
                    *age,
                ))
            })
    }

    /// Find the worst observer by normalized score - O(1)
    /// By default k = 1
    pub fn find_k_worst(&self, k: Option<usize>) -> Vec<usize> {
        if let Some(k_val) = k {
            if k_val < 1 {
                panic!("k must be greater than 0");
            }
        }

        // Lazy calculation: normalized score = observations / age
        let mut scores: Vec<(f64, usize)> = self
            .observations_list
            .iter()
            .filter_map(|entry| {
                let age = self.age_by_index.get(&entry.index)?;
                let score = if *age > 0.0 {
                    entry.observations.0 / *age
                } else {
                    f64::INFINITY
                };
                Some((score, entry.index))
            })
            .collect();

        scores.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        scores
            .iter()
            .take(k.unwrap_or(1))
            .map(|(_, idx)| *idx)
            .collect()
    }

    /// Update observer label observations HashMap - O(1)
    pub fn update_label_observations(
        &mut self,
        index: usize,
        label_observations: HashMap<usize, LabelObservationEntry>,
    ) -> bool {
        if !self.point_by_index.contains_key(&index) {
            return false;
        }
        self.label_observations_by_index
            .insert(index, label_observations);
        true
    }

    /// Update single label observation for an observer - O(1)
    pub fn update_label_observation(
        &mut self,
        index: usize,
        label: usize,
        observations: f64,
        time: f64,
    ) -> bool {
        if !self.point_by_index.contains_key(&index) {
            return false;
        }

        let label_map = self
            .label_observations_by_index
            .entry(index)
            .or_insert_with(HashMap::new);

        label_map.insert(label, LabelObservationEntry { observations, time });
        true
    }

    /// Update observer local threshold - O(1)
    pub fn update_local_threshold(&mut self, index: usize, local_threshold: f64) -> bool {
        if !self.point_by_index.contains_key(&index) {
            return false;
        }
        self.local_threshold_by_index.insert(index, local_threshold);
        true
    }

    /// k-NN search using x_safe sampling strategy
    /// Returns x_safe nearest observers (sorted by distance ascending)
    /// Uses mtree structure directly for efficiency
    pub fn knn_search(
        &self,
        query_point: &[f64],
        x: usize,
    ) -> Vec<(Arc<ObjectNode<Point, usize>>, f64)> {
        let query_point_conv = vec_to_point(query_point);

        // Use mtree knn_search - returns Vec<(Arc<ObjectNode<Point, usize>>, f64)> sorted by distance (ascending)
        self.mtree.knn_search(&query_point_conv, x)
    }

    /// Extract x neighbors from candidates (mtree knn_search result).
    /// Returns Vec<NeighborInfo> for the x closest neighbors (MTree variant).
    /// If active_only is true, only returns active observers; otherwise returns all observers.
    pub fn extract_x_neighbors(
        &self,
        candidates: &[(Arc<ObjectNode<Point, usize>>, f64)],
        x: usize,
        active_only: bool,
    ) -> Vec<NeighborInfo> {
        if active_only {
            candidates
                .iter()
                .filter_map(|(node, dist)| {
                    let idx = node.value.1;
                    if self.is_active(idx) {
                        Some(NeighborInfo::MTree(Arc::clone(node), *dist))
                    } else {
                        None
                    }
                })
                .take(x)
                .collect()
        } else {
            candidates
                .iter()
                .take(x)
                .map(|(node, dist)| NeighborInfo::MTree(Arc::clone(node), *dist))
                .collect()
        }
    }

    /// Batch-Version von knn_search: k-NN für mehrere Punkte in einem Aufruf.
    /// Nutzt parallele mtree knn_search für jeden Punkt.
    /// Returns Vec<Vec<(Arc<ObjectNode<Point, usize>>, f64)>> - mtree structures directly
    pub fn knn_search_batch(
        &self,
        points: &[Vec<f64>],
        x: usize,
    ) -> Vec<Vec<(Arc<ObjectNode<Point, usize>>, f64)>> {
        if points.is_empty() {
            return Vec::new();
        }

        points
            .par_iter()
            .map(|point| {
                let query_point_conv = vec_to_point(point);
                // Use mtree knn_search - returns Vec<(Arc<ObjectNode>, f64)> sorted by distance (ascending)
                self.mtree.knn_search(&query_point_conv, x)
            })
            .collect()
    }

    /// Distanz von einem Punkt zu einem Observer (für Wiederverwendung in SDOstream Step 4).
    /// Bei Euclidean wird die mtree-Distanz (inkl. SIMD) verwendet, sonst compute_distance.
    pub(crate) fn distance_from_point(&self, point: &[f64], observer_index: usize) -> Option<f64> {
        let observer_point = self.point_by_index.get(&observer_index)?;
        let d = if self.distance_metric == DistanceMetric::Euclidean {
            EuclideanDistance::distance_slice(point, &observer_point.0)
        } else {
            compute_distance(&observer_point.0, point, self.distance_metric, self.minkowski_p)
        };
        Some(d)
    }

    /// Finde k nächste Nachbarn für einen Observer unter Verwendung von mtree
    /// O(k log n) mit mtree
    pub fn get_k_nearest_neighbors(&self, observer_index: usize, k: usize) -> Vec<(usize, f64)> {
        let point = match self.point_by_index.get(&observer_index) {
            Some(p) => p,
            None => return Vec::new(),
        };

        let results = self.mtree.knn_search(point, k);
        results
            .into_iter()
            .map(|(node, dist)| (node.value.1, dist))
            .collect()
    }

    /// Finde alle Nachbarn innerhalb eines Thresholds unter Verwendung von mtree
    /// O(log n + m) wobei m die Anzahl der Nachbarn innerhalb des Thresholds ist
    pub fn get_neighbors_within_threshold(
        &self,
        observer_index: usize,
        threshold: f64,
    ) -> Vec<(usize, f64)> {
        let point = match self.point_by_index.get(&observer_index) {
            Some(p) => p,
            None => return Vec::new(),
        };

        let range_query = self.mtree.range_search(point, threshold);
        range_query
            .into_iter()
            .map(|(node, dist)| (node.value.1, dist))
            .collect()
    }

    /// Get observer by index - O(1) - compatibility method
    /// Returns data, observations, time, age tuple
    pub fn get(&self, index: usize) -> Option<(usize, &Vec<f64>, f64, f64, f64)> {
        let point = self.point_by_index.get(&index)?;
        let observations = self.get_observations(index)?;
        let time = self.get_time(index)?;
        let age = self.get_age(index)?;
        Some((index, &point.0, observations, time, age))
    }

    /// Get all observers (for compatibility)
    pub fn get_observers(&self, active: bool) -> Vec<(usize, Vec<f64>, f64, f64, f64)> {
        self.iter_observers(active)
            .map(|(idx, data, obs, time, age)| (idx, data.clone(), obs, time, age))
            .collect()
    }

    /// Update observations only - O(log n)
    pub fn update_observations(&mut self, index: usize, new_observations: f64) -> bool {
        if let Some(age) = self.get_age(index) {
            self.update_observer(index, new_observations, age)
        } else {
            false
        }
    }

    /// Get global threshold
    pub fn get_global_threshold(&self) -> f64 {
        self.global_threshold
    }

    /// Set global threshold
    pub fn set_global_threshold(&mut self, threshold: f64) {
        self.global_threshold = threshold;
    }

    /// Get global threshold (public field access)
    pub fn global_threshold(&self) -> f64 {
        self.global_threshold
    }

    /// Get last label
    pub fn get_last_label(&self) -> usize {
        self.last_label
    }

    /// Set last label
    pub fn set_last_label(&mut self, label: usize) {
        self.last_label = label;
    }

    /// Get last label (public field access)
    pub fn last_label(&self) -> usize {
        self.last_label
    }

    /// Get fading parameter
    pub(crate) fn get_fading(&self) -> Option<f64> {
        self.fading
    }

    /// Rebuild distance lists - no-op for mtree (kept for compatibility)
    pub fn rebuild_distance_lists(&mut self) {
        // No-op: mtree maintains itself automatically
    }

    /// Public version for benchmarks (always available)
    pub fn rebuild_distance_lists_public(&mut self) {
        self.rebuild_distance_lists();
    }

    /// Get label for an observer (helper method)
    /// Returns the label with the highest faded value in label_observations
    pub fn get_label(&self, index: usize, current_time: f64) -> Option<usize> {
        self.get_label_observations(index, current_time)
            .and_then(|lo| {
                lo.iter()
                    .max_by(|(_, &a), (_, &b)| {
                        a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(&label, _)| label)
            })
    }
}

// Clone kann jetzt direkt verwendet werden, da MTree Clone implementiert
impl Clone for ObserverSet {
    fn clone(&self) -> Self {
        Self {
            mtree: self.mtree.clone(),
            point_by_index: self.point_by_index.clone(),
            observations_list: self.observations_list.clone(),
            index_to_obs_entry: self.index_to_obs_entry.clone(),
            age_by_index: self.age_by_index.clone(),
            label_observations_by_index: self.label_observations_by_index.clone(),
            local_threshold_by_index: self.local_threshold_by_index.clone(),
            distance_metric: self.distance_metric,
            minkowski_p: self.minkowski_p,
            fading: self.fading,
            num_active: self.num_active,
            global_threshold: self.global_threshold,
            last_label: self.last_label,
            last_active_observer: self.last_active_observer,
        }
    }
}

impl Default for ObserverSet {
    fn default() -> Self {
        Self::new(DistanceMetric::Euclidean, None, None)
    }
}
