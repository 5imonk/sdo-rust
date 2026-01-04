use acap::distance::Distance;
use acap::kd::KdTree;
use acap::knn::NearestNeighbors;
use acap::vp::VpTree;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::f64;

use crate::sdo_impl::{DistanceMetric, SDOParams, SpatialTreeObserver, TreeType, SDO};
use crate::utils::{
    compute_distance, Observer, ObserverChebyshev, ObserverEuclidean, ObserverManhattan,
    ObserverMinkowski,
};

/// Parameter-Struktur für SDOclust
#[pyclass]
#[derive(Clone)]
pub struct SDOclustParams {
    #[pyo3(get, set)]
    pub k: usize,
    #[pyo3(get, set)]
    pub x: usize,
    #[pyo3(get, set)]
    pub rho: f64,
    #[pyo3(get, set)]
    pub chi: usize,
    #[pyo3(get, set)]
    pub zeta: f64,
    #[pyo3(get, set)]
    pub min_cluster_size: usize,
    #[pyo3(get, set)]
    pub distance: String, // "euclidean", "manhattan", "chebyshev", "minkowski"
    #[pyo3(get, set)]
    pub minkowski_p: Option<f64>, // Für Minkowski-Distanz
    #[pyo3(get, set)]
    pub tree_type: String, // "vptree" (default) oder "kdtree"
}

#[pymethods]
#[allow(clippy::too_many_arguments)]
impl SDOclustParams {
    #[new]
    #[pyo3(signature = (k, x, rho, chi = 4, zeta = 0.5, min_cluster_size = 2, distance = "euclidean".to_string(), minkowski_p = None, tree_type = "vptree".to_string()))]
    pub fn new(
        k: usize,
        x: usize,
        rho: f64,
        chi: usize,
        zeta: f64,
        min_cluster_size: usize,
        distance: String,
        minkowski_p: Option<f64>,
        tree_type: String,
    ) -> Self {
        Self {
            k,
            x,
            rho,
            chi,
            zeta,
            min_cluster_size,
            distance,
            minkowski_p,
            tree_type,
        }
    }
}

/// Sparse Data Observers Clustering (SDOclust) Algorithm
#[pyclass]
pub struct SDOclust {
    sdo: SDO, // Internes SDO-Objekt für Modell-Erstellung
    tree_active: std::cell::RefCell<Option<SpatialTreeObserver>>,
    observer_labels: std::cell::RefCell<Vec<i32>>, // Label für jeden Observer (-1 = entfernt, lazy berechnet)
    #[pyo3(get, set)]
    chi: usize,               // χ - Anzahl der nächsten Observer für lokale Thresholds
    #[pyo3(get, set)]
    zeta: f64,                // ζ - Mixing-Parameter für globale/lokale Thresholds
    #[pyo3(get, set)]
    min_cluster_size: usize,  // e - minimale Clustergröße
}

#[pymethods]
impl SDOclust {
    #[new]
    pub fn new() -> Self {
        Self {
            sdo: SDO::new(),
            tree_active: std::cell::RefCell::new(None),
            observer_labels: std::cell::RefCell::new(Vec::new()),
            chi: 4,
            zeta: 0.5,
            min_cluster_size: 2,
        }
    }

    /// Lernt das Modell aus den Daten und führt Clustering durch
    pub fn learn(&mut self, data: PyReadonlyArray2<f64>, params: &SDOclustParams) -> PyResult<()> {
        let data_slice = data.as_array();
        let rows = data_slice.nrows();

        if rows == 0 || params.k == 0 {
            return Ok(());
        }

        self.chi = params.chi;
        self.zeta = params.zeta;
        self.min_cluster_size = params.min_cluster_size;

        // Verwende SDO für Modell-Erstellung (Sample, Observe, Clean)
        let sdo_params = SDOParams::new(
            params.k,
            params.x,
            params.rho,
            params.distance.clone(),
            params.minkowski_p,
            params.tree_type.clone(),
        );
        self.sdo.learn(data, &sdo_params)?;

        // Reset clustering results, da sich die Parameter geändert haben
        *self.observer_labels.borrow_mut() = Vec::new();
        *self.tree_active.borrow_mut() = None;

        Ok(())
    }

    /// Berechnet den Outlier-Score für einen Datenpunkt (wie SDO)
    pub fn predict_outlier_score(&self, point: PyReadonlyArray2<f64>) -> PyResult<f64> {
        self.sdo.predict(point)
    }

    /// Berechnet das Cluster-Label für einen Datenpunkt
    pub fn predict(&self, point: PyReadonlyArray2<f64>) -> PyResult<i32> {
        // Stelle sicher, dass Clustering durchgeführt wurde
        self.ensure_clustering();

        let active_observers = self.sdo.get_active_observers_internal();
        if active_observers.is_empty() {
            return Ok(-1); // Kein Label (Outlier)
        }

        let point_slice = point.as_array();
        if point_slice.nrows() != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Punkt muss ein 1D-Array oder 2D-Array mit einer Zeile sein",
            ));
        }

        let point_vec: Vec<f64> = (0..point_slice.ncols())
            .map(|j| point_slice[[0, j]])
            .collect();

        // Versuche Tree-basierte Nearest Neighbor Search zu verwenden
        let tree_opt = self.tree_active.borrow();
        if let Some(ref tree) = *tree_opt {
            Ok(self.predict_with_tree(&point_vec, tree))
        } else {
            // Fallback: Brute-Force mit gewählter Distanzfunktion
            Ok(self.predict_brute_force(&point_vec, &active_observers))
        }
    }

    /// Gibt die Anzahl der Cluster zurück
    pub fn n_clusters(&self) -> usize {
        // Stelle sicher, dass Clustering durchgeführt wurde
        self.ensure_clustering();
        let unique_labels: HashSet<i32> = self
            .observer_labels
            .borrow()
            .iter()
            .filter(|&&l| l >= 0)
            .copied()
            .collect();
        unique_labels.len()
    }

    /// Gibt x (Anzahl der Nachbarn) zurück
    #[getter]
    pub fn x(&self) -> usize {
        self.sdo.get_x_internal()
    }

    /// Gibt die Labels der Observer zurück
    pub fn get_observer_labels(&self) -> Vec<i32> {
        // Stelle sicher, dass Clustering durchgeführt wurde
        self.ensure_clustering();
        self.observer_labels.borrow().clone()
    }

    /// Konvertiert active_observers zu NumPy-Array für Python
    pub fn get_active_observers(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        self.sdo.get_active_observers(py)
    }
}

impl SDOclust {
    /// Stellt sicher, dass Clustering durchgeführt wurde (lazy)
    fn ensure_clustering(&self) {
        if self.observer_labels.borrow().is_empty() {
            let active_observers = self.sdo.get_active_observers_internal();
            if !active_observers.is_empty() {
                // Führe Clustering durch
                self.perform_clustering(&active_observers);
                // Entferne kleine Cluster
                self.remove_small_clusters();
                // Baue Tree für aktive Observer mit Labels
                self.build_tree_active();
            }
        }
    }

    /// Baut tree_active aus den aktiven Observers mit gültigen Labels
    fn build_tree_active(&self) {
        if self.tree_active.borrow().is_some() {
            return; // Bereits gebaut
        }

        let active_observers = self.sdo.get_active_observers_with_indices();
        let observer_labels = self.observer_labels.borrow();

        // Filtere Observer mit gültigen Labels und behalte Index
        let valid_observers: Vec<Observer> = active_observers
            .iter()
            .filter(|obs| obs.index < observer_labels.len() && observer_labels[obs.index] >= 0)
            .cloned()
            .collect();

        if valid_observers.is_empty() {
            return;
        }

        // Hole Metrik aus dem internen SDO-Objekt
        let metric = self.sdo.get_distance_metric_internal();
        let minkowski_p = self.sdo.get_minkowski_p_internal();
        let tree_type = self.sdo.get_tree_type_internal();

        // Baue Tree mit Observer-Objekten (direkt mit Index-Information)
        *self.tree_active.borrow_mut() = match (tree_type, metric) {
            (TreeType::VpTree, DistanceMetric::Euclidean) => {
                let points: Vec<ObserverEuclidean> =
                    valid_observers.into_iter().map(ObserverEuclidean).collect();
                Some(SpatialTreeObserver::VpEuclidean(VpTree::balanced(points)))
            }
            (TreeType::VpTree, DistanceMetric::Manhattan) => {
                let points: Vec<ObserverManhattan> =
                    valid_observers.into_iter().map(ObserverManhattan).collect();
                Some(SpatialTreeObserver::VpManhattan(VpTree::balanced(points)))
            }
            (TreeType::VpTree, DistanceMetric::Chebyshev) => {
                let points: Vec<ObserverChebyshev> =
                    valid_observers.into_iter().map(ObserverChebyshev).collect();
                Some(SpatialTreeObserver::VpChebyshev(VpTree::balanced(points)))
            }
            (TreeType::VpTree, DistanceMetric::Minkowski) => {
                let p = minkowski_p.unwrap_or(3.0);
                let points: Vec<ObserverMinkowski> = valid_observers
                    .into_iter()
                    .map(|obs| ObserverMinkowski { observer: obs, p })
                    .collect();
                Some(SpatialTreeObserver::VpMinkowski(VpTree::balanced(points)))
            }
            (TreeType::KdTree, DistanceMetric::Euclidean) => {
                let points: Vec<ObserverEuclidean> =
                    valid_observers.into_iter().map(ObserverEuclidean).collect();
                Some(SpatialTreeObserver::KdEuclidean(KdTree::from_iter(points)))
            }
            (TreeType::KdTree, DistanceMetric::Manhattan) => {
                let points: Vec<ObserverManhattan> =
                    valid_observers.into_iter().map(ObserverManhattan).collect();
                Some(SpatialTreeObserver::KdManhattan(KdTree::from_iter(points)))
            }
            (TreeType::KdTree, DistanceMetric::Chebyshev) => {
                let points: Vec<ObserverChebyshev> =
                    valid_observers.into_iter().map(ObserverChebyshev).collect();
                Some(SpatialTreeObserver::KdChebyshev(KdTree::from_iter(points)))
            }
            (TreeType::KdTree, DistanceMetric::Minkowski) => {
                let p = minkowski_p.unwrap_or(3.0);
                let points: Vec<ObserverMinkowski> = valid_observers
                    .into_iter()
                    .map(|obs| ObserverMinkowski { observer: obs, p })
                    .collect();
                Some(SpatialTreeObserver::KdMinkowski(KdTree::from_iter(points)))
            }
        };
    }

    fn perform_clustering(&self, active_observers: &[Vec<f64>]) {
        let n = active_observers.len();
        if n == 0 {
            *self.observer_labels.borrow_mut() = Vec::new();
            return;
        }

        // Hole tree_active_observers aus SDO (für Nearest Neighbor Search)
        let tree_opt = self.sdo.get_tree_active_observers();
        let distance_metric = self.sdo.get_distance_metric_internal();
        let minkowski_p = self.sdo.get_minkowski_p_internal();

        // Schritt 1: Berechne lokale Cutoff-Thresholds h_ω mit Tree-basierter Nearest Neighbor Search
        let mut local_thresholds = vec![0.0; n];

        if let Some(ref tree_ref) = tree_opt {
            if let Some(ref tree) = **tree_ref {
                // Verwende Tree für Nearest Neighbor Search
                for (idx, observer) in active_observers.iter().enumerate() {
                    let chi_actual = (self.chi + 1).min(n); // +1 weil wir den Observer selbst ausschließen
                    let distances = self.get_k_nearest_distances(observer, tree, chi_actual);
                    if distances.len() >= self.chi {
                        local_thresholds[idx] = distances[self.chi - 1];
                    } else if !distances.is_empty() {
                        local_thresholds[idx] = distances[distances.len() - 1];
                    } else {
                        local_thresholds[idx] = f64::INFINITY;
                    }
                }
            } else {
                // Fallback zu Brute-Force
                self.compute_local_thresholds_brute_force(active_observers, &mut local_thresholds);
            }
        } else {
            // Fallback zu Brute-Force
            self.compute_local_thresholds_brute_force(active_observers, &mut local_thresholds);
        }

        // Schritt 2: Berechne globalen Density-Threshold h
        let global_threshold: f64 =
            local_thresholds.iter().sum::<f64>() / local_thresholds.len() as f64;

        // Schritt 3: Berechne finale Thresholds mit Mixture-Modell: h'_ω = ζ·h_ω + (1-ζ)·h
        let final_thresholds: Vec<f64> = local_thresholds
            .iter()
            .map(|&h_omega| self.zeta * h_omega + (1.0 - self.zeta) * global_threshold)
            .collect();

        // Schritt 4 & 5: Finde Connected Components mit DFS und weise Labels zu
        // Zwei Observer sind verbunden wenn d(ν,ω) < h'_ω UND d(ν,ω) < h'_ν
        let mut observer_labels = vec![-1; n]; // -1 = kein Label/entfernt
        let mut visited = vec![false; n];
        let mut current_label = 0;

        // DFS für jeden unbesuchten Observer
        for start_idx in 0..n {
            if !visited[start_idx] {
                // Starte DFS von diesem Observer
                let mut stack = vec![start_idx];
                visited[start_idx] = true;
                observer_labels[start_idx] = current_label;

                while let Some(current_idx) = stack.pop() {
                    // Finde alle verbundenen Nachbarn
                    for neighbor_idx in 0..n {
                        if neighbor_idx != current_idx && !visited[neighbor_idx] {
                            // Berechne Distanz nur wenn nötig
                            let dist = compute_distance(
                                &active_observers[current_idx],
                                &active_observers[neighbor_idx],
                                distance_metric,
                                minkowski_p,
                            );
                            // Zwei Observer sind verbunden wenn d(ν,ω) < h'_ω UND d(ν,ω) < h'_ν
                            if dist < final_thresholds[current_idx]
                                && dist < final_thresholds[neighbor_idx]
                            {
                                visited[neighbor_idx] = true;
                                observer_labels[neighbor_idx] = current_label;
                                stack.push(neighbor_idx);
                            }
                        }
                    }
                }
                current_label += 1;
            }
        }
        *self.observer_labels.borrow_mut() = observer_labels;
    }

    /// Berechnet lokale Thresholds mit Brute-Force (Fallback)
    fn compute_local_thresholds_brute_force(
        &self,
        active_observers: &[Vec<f64>],
        local_thresholds: &mut [f64],
    ) {
        let distance_metric = self.sdo.get_distance_metric_internal();
        let minkowski_p = self.sdo.get_minkowski_p_internal();

        for (idx, observer) in active_observers.iter().enumerate() {
            let mut distances: Vec<f64> = active_observers
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != idx)
                .map(|(_, other)| compute_distance(observer, other, distance_metric, minkowski_p))
                .collect();

            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let chi_actual = self.chi.min(distances.len());
            if chi_actual > 0 {
                local_thresholds[idx] = distances[chi_actual - 1];
            } else {
                local_thresholds[idx] = f64::INFINITY;
            }
        }
    }

    /// Holt k-nearest distances vom Tree für einen Observer
    fn get_k_nearest_distances(
        &self,
        observer: &[f64],
        tree: &SpatialTreeObserver,
        k: usize,
    ) -> Vec<f64> {
        let minkowski_p = self.sdo.get_minkowski_p_internal();

        // Erstelle Query-Punkt als Observer
        let query_observer = Observer {
            data: observer.to_vec(),
            observations: 0.0,
            age: 0.0,
            index: 0,
        };

        let neighbors = match tree {
            SpatialTreeObserver::VpEuclidean(ref t) => {
                let query = ObserverEuclidean(query_observer);
                t.k_nearest(&query, k)
                    .into_iter()
                    .map(|neighbor| neighbor.distance.value())
                    .collect::<Vec<_>>()
            }
            SpatialTreeObserver::VpManhattan(ref t) => {
                let query = ObserverManhattan(query_observer.clone());
                t.k_nearest(&query, k)
                    .into_iter()
                    .map(|neighbor| neighbor.distance)
                    .collect::<Vec<_>>()
            }
            SpatialTreeObserver::VpChebyshev(ref t) => {
                let query = ObserverChebyshev(query_observer.clone());
                t.k_nearest(&query, k)
                    .into_iter()
                    .map(|neighbor| neighbor.distance)
                    .collect::<Vec<_>>()
            }
            SpatialTreeObserver::VpMinkowski(ref t) => {
                let p = minkowski_p.unwrap_or(3.0);
                let query = ObserverMinkowski {
                    observer: query_observer.clone(),
                    p,
                };
                t.k_nearest(&query, k)
                    .into_iter()
                    .map(|neighbor| neighbor.distance)
                    .collect::<Vec<_>>()
            }
            SpatialTreeObserver::KdEuclidean(ref t) => {
                let query = ObserverEuclidean(query_observer);
                t.k_nearest(&query, k)
                    .into_iter()
                    .map(|neighbor| neighbor.distance.value())
                    .collect::<Vec<_>>()
            }
            SpatialTreeObserver::KdManhattan(ref t) => {
                let query = ObserverManhattan(query_observer.clone());
                t.k_nearest(&query, k)
                    .into_iter()
                    .map(|neighbor| neighbor.distance)
                    .collect::<Vec<_>>()
            }
            SpatialTreeObserver::KdChebyshev(ref t) => {
                let query = ObserverChebyshev(query_observer.clone());
                t.k_nearest(&query, k)
                    .into_iter()
                    .map(|neighbor| neighbor.distance)
                    .collect::<Vec<_>>()
            }
            SpatialTreeObserver::KdMinkowski(ref t) => {
                let p = minkowski_p.unwrap_or(3.0);
                let query = ObserverMinkowski {
                    observer: query_observer.clone(),
                    p,
                };
                t.k_nearest(&query, k)
                    .into_iter()
                    .map(|neighbor| neighbor.distance)
                    .collect::<Vec<_>>()
            }
        };
        neighbors
    }

    fn remove_small_clusters(&self) {
        let mut observer_labels = self.observer_labels.borrow_mut();
        // Zähle Größe jedes Clusters
        let mut cluster_sizes: HashMap<i32, usize> = HashMap::new();
        for &label in observer_labels.iter() {
            if label >= 0 {
                *cluster_sizes.entry(label).or_insert(0) += 1;
            }
        }

        // Entferne Observer aus zu kleinen Clustern
        for label in observer_labels.iter_mut() {
            if *label >= 0 {
                if let Some(&size) = cluster_sizes.get(label) {
                    if size < self.min_cluster_size {
                        *label = -1; // Markiere als entfernt
                    }
                }
            }
        }
    }

    /// Verwendet Tree für k-nearest neighbors und gibt das häufigste Label zurück
    fn predict_with_tree(&self, point: &[f64], tree: &SpatialTreeObserver) -> i32 {
        let observer_labels = self.observer_labels.borrow();
        let x = self.sdo.get_x_internal();
        let minkowski_p = self.sdo.get_minkowski_p_internal();

        // Erstelle Query-Punkt (als Observer-Wrapper)
        let point_observer = Observer {
            data: point.to_vec(),
            observations: 0.0,
            age: 0.0,
            index: 0,
        };

        let mut label_counts: HashMap<i32, usize> = HashMap::new();
        let k = x.min(observer_labels.len());

        // Hole k-nearest neighbors vom Tree - direkt als Observer-Objekte!
        match tree {
            SpatialTreeObserver::VpEuclidean(ref t) => {
                let query = ObserverEuclidean(point_observer);
                for neighbor in t.k_nearest(&query, k) {
                    let idx = neighbor.item.0.index; // Direkter Zugriff auf Index!
                    if idx < observer_labels.len() {
                        let label = observer_labels[idx];
                        if label >= 0 {
                            *label_counts.entry(label).or_insert(0) += 1;
                        }
                    }
                }
            }
            SpatialTreeObserver::KdEuclidean(ref t) => {
                let query = ObserverEuclidean(point_observer);
                for neighbor in t.k_nearest(&query, k) {
                    let idx = neighbor.item.0.index;
                    if idx < observer_labels.len() {
                        let label = observer_labels[idx];
                        if label >= 0 {
                            *label_counts.entry(label).or_insert(0) += 1;
                        }
                    }
                }
            }
            SpatialTreeObserver::VpManhattan(ref t) => {
                let query = ObserverManhattan(point_observer);
                for neighbor in t.k_nearest(&query, k) {
                    let idx = neighbor.item.0.index;
                    if idx < observer_labels.len() {
                        let label = observer_labels[idx];
                        if label >= 0 {
                            *label_counts.entry(label).or_insert(0) += 1;
                        }
                    }
                }
            }
            SpatialTreeObserver::KdManhattan(ref t) => {
                let query = ObserverManhattan(point_observer);
                for neighbor in t.k_nearest(&query, k) {
                    let idx = neighbor.item.0.index;
                    if idx < observer_labels.len() {
                        let label = observer_labels[idx];
                        if label >= 0 {
                            *label_counts.entry(label).or_insert(0) += 1;
                        }
                    }
                }
            }
            SpatialTreeObserver::VpChebyshev(ref t) => {
                let query = ObserverChebyshev(point_observer);
                for neighbor in t.k_nearest(&query, k) {
                    let idx = neighbor.item.0.index;
                    if idx < observer_labels.len() {
                        let label = observer_labels[idx];
                        if label >= 0 {
                            *label_counts.entry(label).or_insert(0) += 1;
                        }
                    }
                }
            }
            SpatialTreeObserver::KdChebyshev(ref t) => {
                let query = ObserverChebyshev(point_observer);
                for neighbor in t.k_nearest(&query, k) {
                    let idx = neighbor.item.0.index;
                    if idx < observer_labels.len() {
                        let label = observer_labels[idx];
                        if label >= 0 {
                            *label_counts.entry(label).or_insert(0) += 1;
                        }
                    }
                }
            }
            SpatialTreeObserver::VpMinkowski(ref t) => {
                let p = minkowski_p.unwrap_or(3.0);
                let query = ObserverMinkowski {
                    observer: point_observer,
                    p,
                };
                for neighbor in t.k_nearest(&query, k) {
                    let idx = neighbor.item.observer.index;
                    if idx < observer_labels.len() {
                        let label = observer_labels[idx];
                        if label >= 0 {
                            *label_counts.entry(label).or_insert(0) += 1;
                        }
                    }
                }
            }
            SpatialTreeObserver::KdMinkowski(ref t) => {
                let p = minkowski_p.unwrap_or(3.0);
                let query = ObserverMinkowski {
                    observer: point_observer,
                    p,
                };
                for neighbor in t.k_nearest(&query, k) {
                    let idx = neighbor.item.observer.index;
                    if idx < observer_labels.len() {
                        let label = observer_labels[idx];
                        if label >= 0 {
                            *label_counts.entry(label).or_insert(0) += 1;
                        }
                    }
                }
            }
        }

        // Gib das häufigste Label zurück
        if let Some((&most_common_label, _)) = label_counts.iter().max_by_key(|(_, &count)| count) {
            most_common_label
        } else {
            -1
        }
    }

    fn predict_brute_force(&self, point: &[f64], active_observers: &[Vec<f64>]) -> i32 {
        let distance_metric = self.sdo.get_distance_metric_internal();
        let minkowski_p = self.sdo.get_minkowski_p_internal();
        let x = self.sdo.get_x_internal();
        let observer_labels = self.observer_labels.borrow();

        let mut distances: Vec<(usize, f64)> = active_observers
            .iter()
            .enumerate()
            .map(|(idx, observer)| {
                (
                    idx,
                    compute_distance(point, observer, distance_metric, minkowski_p),
                )
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let k = x.min(distances.len());
        let mut label_counts: HashMap<i32, usize> = HashMap::new();

        for (idx, _) in distances.iter().take(k) {
            if *idx < observer_labels.len() {
                let label = observer_labels[*idx];
                if label >= 0 {
                    *label_counts.entry(label).or_insert(0) += 1;
                }
            }
        }

        if let Some((&most_common_label, _)) = label_counts.iter().max_by_key(|(_, &count)| count) {
            most_common_label
        } else {
            -1
        }
    }
}

impl Default for SDOclust {
    fn default() -> Self {
        Self::new()
    }
}
