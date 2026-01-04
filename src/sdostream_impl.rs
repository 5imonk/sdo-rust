use acap::distance::Distance;
use acap::kd::KdTree;
use acap::knn::NearestNeighbors;
use acap::vp::VpTree;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use std::cell::RefCell;
use std::collections::HashSet;
use std::f64;

use crate::sdo_impl::{DistanceMetric, SpatialTreeObserver, TreeType};
use crate::utils::{
    compute_distance, Observer, ObserverChebyshev, ObserverEuclidean, ObserverManhattan,
    ObserverMinkowski, ObserverSet,
};

/// Parameter-Struktur für SDOstream
#[pyclass]
#[derive(Clone)]
pub struct SDOstreamParams {
    #[pyo3(get, set)]
    pub k: usize, // Anzahl der Observer (Modellgröße)
    #[pyo3(get, set)]
    pub x: usize, // Anzahl der nächsten Observer für Prediction
    #[pyo3(get, set)]
    pub t: f64, // T-Parameter für fading: f = exp(-T^-1)
    #[pyo3(get, set)]
    pub distance: String, // "euclidean", "manhattan", "chebyshev", "minkowski"
    #[pyo3(get, set)]
    pub minkowski_p: Option<f64>, // Für Minkowski-Distanz
    #[pyo3(get, set)]
    pub tree_type: String, // "vptree" (default) oder "kdtree"
}

#[pymethods]
impl SDOstreamParams {
    #[new]
    #[pyo3(signature = (k, x, t, distance = "euclidean".to_string(), minkowski_p = None, tree_type = "vptree".to_string()))]
    pub fn new(
        k: usize,
        x: usize,
        t: f64,
        distance: String,
        minkowski_p: Option<f64>,
        tree_type: String,
    ) -> Self {
        Self {
            k,
            x,
            t,
            distance,
            minkowski_p,
            tree_type,
        }
    }
}

impl SDOstreamParams {
    fn get_metric(&self) -> DistanceMetric {
        match self.distance.to_lowercase().as_str() {
            "manhattan" => DistanceMetric::Manhattan,
            "chebyshev" => DistanceMetric::Chebyshev,
            "minkowski" => DistanceMetric::Minkowski,
            _ => DistanceMetric::Euclidean,
        }
    }

    fn get_tree_type(&self) -> TreeType {
        match self.tree_type.to_lowercase().as_str() {
            "kdtree" => TreeType::KdTree,
            _ => TreeType::VpTree, // Default: VpTree
        }
    }

    /// Berechnet den Fading-Parameter f = exp(-T^-1)
    fn get_fading(&self) -> f64 {
        (-1.0 / self.t).exp()
    }

    /// Berechnet die Sampling-Rate T_k = -k · ln(f)
    fn get_sampling_rate(&self) -> f64 {
        let f = self.get_fading();
        -(self.k as f64) * f.ln()
    }
}

/// Observer mit zusätzlichen Feldern für SDOstream
#[derive(Clone, Debug)]
struct StreamObserver {
    observer: Observer,
    age: f64, // Hω - Alter des Observers (maximal möglicher Wert für Pω)
}

/// SDOstream Algorithm - Streaming-Version von SDO
#[pyclass]
#[allow(clippy::upper_case_acronyms)]
pub struct SDOstream {
    observers: ObserverSet,
    stream_observers: Vec<StreamObserver>, // Zusätzliche Metadaten für Streaming
    tree_observers: RefCell<Option<SpatialTreeObserver>>,
    distance_metric: DistanceMetric,
    minkowski_p: Option<f64>,
    tree_type: TreeType,
    fading: f64, // f = exp(-T^-1)
    sampling_rate: f64, // T_k = -k · ln(f)
    #[pyo3(get, set)]
    x: usize,
    data_points_processed: usize, // Zähler für Sampling
}

#[pymethods]
impl SDOstream {
    #[new]
    pub fn new() -> Self {
        Self {
            observers: ObserverSet::new(),
            stream_observers: Vec::new(),
            tree_observers: RefCell::new(None),
            distance_metric: DistanceMetric::Euclidean,
            minkowski_p: None,
            tree_type: TreeType::VpTree,
            fading: 0.99,
            sampling_rate: 0.01,
            x: 10,
            data_points_processed: 0,
        }
    }

    /// Initialisiert das Modell mit initialen Daten (optional)
    pub fn initialize(
        &mut self,
        data: PyReadonlyArray2<f64>,
        params: &SDOstreamParams,
    ) -> PyResult<()> {
        let data_slice = data.as_array();
        let rows = data_slice.nrows();
        let cols = data_slice.ncols();

        if rows == 0 || params.k == 0 {
            return Ok(());
        }

        // Konvertiere NumPy-Array zu Vec<Vec<f64>>
        let data_vec: Vec<Vec<f64>> = (0..rows)
            .map(|i| (0..cols).map(|j| data_slice[[i, j]]).collect())
            .collect();

        self.x = params.x;
        self.distance_metric = params.get_metric();
        self.minkowski_p = params.minkowski_p;
        self.tree_type = params.get_tree_type();
        self.fading = params.get_fading();
        self.sampling_rate = params.get_sampling_rate();

        // Initialisiere mit zufälligen Samples
        let mut rng = thread_rng();
        let num_samples = params.k.min(data_vec.len());
        let indices: Vec<usize> = (0..data_vec.len()).collect();
        let sampled_indices: Vec<usize> = indices
            .choose_multiple(&mut rng, num_samples)
            .cloned()
            .collect();

        self.observers = ObserverSet::new();
        self.stream_observers = Vec::new();

        for (idx, &data_idx) in sampled_indices.iter().enumerate() {
            let observer = Observer {
                data: data_vec[data_idx].clone(),
                observations: 0.0, // Start mit 0
                index: idx,
            };
            self.observers.insert(observer.clone());
            self.stream_observers.push(StreamObserver {
                observer: observer.clone(),
                age: 1.0, // Start-Alter
            });
        }

        // Baue initialen Tree
        self.rebuild_tree();

        Ok(())
    }

    /// Verarbeitet einen einzelnen Datenpunkt aus dem Stream
    pub fn learn(&mut self, point: PyReadonlyArray2<f64>, params: &SDOstreamParams) -> PyResult<()> {
        let point_slice = point.as_array();
        if point_slice.nrows() != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Punkt muss ein 1D-Array oder 2D-Array mit einer Zeile sein",
            ));
        }

        let point_vec: Vec<f64> = (0..point_slice.ncols())
            .map(|j| point_slice[[0, j]])
            .collect();

        // Aktualisiere Parameter falls nötig
        self.x = params.x;
        self.distance_metric = params.get_metric();
        self.minkowski_p = params.minkowski_p;
        self.tree_type = params.get_tree_type();
        self.fading = params.get_fading();
        self.sampling_rate = params.get_sampling_rate();

        // Schritt 1: Finde x-nächste Observer
        let nearest_observers = self.find_x_nearest_observers(&point_vec, self.x);

        // Schritt 2: Update Pω für alle Observer mit Exponential Moving Average
        self.update_observations(&nearest_observers);

        // Schritt 3: Erhöhe Alter Hω für alle Observer
        self.increment_ages();

        // Schritt 4: Sampling - ersetze Observer basierend auf normalisierter Qualität
        self.data_points_processed += 1;
        if self.should_sample() {
            self.replace_observer(&point_vec);
        }

        // Schritt 5: Rebuild Tree (lazy, nur wenn nötig)
        *self.tree_observers.borrow_mut() = None;

        Ok(())
    }

    /// Berechnet den Outlier-Score für einen Datenpunkt (wie SDO)
    pub fn predict(&self, point: PyReadonlyArray2<f64>) -> PyResult<f64> {
        if self.observers.is_empty() {
            return Ok(f64::INFINITY);
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

        // Berechne aktive Observer (obere (1-rho) * k)
        // In SDOstream verwenden wir rho = 0 (keine Entfernung), also alle Observer
        let active_observers = self.get_active_observers();
        if active_observers.is_empty() {
            return Ok(f64::INFINITY);
        }

        let k = self.x.min(active_observers.len());

        // Verwende Tree für k-nearest neighbors
        self.ensure_tree_active_observers();

        let distances = {
            let tree_opt = self.tree_observers.borrow();
            if let Some(ref tree) = *tree_opt {
                self.predict_with_tree(&point_vec, tree, k)
            } else {
                // Fallback: Brute-Force
                let mut distances: Vec<f64> = active_observers
                    .iter()
                    .map(|obs| self.compute_distance(&point_vec, &obs.data))
                    .collect();
                distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
                distances.into_iter().take(k).collect()
            }
        };

        if distances.is_empty() {
            return Ok(f64::INFINITY);
        }

        // Berechne Median
        let mid = distances.len() / 2;
        let median = if distances.len() % 2 == 0 && mid > 0 {
            (distances[mid - 1] + distances[mid]) / 2.0
        } else {
            distances[mid]
        };

        Ok(median)
    }

    /// Gibt die Anzahl der Observer zurück
    pub fn n_observers(&self) -> usize {
        self.observers.len()
    }
}

impl SDOstream {
    /// Findet die x-nächsten Observer zu einem Punkt
    fn find_x_nearest_observers(&self, point: &[f64], x: usize) -> Vec<usize> {
        let all_observers = self.observers.all();
        let mut distances: Vec<(usize, f64)> = all_observers
            .iter()
            .enumerate()
            .map(|(idx, obs)| {
                (
                    idx,
                    compute_distance(point, &obs.data, self.distance_metric, self.minkowski_p),
                )
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.into_iter().take(x).map(|(idx, _)| idx).collect()
    }

    /// Aktualisiert Pω für alle Observer mit Exponential Moving Average
    /// Pω ← f · Pω + 1 wenn ω unter den x-nächsten, sonst Pω ← f · Pω
    fn update_observations(&mut self, nearest_indices: &[usize]) {
        let all_observers = self.observers.all();
        let nearest_set: HashSet<usize> = nearest_indices.iter().cloned().collect();

        for (idx, observer) in all_observers.iter().enumerate() {
            let new_observations = if nearest_set.contains(&idx) {
                // Pω ← f · Pω + 1
                self.fading * observer.observations + 1.0
            } else {
                // Pω ← f · Pω
                self.fading * observer.observations
            };

            // Aktualisiere Observer im Set
            self.observers.update_observations(observer.index, new_observations);

            // Aktualisiere auch in stream_observers
            if idx < self.stream_observers.len() {
                self.stream_observers[idx].observer.observations = new_observations;
            }
        }
    }

    /// Erhöht das Alter Hω für alle Observer
    /// Hω kann inkrementell aktualisiert werden oder mit konstanter Inter-Arrival-Zeit approximiert werden
    fn increment_ages(&mut self) {
        for stream_obs in self.stream_observers.iter_mut() {
            // Inkrementelle Aktualisierung: Hω ← Hω + 1
            // Alternativ könnte man Hω = 1 / (1 - f) approximieren für konstante Inter-Arrival-Zeit
            stream_obs.age += 1.0;
        }
    }

    /// Prüft, ob ein Observer ersetzt werden sollte basierend auf Sampling-Rate
    fn should_sample(&self) -> bool {
        // Sampling-Rate T_k = -k · ln(f)
        // Ersetze im Durchschnitt alle T_k Datenpunkte
        // Verwende stochastisches Sampling: Sample mit Wahrscheinlichkeit 1 / T_k
        if self.sampling_rate <= 0.0 {
            return false;
        }
        let probability = 1.0 / self.sampling_rate;
        thread_rng().gen::<f64>() < probability
    }

    /// Ersetzt einen Observer basierend auf normalisierter Qualitätsmetrik P̃ω = Pω / Hω
    fn replace_observer(&mut self, new_point: &[f64]) {
        if self.stream_observers.is_empty() {
            return;
        }

        // Finde Observer mit niedrigster normalisierter Qualität
        let mut min_normalized_quality = f64::INFINITY;
        let mut replace_idx = 0;

        for (idx, stream_obs) in self.stream_observers.iter().enumerate() {
            let normalized_quality = if stream_obs.age > 0.0 {
                stream_obs.observer.observations / stream_obs.age
            } else {
                f64::INFINITY
            };

            if normalized_quality < min_normalized_quality {
                min_normalized_quality = normalized_quality;
                replace_idx = idx;
            }
        }

        // Ersetze den Observer
        let old_observer = &self.stream_observers[replace_idx].observer;
        let new_observer = Observer {
            data: new_point.to_vec(),
            observations: 0.0, // Neuer Observer startet mit Pω = 0
            index: old_observer.index,
        };

        // Aktualisiere ObserverSet
        self.observers.replace(old_observer.index, new_observer.clone());

        // Aktualisiere stream_observers
        self.stream_observers[replace_idx] = StreamObserver {
            observer: new_observer,
            age: 1.0, // Neuer Observer startet mit Hω = 1
        };
    }

    /// Gibt aktive Observer zurück (in SDOstream: alle Observer, da keine Entfernung)
    fn get_active_observers(&self) -> Vec<Observer> {
        self.observers.all()
    }

    /// Baut Tree aus aktiven Observers neu auf
    fn rebuild_tree(&self) {
        let active_observers = self.get_active_observers();
        if active_observers.is_empty() {
            *self.tree_observers.borrow_mut() = None;
            return;
        }

        // Erstelle Observer mit Indizes für Tree
        let observers_with_indices: Vec<Observer> = active_observers
            .iter()
            .enumerate()
            .map(|(idx, obs)| Observer {
                data: obs.data.clone(),
                observations: obs.observations,
                index: idx,
            })
            .collect();

        *self.tree_observers.borrow_mut() = self.build_tree_from_observers(
            &observers_with_indices,
            self.distance_metric,
            self.minkowski_p,
            self.tree_type,
        );
    }

    /// Erstellt tree_active_observers lazy, wenn noch nicht vorhanden
    fn ensure_tree_active_observers(&self) {
        if self.tree_observers.borrow().is_none() {
            self.rebuild_tree();
        }
    }

    /// Baut einen Tree aus Observer-Objekten
    fn build_tree_from_observers(
        &self,
        observers: &[Observer],
        metric: DistanceMetric,
        minkowski_p: Option<f64>,
        tree_type: TreeType,
    ) -> Option<SpatialTreeObserver> {
        if observers.is_empty() {
            return None;
        }

        match (tree_type, metric) {
            (TreeType::VpTree, DistanceMetric::Euclidean) => {
                let points: Vec<ObserverEuclidean> = observers
                    .iter()
                    .map(|obs| ObserverEuclidean(obs.clone()))
                    .collect();
                Some(SpatialTreeObserver::VpEuclidean(VpTree::balanced(points)))
            }
            (TreeType::VpTree, DistanceMetric::Manhattan) => {
                let points: Vec<ObserverManhattan> = observers
                    .iter()
                    .map(|obs| ObserverManhattan(obs.clone()))
                    .collect();
                Some(SpatialTreeObserver::VpManhattan(VpTree::balanced(points)))
            }
            (TreeType::VpTree, DistanceMetric::Chebyshev) => {
                let points: Vec<ObserverChebyshev> = observers
                    .iter()
                    .map(|obs| ObserverChebyshev(obs.clone()))
                    .collect();
                Some(SpatialTreeObserver::VpChebyshev(VpTree::balanced(points)))
            }
            (TreeType::VpTree, DistanceMetric::Minkowski) => {
                let p = minkowski_p.unwrap_or(3.0);
                let points: Vec<ObserverMinkowski> = observers
                    .iter()
                    .map(|obs| ObserverMinkowski {
                        observer: obs.clone(),
                        p,
                    })
                    .collect();
                Some(SpatialTreeObserver::VpMinkowski(VpTree::balanced(points)))
            }
            (TreeType::KdTree, DistanceMetric::Euclidean) => {
                let points: Vec<ObserverEuclidean> = observers
                    .iter()
                    .map(|obs| ObserverEuclidean(obs.clone()))
                    .collect();
                Some(SpatialTreeObserver::KdEuclidean(KdTree::from_iter(points)))
            }
            (TreeType::KdTree, DistanceMetric::Manhattan) => {
                let points: Vec<ObserverManhattan> = observers
                    .iter()
                    .map(|obs| ObserverManhattan(obs.clone()))
                    .collect();
                Some(SpatialTreeObserver::KdManhattan(KdTree::from_iter(points)))
            }
            (TreeType::KdTree, DistanceMetric::Chebyshev) => {
                let points: Vec<ObserverChebyshev> = observers
                    .iter()
                    .map(|obs| ObserverChebyshev(obs.clone()))
                    .collect();
                Some(SpatialTreeObserver::KdChebyshev(KdTree::from_iter(points)))
            }
            (TreeType::KdTree, DistanceMetric::Minkowski) => {
                let p = minkowski_p.unwrap_or(3.0);
                let points: Vec<ObserverMinkowski> = observers
                    .iter()
                    .map(|obs| ObserverMinkowski {
                        observer: obs.clone(),
                        p,
                    })
                    .collect();
                Some(SpatialTreeObserver::KdMinkowski(KdTree::from_iter(points)))
            }
        }
    }

    /// Verwendet Tree für k-nearest neighbors und gibt Distanzen zurück
    fn predict_with_tree(
        &self,
        point: &[f64],
        tree: &SpatialTreeObserver,
        k: usize,
    ) -> Vec<f64> {
        let point_observer = Observer {
            data: point.to_vec(),
            observations: 0.0,
            index: 0,
        };
        match tree {
            SpatialTreeObserver::VpEuclidean(ref t) => {
                let query = ObserverEuclidean(point_observer);
                t.k_nearest(&query, k)
                    .into_iter()
                    .map(|neighbor| neighbor.distance.value())
                    .collect()
            }
            SpatialTreeObserver::VpManhattan(ref t) => {
                let query = ObserverManhattan(point_observer.clone());
                t.k_nearest(&query, k)
                    .into_iter()
                    .map(|neighbor| neighbor.distance)
                    .collect()
            }
            SpatialTreeObserver::VpChebyshev(ref t) => {
                let query = ObserverChebyshev(point_observer.clone());
                t.k_nearest(&query, k)
                    .into_iter()
                    .map(|neighbor| neighbor.distance)
                    .collect()
            }
            SpatialTreeObserver::VpMinkowski(ref t) => {
                let p = self.minkowski_p.unwrap_or(3.0);
                let query = ObserverMinkowski {
                    observer: point_observer.clone(),
                    p,
                };
                t.k_nearest(&query, k)
                    .into_iter()
                    .map(|neighbor| neighbor.distance)
                    .collect()
            }
            SpatialTreeObserver::KdEuclidean(ref t) => {
                let query = ObserverEuclidean(point_observer);
                t.k_nearest(&query, k)
                    .into_iter()
                    .map(|neighbor| neighbor.distance.value())
                    .collect()
            }
            SpatialTreeObserver::KdManhattan(ref t) => {
                let query = ObserverManhattan(point_observer.clone());
                t.k_nearest(&query, k)
                    .into_iter()
                    .map(|neighbor| neighbor.distance)
                    .collect()
            }
            SpatialTreeObserver::KdChebyshev(ref t) => {
                let query = ObserverChebyshev(point_observer.clone());
                t.k_nearest(&query, k)
                    .into_iter()
                    .map(|neighbor| neighbor.distance)
                    .collect()
            }
            SpatialTreeObserver::KdMinkowski(ref t) => {
                let p = self.minkowski_p.unwrap_or(3.0);
                let query = ObserverMinkowski {
                    observer: point_observer.clone(),
                    p,
                };
                t.k_nearest(&query, k)
                    .into_iter()
                    .map(|neighbor| neighbor.distance)
                    .collect()
            }
        }
    }

    fn compute_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        compute_distance(a, b, self.distance_metric, self.minkowski_p)
    }
}

impl Default for SDOstream {
    fn default() -> Self {
        Self::new()
    }
}

