use std::cell::RefCell;

use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::f64;

use crate::observer_set::{Observer, ObserverSet, SpatialTreeObserver};
use crate::utils::{compute_distance, DistanceMetric};

// SpatialTreeObserver moved to observer_set.rs (now uses HNSW)

/// Parameter-Struktur für SDO
#[pyclass]
#[derive(Clone)]
pub struct SDOParams {
    #[pyo3(get, set)]
    pub k: usize,
    #[pyo3(get, set)]
    pub x: usize,
    #[pyo3(get, set)]
    pub rho: f64,
    #[pyo3(get, set)]
    pub distance: String, // "euclidean", "manhattan", "chebyshev", "minkowski"
    #[pyo3(get, set)]
    pub minkowski_p: Option<f64>, // Für Minkowski-Distanz
}

#[pymethods]
impl SDOParams {
    #[new]
    #[pyo3(signature = (k, x, rho, distance = "euclidean".to_string(), minkowski_p = None))]
    pub fn new(k: usize, x: usize, rho: f64, distance: String, minkowski_p: Option<f64>) -> Self {
        Self {
            k,
            x,
            rho,
            distance,
            minkowski_p,
        }
    }
}

impl SDOParams {
    fn get_metric(&self) -> DistanceMetric {
        match self.distance.to_lowercase().as_str() {
            "manhattan" => DistanceMetric::Manhattan,
            "chebyshev" => DistanceMetric::Chebyshev,
            "minkowski" => DistanceMetric::Minkowski,
            _ => DistanceMetric::Euclidean,
        }
    }
}

/// Sparse Data Observers (SDO) Algorithm
#[pyclass]
#[allow(clippy::upper_case_acronyms)]
pub struct SDO {
    // Observer-Set, sortiert nach observations (verwendet HNSW intern)
    pub(crate) observers: ObserverSet,
    #[allow(dead_code)] // Wird von SDOstream verwendet
    pub(crate) tree_active_observers: RefCell<Option<SpatialTreeObserver>>,
    distance_metric: DistanceMetric,
    minkowski_p: Option<f64>,
    rho: f64,
    #[pyo3(get, set)]
    pub(crate) x: usize,
}

#[pymethods]
impl SDO {
    #[new]
    pub fn new() -> Self {
        Self {
            observers: ObserverSet::new(),
            tree_active_observers: RefCell::new(None),
            distance_metric: DistanceMetric::Euclidean,
            minkowski_p: None,
            rho: 0.1,
            x: 10,
        }
    }

    /// Lernt das Modell aus den Daten
    pub fn learn(&mut self, data: PyReadonlyArray2<f64>, params: &SDOParams) -> PyResult<()> {
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
        self.rho = params.rho;
        // Set tree parameters in ObserverSet
        self.observers
            .set_tree_params(self.distance_metric, self.minkowski_p);
        // Reset tree_active_observers, da sich die Parameter geändert haben
        *self.tree_active_observers.borrow_mut() = None;

        // Schritt 1: Sample
        let mut rng = thread_rng();
        let observers_data: Vec<Vec<f64>> = data_vec
            .choose_multiple(&mut rng, params.k.min(data_vec.len()))
            .cloned()
            .collect();

        // Schritt 2: Erstelle ObserverSet mit allen Observers (ohne observations)
        self.observers = ObserverSet::new();
        self.observers
            .set_tree_params(self.distance_metric, self.minkowski_p);
        for (idx, observer_data) in observers_data.iter().enumerate() {
            let observer = Observer {
                data: observer_data.clone(),
                observations: 0.0,
                time: 0.0,
                age: 1.0,
                index: idx,
            };
            self.observers.insert(observer);
        }

        // Schritt 3: Berechne observations für jeden Observer mit Nearest Neighbor Search
        // Für jeden Datenpunkt: Finde die x nächsten Observer und erhöhe deren observations um 1
        self.observers.ensure_spatial_tree();

        // Für jeden Datenpunkt: Finde x nächste Observer und erhöhe deren observations
        for data_point in &data_vec {
            // Finde die Indizes der x nächsten Observer zu diesem Datenpunkt
            let nearest_indices = self
                .observers
                .search_k_nearest_indices(data_point, params.x);

            // Erhöhe observations für jeden dieser Observer um 1
            for idx in nearest_indices {
                if let Some(observer) = self.observers.get(idx) {
                    let current_obs = observer.observations;
                    self.observers.update_observations(idx, current_obs + 1.0);
                }
            }
        }

        Ok(())
    }

    /// Berechnet den Outlier-Score für einen Datenpunkt
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

        // Lazy: Berechne aktive Observer nur wenn nötig
        let num_active = ((self.observers.len() as f64) * (1.0 - self.rho)).ceil() as usize;
        let num_active = num_active.max(1).min(self.observers.len());
        let active_observers = self.observers.get_active(num_active);

        if active_observers.is_empty() {
            return Ok(f64::INFINITY);
        }

        let k = self.x.min(active_observers.len());

        // Versuche kiddo KD-Tree-basierte Nearest Neighbor Search zu verwenden
        // Erstelle tree_active_observers lazy, wenn noch nicht vorhanden
        self.ensure_tree_active_observers();

        // Verwende ObserverSet's search_k_nearest direkt, um Borrow-Konflikte zu vermeiden
        let distances = self.observers.search_k_nearest(&point_vec, k);

        if distances.is_empty() {
            return Ok(f64::INFINITY);
        }

        let mid = distances.len() / 2;
        let median = if distances.len().is_multiple_of(2) && mid > 0 {
            (distances[mid - 1] + distances[mid]) / 2.0
        } else {
            distances[mid]
        };

        Ok(median)
    }

    /// Konvertiert active_observers zu NumPy-Array für Python
    pub fn get_active_observers(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        let num_active = ((self.observers.len() as f64) * (1.0 - self.rho)).ceil() as usize;
        let num_active = num_active.max(1).min(self.observers.len());
        let active_observers = self.observers.get_active(num_active);

        if active_observers.is_empty() {
            let array = PyArray2::zeros_bound(py, (0, 0), false);
            return Ok(array.unbind());
        }

        let rows = active_observers.len();
        let cols = active_observers[0].data.len();
        let array = PyArray2::zeros_bound(py, (rows, cols), false);

        unsafe {
            let mut array_mut = array.as_array_mut();
            for (i, observer) in active_observers.iter().enumerate() {
                for (j, &value) in observer.data.iter().enumerate() {
                    array_mut[[i, j]] = value;
                }
            }
        }

        Ok(array.unbind())
    }
}

// Private Hilfsmethoden (nicht für Python)
impl SDO {
    /// Zählt Punkte in der Nachbarschaft mit R*-Tree-basierter Nearest Neighbor Search
    fn count_points_in_neighborhood_with_tree(
        &self,
        observer: &[f64],
        _data: &[Vec<f64>],
        x: usize,
    ) -> usize {
        // Verwende ObserverSet's R*-Tree für k-NN-Suche
        self.observers.count_points_in_neighborhood(observer, x)
    }
}

impl SDO {
    /// Erstellt tree_active_observers lazy, wenn noch nicht vorhanden
    pub(crate) fn ensure_tree_active_observers(&self) {
        // Der Tree wird jetzt in ObserverSet verwaltet (HNSW)
        self.observers.ensure_spatial_tree();
    }

    /// Interne Methode, um active_observers zu erhalten (für SDOclust)
    pub(crate) fn get_active_observers_internal(&self) -> Vec<Vec<f64>> {
        let num_active = ((self.observers.len() as f64) * (1.0 - self.rho)).ceil() as usize;
        let num_active = num_active.max(1).min(self.observers.len());
        self.observers
            .get_active(num_active)
            .iter()
            .map(|obs| obs.data.clone())
            .collect()
    }

    /// Interne Methode, um active_observers mit Indizes zu erhalten (für SDOclust)
    /// Gibt Observer-Objekte zurück, wobei index die Position in der active_observers Liste ist
    pub(crate) fn get_active_observers_with_indices(&self) -> Vec<Observer> {
        let num_active = ((self.observers.len() as f64) * (1.0 - self.rho)).ceil() as usize;
        let num_active = num_active.max(1).min(self.observers.len());
        self.observers
            .get_active(num_active)
            .iter()
            .enumerate()
            .map(|(idx, obs)| Observer {
                data: obs.data.clone(),
                observations: obs.observations,
                time: obs.time,
                age: obs.age,
                index: idx, // Index in active_observers Liste
            })
            .collect()
    }

    /// Erstellt und gibt tree_active_observers zurück (für SDOclust)
    /// Diese Funktion stellt sicher, dass der Tree erstellt ist und gibt eine Referenz zurück
    pub(crate) fn get_tree_active_observers(
        &self,
    ) -> Option<std::cell::Ref<'_, Option<SpatialTreeObserver>>> {
        self.ensure_tree_active_observers();
        // Tree wird jetzt direkt aus ObserverSet geholt
        self.observers.get_spatial_tree()
    }

    /// Interne Methode, um x zu erhalten (für SDOclust)
    pub(crate) fn get_x_internal(&self) -> usize {
        self.x
    }

    /// Interne Methode, um distance_metric zu erhalten (für SDOclust)
    pub(crate) fn get_distance_metric_internal(&self) -> DistanceMetric {
        self.distance_metric
    }

    /// Interne Methode, um minkowski_p zu erhalten (für SDOclust)
    pub(crate) fn get_minkowski_p_internal(&self) -> Option<f64> {
        self.minkowski_p
    }

    pub(crate) fn compute_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        compute_distance(a, b, self.distance_metric, self.minkowski_p)
    }

    /// Interne Methode, um Observer-observations zu aktualisieren (für SDOstream)
    #[allow(dead_code)]
    pub(crate) fn update_observer_observations(
        &mut self,
        index: usize,
        new_observations: f64,
    ) -> bool {
        self.observers.update_observations(index, new_observations)
    }

    /// Interne Methode, um Observer-observations und age zu aktualisieren (für SDOstream)
    #[allow(dead_code)]
    pub(crate) fn update_observer_age(&mut self, index: usize, new_age: f64) -> bool {
        self.observers.update_age(index, new_age)
    }

    /// Interne Methode, um den schlechtesten Observer nach normalisiertem Score zu finden (für SDOstream)
    pub(crate) fn find_worst_normalized_score(&self) -> Option<(usize, f64)> {
        // Verwende die optimierte find_worst_normalized_score Methode - O(1)
        self.observers.find_worst_normalized_score()
    }

    /// Interne Methode, um alle aktiven Observer-Indizes zu erhalten (für SDOstream)
    /// Effizienter als get_active_observers_with_indices, wenn nur Indizes benötigt werden
    #[allow(dead_code)] // Für zukünftige Verwendung
    pub(crate) fn get_active_observer_indices(&self) -> Vec<usize> {
        let num_active = ((self.observers.len() as f64) * (1.0 - self.rho)).ceil() as usize;
        let num_active = num_active.max(1).min(self.observers.len());
        self.observers
            .iter_active(num_active)
            .map(|obs| obs.index)
            .collect()
    }

    /// Interne Methode, um aktive Observer zu iterieren ohne Kopie (für SDOstream)
    /// Effizienter als get_active_observers_with_indices, wenn nur gelesen wird
    pub(crate) fn iter_active_observers(&self) -> impl Iterator<Item = &Observer> {
        let num_active = ((self.observers.len() as f64) * (1.0 - self.rho)).ceil() as usize;
        let num_active = num_active.max(1).min(self.observers.len());
        self.observers.iter_active(num_active)
    }

    /// Interne Methode, um einen Observer zu ersetzen (für SDOstream)
    pub(crate) fn replace_observer(&mut self, old_index: usize, new_observer: Observer) -> bool {
        self.observers.replace(old_index, new_observer)
    }
}

impl Default for SDO {
    fn default() -> Self {
        Self::new()
    }
}
