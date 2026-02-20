use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::f64;

use crate::obs::{neighbor_distance, NeighborInfo};
use crate::obset::{min_sample_size_hypergeometric, ObserverSet};
use crate::utils::DistanceMetric;
use crate::utils::{compute_median, data_to_matrix, scores_single_or_list_to_py};

/// Sparse Data Observers (SDO) Algorithm
#[pyclass]
#[allow(clippy::upper_case_acronyms)]
pub struct SDO {
    // Observer-Set, sortiert nach observations (verwendet immer Brute-Force)
    pub(crate) observers: ObserverSet,
    rho: f64,
    #[pyo3(get, set)]
    pub(crate) x: usize,
    k: usize,      // Anzahl der Observer (Modellgröße)
    p_safe: f64,   // Probability threshold for x_safe calculation (default 0.98)
    x_safe: usize, // Calculated safe sample size for k-NN search
}

#[pymethods]
impl SDO {
    #[new]
    #[pyo3(signature = (k, x, rho, distance = "euclidean".to_string(), minkowski_p = None, fading = None, p_safe = 0.98))]
    pub fn new(
        k: usize,
        x: usize,
        rho: f64,
        distance: String,
        minkowski_p: Option<f64>,
        fading: Option<f64>,
        p_safe: f64,
    ) -> Self {
        let distance_metric = match distance.to_lowercase().as_str() {
            "manhattan" => DistanceMetric::Manhattan,
            "chebyshev" => DistanceMetric::Chebyshev,
            "minkowski" => DistanceMetric::Minkowski,
            _ => DistanceMetric::Euclidean,
        };

        let instance = Self {
            observers: ObserverSet::new(distance_metric, minkowski_p, fading),
            rho,
            x,
            k,
            p_safe,
            x_safe: x, // Initial value, will be updated when num_active is set
        };
        // x_safe will be calculated after num_active is set in learn_impl
        instance
    }

    /// Lernt das Modell aus den Daten
    #[pyo3(signature = (data, *))]
    pub fn learn(&mut self, data: PyReadonlyArray2<f64>) -> PyResult<()> {
        // Konvertiere Daten zu Vec<Vec<f64>>
        let (data_vec, rows) = data_to_matrix(data);

        // Überprüfe auf leere Daten oder k=0
        if rows == 0 || self.k == 0 {
            return Ok(());
        }

        // Überprüfe, ob Anzahl der Datenpunkte mindestens k ist
        if rows < self.k {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Anzahl der Datenpunkte muss mindestens k sein",
            ));
        }

        // Rufe die interne Lernmethode auf
        self.learn_impl(&data_vec);

        Ok(())
    }

    /// Berechnet den Outlier-Score für einen oder mehrere Datenpunkte (Batch-Verarbeitung).
    /// Ein einzelner Punkt wird als Batch der Größe 1 behandelt.
    #[pyo3(signature = (points))]
    pub fn predict(&self, points: PyReadonlyArray2<f64>) -> PyResult<PyObject> {
        let (points_vec, rows) = data_to_matrix(points);
        let results = self.predict_impl(&points_vec, None, None);
        let scores: Vec<f64> = results.iter().map(|(median, _, _)| *median).collect();
        Python::with_gil(|py| scores_single_or_list_to_py(&scores, rows, py))
    }

    /// Konvertiert active_observers zu NumPy-Array für Python
    pub fn get_active_observers(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        let active_observers = self.observers.get_observers(true);

        if active_observers.is_empty() {
            let array = PyArray2::zeros_bound(py, (0, 0), false);
            return Ok(array.unbind());
        }

        let rows = active_observers.len();
        let cols = active_observers[0].1.len(); // data is second element in tuple
        let array = PyArray2::zeros_bound(py, (rows, cols), false);

        unsafe {
            let mut array_mut = array.as_array_mut();
            for (i, (_idx, data, _obs, _time, _age)) in active_observers.iter().enumerate() {
                for (j, &value) in data.iter().enumerate() {
                    array_mut[[i, j]] = value;
                }
            }
        }

        Ok(array.unbind())
    }
}

impl SDO {
    /// Gibt k zurück (für SDOstream etc.)
    pub(crate) fn get_k(&self) -> usize {
        self.k
    }

    /// Gibt rho zurück (für SDOstream etc.)
    pub(crate) fn get_rho(&self) -> f64 {
        self.rho
    }

    /// Set num_active and update x_safe accordingly
    /// Should be used instead of directly calling observers.set_num_active()
    pub(crate) fn set_num_active(&mut self, num_active: usize) {
        self.observers.set_num_active(num_active);
        self.update_x_safe();
    }

    /// Update x_safe based on current k, num_active, and x
    fn update_x_safe(&mut self) {
        let num_active = self.observers.get_num_active();
        let k_total = self.observers.len();

        if k_total == 0 || num_active == 0 {
            self.x_safe = self.x;
            return;
        }

        // Calculate x_safe using hypergeometric distribution
        let x_safe_result = min_sample_size_hypergeometric(
            k_total as u64,
            num_active as u64,
            self.x as u64,
            self.p_safe,
        );

        self.x_safe = x_safe_result.map(|v| v as usize).unwrap_or_else(|| {
            // Fallback: if calculation fails, use a conservative estimate
            // Search at least x observers, but try to get enough to likely have x active
            let ratio = num_active as f64 / k_total as f64;
            if ratio > 0.0 {
                // Estimate: need to sample enough to get x active with high probability
                ((self.x as f64 / ratio).ceil() as usize)
                    .min(k_total)
                    .max(self.x)
            } else {
                self.x
            }
        });
    }

    pub(crate) fn learn_impl(&mut self, data: &Vec<Vec<f64>>) {
        // Schritt 1: Sample
        let mut rng = thread_rng();
        let observers_data: Vec<Vec<f64>> = data
            .choose_multiple(&mut rng, self.k.min(data.len()))
            .cloned()
            .collect();

        // Schritt 2: Erstelle ObserverSet mit allen Observers (ohne observations)
        for (idx, observer_data) in observers_data.iter().enumerate() {
            self.observers.insert(
                idx,
                observer_data.clone(),
                0.0,               // observations
                0.0,               // time
                data.len() as f64, // age
            );
        }

        // Schritt 3: Berechne observations für jeden Observer mit Nearest Neighbor Search
        // Für jeden Datenpunkt: Finde x nächste Observer (unter allen Observern) und erhöhe deren observations.
        // During learning, num_active is still 0, so we use x_safe = x (fallback)
        // We search for x observers directly since all are "active" during learning
        let x_safe_learn = self.x; // During learning, use x directly

        for data_point in data {
            // Use knn_search to get x nearest observers
            let candidates = self.observers.knn_search(data_point, x_safe_learn);

            // Extract indices (all are considered during learning)
            let nearest_indices: Vec<usize> = candidates
                .iter()
                .take(self.x)
                .map(|(node, _)| node.value.1)
                .collect();

            // Erhöhe observations für jeden dieser Observer um 1
            for idx in nearest_indices {
                if let Some(current_obs) = self.observers.get_observations(idx) {
                    self.observers.update_observations(idx, current_obs + 1.0);
                }
            }
        }

        // Set num_active (this also updates x_safe)
        self.set_num_active(((self.observers.len() as f64) * (1.0 - self.rho)).ceil() as usize);
    }

    /// Vorhersage für einen einzelnen Punkt (Rust-intern).
    pub(crate) fn predict_point(
        &self,
        point: &[f64],
        learn: Option<bool>,
        k_learn: Option<usize>,
    ) -> (f64, Vec<NeighborInfo>, Option<Vec<NeighborInfo>>) {
        if self.observers.is_empty() {
            panic!("No observers found during prediction!");
        }

        // Calculate total search size: x_safe + k_learn if learning
        let k_learn_val = k_learn.unwrap_or(0);
        let total_search_size = if learn == Some(true) {
            self.x_safe + k_learn_val
        } else {
            self.x_safe
        };

        // Use knn_search to get candidates
        let candidates = self.observers.knn_search(point, total_search_size);

        // Extract x active neighbors (returns Vec<NeighborInfo>)
        let active_neighbors = self.observers.extract_x_neighbors(&candidates, self.x, true);

        if active_neighbors.is_empty() {
            panic!("No active observers found during prediction!");
        }

        let distances: Vec<f64> = active_neighbors.iter().map(neighbor_distance).collect();
        let median = compute_median(&distances);

        // For learn mode, return all candidates as all_neighbors (mtree structure as NeighborInfo)
        let all_neighbors_opt = if learn == Some(true) {
            Some(
                candidates
                    .iter()
                    .take(total_search_size)
                    .map(|(node, dist)| NeighborInfo::MTree(std::sync::Arc::clone(node), *dist))
                    .collect(),
            )
        } else {
            None
        };

        (median, active_neighbors, all_neighbors_opt)
    }

    /// Batch-Vorhersage (Rust-intern). Nutzt knn_search_batch für mehrere Punkte.
    pub(crate) fn predict_impl(
        &self,
        points: &[Vec<f64>],
        learn: Option<bool>,
        k_learn: Option<usize>,
    ) -> Vec<(f64, Vec<NeighborInfo>, Option<Vec<NeighborInfo>>)> {
        if points.is_empty() {
            return Vec::new();
        }
        if self.observers.is_empty() {
            panic!("No observers found during prediction!");
        }

        // Calculate total search size: x_safe + k_learn if learning
        let k_learn_val = k_learn.unwrap_or(0);
        let total_search_size = if learn == Some(true) {
            self.x_safe + k_learn_val
        } else {
            self.x_safe
        };

        // Use knn_search_batch to get candidates for all points
        let candidates_batch = self.observers.knn_search_batch(points, total_search_size);

        candidates_batch
            .into_iter()
            .map(|candidates| {
                // Extract x active neighbors (returns Vec<NeighborInfo>)
                let active_neighbors =
                    self.observers.extract_x_neighbors(&candidates, self.x, true);

                if active_neighbors.is_empty() {
                    panic!("No active observers found during prediction!");
                }

                let distances: Vec<f64> = active_neighbors.iter().map(neighbor_distance).collect();
                let median = compute_median(&distances);

                // For learn mode, return all candidates as all_neighbors
                let all_neighbors_opt = if learn == Some(true) {
                    Some(
                        candidates
                            .iter()
                            .take(total_search_size)
                            .map(|(node, dist)| {
                                NeighborInfo::MTree(std::sync::Arc::clone(node), *dist)
                            })
                            .collect(),
                    )
                } else {
                    None
                };

                (median, active_neighbors, all_neighbors_opt)
            })
            .collect()
    }

    /// Interne Methode, um einen Observer zu ersetzen (für SDOstream)
    /// Wird nicht mehr benötigt - replace() ist jetzt direkt auf ObserverSet
    #[allow(dead_code)]
    pub(crate) fn replace_observer_legacy(
        &mut self,
        _old_index: usize,
        _new_index: usize,
        _new_data: Vec<f64>,
        _new_observations: f64,
        _new_time: f64,
        _new_age: f64,
    ) -> bool {
        // Legacy method - use observers.replace() directly
        false
    }
}

impl Default for SDO {
    fn default() -> Self {
        Self::new(200, 5, 0.2, "euclidean".to_string(), None, None, 0.98)
    }
}
