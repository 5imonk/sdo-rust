use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashMap;
use std::f64;

use crate::obs::{NeighborInfo, Observer};
use crate::obset::ObserverSet;
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
    k: usize, // Anzahl der Observer (Modellgröße)
}

#[pymethods]
impl SDO {
    #[new]
    #[pyo3(signature = (k, x, rho, distance = "euclidean".to_string(), minkowski_p = None))]
    pub fn new(k: usize, x: usize, rho: f64, distance: String, minkowski_p: Option<f64>) -> Self {
        let distance_metric = match distance.to_lowercase().as_str() {
            "manhattan" => DistanceMetric::Manhattan,
            "chebyshev" => DistanceMetric::Chebyshev,
            "minkowski" => DistanceMetric::Minkowski,
            _ => DistanceMetric::Euclidean,
        };

        let instance = Self {
            observers: ObserverSet::new(distance_metric, minkowski_p),
            rho,
            x,
            k,
        };
        instance
    }

    /// Lernt das Modell aus den Daten
    /// Wenn time angegeben, wird dieser Wert für alle Observer als time gesetzt
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

impl SDO {
    /// Gibt k zurück (für SDOstream etc.)
    pub(crate) fn get_k(&self) -> usize {
        self.k
    }

    /// Gibt rho zurück (für SDOstream etc.)
    pub(crate) fn get_rho(&self) -> f64 {
        self.rho
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
            let observer = Observer {
                data: observer_data.clone(),
                observations: 0.0,
                time: 0.0,
                age: data.len() as f64,
                index: idx,
                local_threshold: f64::INFINITY,
                label_observations: HashMap::new(),
                label_time: 0.0,
            };
            self.observers.insert(observer);
        }

        // Schritt 3: Berechne observations für jeden Observer mit Nearest Neighbor Search
        // Für jeden Datenpunkt: Finde x nächste Observer (unter allen Observern) und erhöhe deren observations.
        // learn: Some(true) → alle Observer werden durchsucht, Rückgabe in nearest_all (num_active ist hier noch 0).
        for data_point in data {
            let (_nearest_active, nearest_all_opt) =
                self.observers
                    .search_neighbors_unified(data_point, self.x, Some(true), None);
            let nearest_indices: Vec<usize> = nearest_all_opt
                .as_ref()
                .expect("learn=true ⇒ nearest_all is always filled")
                .iter()
                .map(|n| n.index)
                .collect();

            // Erhöhe observations für jeden dieser Observer um 1
            for idx in nearest_indices {
                if let Some(observer) = self.observers.get(idx) {
                    let current_obs = observer.observations;
                    self.observers.update_observations(idx, current_obs + 1.0);
                }
            }
        }

        // Set num_active
        self.observers
            .set_num_active(((self.observers.len() as f64) * (1.0 - self.rho)).ceil() as usize);
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

        // Suche nur unter den aktiven Observers (using optimized unified search mit aktiven info)
        let (active_neighbors, all_neighbors_opt) = self
            .observers
            .search_neighbors_unified(point, self.x, learn, k_learn);
        let distances: Vec<f64> = active_neighbors.iter().map(|n| n.distance).collect();

        if distances.is_empty() {
            panic!("No active observers found during prediction!");
        }

        let median = compute_median(&distances);

        (median, active_neighbors, all_neighbors_opt)
    }

    /// Batch-Vorhersage (Rust-intern).
    pub(crate) fn predict_impl(
        &self,
        points: &Vec<Vec<f64>>,
        learn: Option<bool>,
        k_learn: Option<usize>,
    ) -> Vec<(f64, Vec<NeighborInfo>, Option<Vec<NeighborInfo>>)> {
        points
            .iter()
            .map(|point| self.predict_point(point, learn, k_learn))
            .collect()
    }

    /// Interne Methode, um einen Observer zu ersetzen (für SDOstream)
    pub(crate) fn replace_observer(&mut self, old_index: usize, new_observer: Observer) -> bool {
        self.observers.replace(old_index, new_observer)
    }
}

impl Default for SDO {
    fn default() -> Self {
        Self::new(200, 5, 0.2, "euclidean".to_string(), None)
    }
}
