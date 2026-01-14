use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::f64;

use crate::observer::{Observer, ObserverSet};
use crate::utils::DistanceMetric;

// SpatialTreeObserver moved to observer_set.rs (now uses HNSW)

/// Sparse Data Observers (SDO) Algorithm
#[pyclass]
#[allow(clippy::upper_case_acronyms)]
pub struct SDO {
    // Observer-Set, sortiert nach observations (verwendet HNSW intern)
    pub(crate) observers: ObserverSet,
    distance_metric: DistanceMetric,
    minkowski_p: Option<f64>,
    rho: f64,
    #[pyo3(get, set)]
    pub(crate) x: usize,
    k: usize, // Anzahl der Observer (Modellgröße)
}

#[pymethods]
impl SDO {
    #[new]
    #[pyo3(signature = (k, x, rho, distance = "euclidean".to_string(), minkowski_p = None, use_brute_force = false))]
    pub fn new(k: usize, x: usize, rho: f64, distance: String, minkowski_p: Option<f64>, use_brute_force: bool) -> Self {
        let distance_metric = match distance.to_lowercase().as_str() {
            "manhattan" => DistanceMetric::Manhattan,
            "chebyshev" => DistanceMetric::Chebyshev,
            "minkowski" => DistanceMetric::Minkowski,
            _ => DistanceMetric::Euclidean,
        };

        let mut instance = Self {
            observers: ObserverSet::new(),
            distance_metric,
            minkowski_p,
            rho,
            x,
            k,
        };
        instance.observers.set_use_brute_force(use_brute_force);
        instance
    }

    /// Lernt das Modell aus den Daten
    /// Wenn time angegeben, wird dieser Wert für alle Observer als time gesetzt
    #[pyo3(signature = (data, *, time = None))]
    pub fn learn(
        &mut self,
        data: PyReadonlyArray2<f64>,
        time: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<()> {
        let data_slice = data.as_array();
        let rows = data_slice.nrows();
        let cols = data_slice.ncols();

        if rows == 0 || self.k == 0 {
            return Ok(());
        }

        if rows < self.k {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Anzahl der Datenpunkte muss mindestens k sein",
            ));
        }

        // Konvertiere NumPy-Array zu Vec<Vec<f64>>
        let data_vec: Vec<Vec<f64>> = (0..rows)
            .map(|i| (0..cols).map(|j| data_slice[[i, j]]).collect())
            .collect();

        // Bestimme Zeit für alle Punkte
        let t0 = if let Some(time_array) = &time {
            let time_slice = time_array.as_array();
            if time_slice.len() != 1 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Zeit muss ein 1D-Array mit einem Wert sein",
                ));
            }
            time_slice[[0]]
        } else {
            0.0 // Default: time = 0
        };

        // Schritt 1: Sample
        let mut rng = thread_rng();
        let observers_data: Vec<Vec<f64>> = data_vec
            .choose_multiple(&mut rng, self.k.min(data_vec.len()))
            .cloned()
            .collect();

        // Schritt 2: Erstelle ObserverSet mit allen Observers (ohne observations)
        // ObserverSet wurde bereits in initialize() erstellt, aber wir müssen sicherstellen,
        // dass die Parameter gesetzt sind
        self.observers
            .set_tree_params(self.distance_metric, self.minkowski_p);
        for (idx, observer_data) in observers_data.iter().enumerate() {
            let observer = Observer {
                data: observer_data.clone(),
                observations: 0.0,
                time: t0,
                age: rows as f64,
                index: idx,
                label: None,
                cluster_observations: Vec::new(),
            };
            self.observers.insert(observer);
        }

        // Schritt 3: Berechne observations für jeden Observer mit Nearest Neighbor Search

        // Für jeden Datenpunkt: Finde x nächste Observer und erhöhe deren observations
        for data_point in &data_vec {
            // Finde die Indizes der x nächsten Observer zu diesem Datenpunkt
            let nearest_indices = self
                .observers
                .search_k_nearest_indices(data_point, self.x, false);

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

        // Suche nur unter den aktiven Observers
        let distances = self
            .observers
            .search_k_nearest_distances(&point_vec, self.x, true);

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
    /// Interne Methode, um einen Observer zu ersetzen (für SDOstream)
    pub(crate) fn replace_observer(&mut self, old_index: usize, new_observer: Observer) -> bool {
        self.observers.replace(old_index, new_observer)
    }
}

impl Default for SDO {
    fn default() -> Self {
        Self::new(10, 5, 0.2, "euclidean".to_string(), None, false)
    }
}
