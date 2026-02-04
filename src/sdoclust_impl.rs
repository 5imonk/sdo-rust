use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use std::collections::{HashMap, HashSet};
use std::f64;

use crate::sdo_impl::SDO;
use crate::utils::{data_to_matrix, label_score_results_to_py};

/// Sparse Data Observers Clustering (SDOclust) Algorithm
#[pyclass]
pub struct SDOclust {
    sdo: SDO, // Internes SDO-Objekt für Modell-Erstellung
    #[pyo3(get, set)]
    chi: usize, // χ - Anzahl der nächsten Observer für lokale Thresholds
    #[pyo3(get, set)]
    zeta: f64, // ζ - Mixing-Parameter für globale/lokale Thresholds
    #[pyo3(get, set)]
    min_cluster_size: usize, // e - minimale Clustergröße
    k: usize, // Anzahl der Observer (Modellgröße)
}

#[pymethods]
#[allow(clippy::too_many_arguments)]
impl SDOclust {
    #[new]
    #[pyo3(signature = (k, x, rho, chi = 4, zeta = 0.5, min_cluster_size = 2, distance = "euclidean".to_string(), minkowski_p = None))]
    pub fn new(
        k: usize,
        x: usize,
        rho: f64,
        chi: usize,
        zeta: f64,
        min_cluster_size: usize,
        distance: String,
        minkowski_p: Option<f64>,
    ) -> Self {
        Self {
            sdo: SDO::new(k, x, rho, distance, minkowski_p),
            chi,
            zeta,
            min_cluster_size,
            k,
        }
    }

    /// Lernt das Modell aus den Daten und führt Clustering durch
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

    /// Berechnet das Cluster-Label für einen oder mehrere Datenpunkte (Batch-Verarbeitung).
    /// Ein einzelner Punkt wird als Batch der Größe 1 behandelt.
    /// Returns (label, outlier_score) für einen Punkt oder Liste von (label, outlier_score) Tupeln für mehrere Punkte
    #[pyo3(signature = (points))]
    pub fn predict(&self, points: PyReadonlyArray2<f64>) -> PyResult<PyObject> {
        let (points_vec, rows) = data_to_matrix(points);
        let results = self.predict_impl(&points_vec, None);
        Python::with_gil(|py| label_score_results_to_py(&results, rows, py))
    }

    /// Konvertiert active_observers zu NumPy-Array für Python
    pub fn get_active_observers(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        self.sdo.get_active_observers(py)
    }

    /// Gibt die Cluster-Labels der aktiven Observer zurück (-1 = kein Label/Outlier).
    pub fn get_observer_labels(&self) -> Vec<i32> {
        self.sdo
            .observers
            .iter_observers(true)
            .map(|obs| obs.get_label().map(|l| l as i32).unwrap_or(-1))
            .collect()
    }

    /// Gibt die Anzahl der Cluster zurück
    pub fn n_clusters(&self) -> usize {
        // Gehe durch alle aktiven Observer und sammle eindeutige Labels
        let unique_labels: HashSet<usize> = self
            .sdo
            .observers
            .iter_observers(true)
            .filter_map(|obs| obs.get_label())
            .collect();
        unique_labels.len()
    }

    /// Gibt x zurück (Anzahl der nächsten Nachbarn)
    #[getter]
    pub fn x(&self) -> usize {
        self.sdo.x
    }

    /// Debug/Inspection: Gibt den aktuellen global_threshold aus dem ObserverSet zurück.
    /// (Dieser Wert wird in `set_thresholds` gesetzt und in `find_connected_components` verwendet.)
    pub fn get_global_threshold(&self) -> f64 {
        self.sdo.observers.global_threshold
    }

    /// Debug: Gibt die gefundenen Connected Components zurück (ohne Labels).
    /// Jede Komponente ist eine Liste von Observer-Indizes (aktive Observer).
    /// Nützlich um zu prüfen, ob falsch verbunden (zu viele in einer Komponente) oder falsch gelabelt.
    pub fn get_connected_components_debug(&mut self) -> Vec<Vec<usize>> {
        self.sdo
            .observers
            .get_connected_components_for_debug(self.zeta, self.min_cluster_size)
    }

    /// Gibt aktive Observer inkl. finaler Threshold-Radien für Visualisierung.
    /// Returns (points, labels, final_threshold_radii) mit final = zeta * local + (1-zeta) * global.
    pub fn get_active_observers_with_final_thresholds(
        &self,
        py: Python<'_>,
    ) -> PyResult<(Py<PyArray2<f64>>, Vec<i32>, Vec<f64>)> {
        let active_observers = self.sdo.observers.get_observers(true);
        if active_observers.is_empty() {
            let empty = PyArray2::zeros_bound(py, (0, 0), false);
            return Ok((empty.unbind(), vec![], vec![]));
        }
        let local_thresholds: Vec<f64> = active_observers.iter().map(|o| o.local_threshold).collect();
        let global_threshold = local_thresholds.iter().sum::<f64>() / local_thresholds.len() as f64;
        let zeta = self.zeta;
        let final_radii: Vec<f64> = local_thresholds
            .iter()
            .map(|h| zeta * h + (1.0 - zeta) * global_threshold)
            .collect();
        let labels: Vec<i32> = active_observers
            .iter()
            .map(|o| o.get_label().map(|l| l as i32).unwrap_or(-1))
            .collect();
        let rows = active_observers.len();
        let cols = active_observers[0].data.len();
        let array = PyArray2::zeros_bound(py, (rows, cols), false);
        unsafe {
            let mut arr = array.as_array_mut();
            for (i, obs) in active_observers.iter().enumerate() {
                for (j, &v) in obs.data.iter().enumerate() {
                    arr[[i, j]] = v;
                }
            }
        }
        Ok((array.unbind(), labels, final_radii))
    }

    /// Gibt alle Observer-Informationen inklusive Thresholds und Distanzen zurück
    /// Für Connected Components Test
    /// Returns: (observers_data, global_threshold, active_indices, distance_matrix_dict)
    /// observers_data: Vec<(data, observations, age, is_active, local_threshold, index)>
    /// distance_matrix_dict: Python-Dict {index: [(neighbor_index, distance), ...]} für aktive Observer
    pub fn get_all_observer_data_for_testing(&self, py: Python) -> PyResult<PyObject> {
        use pyo3::types::PyDict;

        let mut observers_data = Vec::new();
        let mut distance_matrix: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();

        // Sammle alle Observer-Daten
        for observer in self.sdo.observers.iter_observers(false) {
            let is_active = self.sdo.observers.is_active(observer.index);
            observers_data.push((
                observer.data.clone(),
                observer.observations,
                observer.age,
                is_active,
                observer.local_threshold,
                observer.index,
            ));
        }

        // Sammle aktive Indizes (strict Top-N via iter_observers(true))
        let active_indices: Vec<usize> = self
            .sdo
            .observers
            .iter_observers(true)
            .map(|obs| obs.index)
            .collect();
        let active_set: std::collections::HashSet<usize> = active_indices.iter().copied().collect();

        // Sammle Distanzen für jeden aktiven Observer
        for &idx in &active_indices {
            let neighbors = self
                .sdo
                .observers
                .get_neighbors_within_threshold(idx, f64::INFINITY);
            // Filtere nur aktive Nachbarn (strict Top-N)
            let active_neighbors: Vec<(usize, f64)> = neighbors
                .into_iter()
                .filter(|(neighbor_idx, _)| active_set.contains(neighbor_idx))
                .collect();
            distance_matrix.insert(idx, active_neighbors);
        }

        // Berechne global_threshold als Durchschnitt der lokalen Thresholds der aktiven Observer
        let local_thresholds: Vec<f64> = active_indices
            .iter()
            .filter_map(|&idx| {
                observers_data
                    .iter()
                    .find(|obs| obs.5 == idx)
                    .map(|obs| obs.4)
            })
            .collect();

        let global_threshold = if !local_thresholds.is_empty() {
            local_thresholds.iter().sum::<f64>() / local_thresholds.len() as f64
        } else {
            f64::INFINITY
        };

        // Konvertiere zu Python-Objekten
        let observers_list = PyList::new_bound(
            py,
            observers_data.iter().map(|obs| {
                PyTuple::new_bound(
                    py,
                    [
                        obs.0.clone().into_py(py),
                        obs.1.into_py(py),
                        obs.2.into_py(py),
                        obs.3.into_py(py),
                        obs.4.into_py(py),
                        obs.5.into_py(py),
                    ],
                )
            }),
        );

        let active_indices_list =
            PyList::new_bound(py, active_indices.iter().map(|&idx| idx.into_py(py)));

        let distance_dict = PyDict::new_bound(py);
        for (idx, neighbors) in &distance_matrix {
            let neighbors_list = PyList::new_bound(
                py,
                neighbors.iter().map(|(n_idx, dist)| {
                    PyTuple::new_bound(py, [n_idx.into_py(py), dist.into_py(py)])
                }),
            );
            distance_dict.set_item(idx.into_py(py), neighbors_list)?;
        }

        let result = PyTuple::new_bound(
            py,
            [
                observers_list.into(),
                global_threshold.into_py(py),
                active_indices_list.into(),
                distance_dict.into(),
            ],
        );

        Ok(result.into_py(py))
    }
}

impl SDOclust {
    pub fn learn_impl(&mut self, data: &Vec<Vec<f64>>) {
        // Verwende SDO für Modell-Erstellung (Sample, Observe, Clean)
        self.sdo.learn_impl(data);

        // Führe vollständiges Clustering durch (inkl. Thresholds, Connected Components, Label-Zuweisung)
        // Kein Fading für statisches Clustering
        self.sdo.observers.learn_clustering(
            self.chi,
            self.zeta,
            self.min_cluster_size,
            None, // Kein Fading für statisches Clustering
            None, // Keine Zeit für statisches Clustering
        );
    }

    /// Vorhersage für einen einzelnen Punkt (Rust-intern).
    pub fn predict_point(&self, point: &[f64], learn: Option<bool>) -> (i32, f64) {
        let (median, active_neighbors, _all_neighbors_opt) = self.sdo.predict_point(point, learn);
        let nearest_indices: Vec<usize> = active_neighbors.iter().map(|n| n.index).collect();

        // Zähle die Häufigkeit der Labels
        let mut label_counts: HashMap<usize, usize> = HashMap::new();
        for idx in nearest_indices {
            if let Some(obs) = self.sdo.observers.get(idx) {
                if let Some(label) = obs.get_label() {
                    *label_counts.entry(label).or_insert(0) += 1;
                }
            }
        }

        // Gib das häufigste Label zurück (konvertiere zu i32 für Python-API)
        if let Some((&most_common_label, _)) = label_counts.iter().max_by_key(|(_, &count)| count) {
            (most_common_label as i32, median)
        } else {
            (-1, median) // Kein Label gefunden (Outlier)
        }
    }

    /// Batch-Vorhersage (Rust-intern).
    pub fn predict_impl(&self, points: &Vec<Vec<f64>>, learn: Option<bool>) -> Vec<(i32, f64)> {
        points
            .iter()
            .map(|point| self.predict_point(point, learn))
            .collect()
    }
}

impl Default for SDOclust {
    fn default() -> Self {
        Self::new(200, 5, 0.2, 4, 0.5, 2, "euclidean".to_string(), None)
    }
}
