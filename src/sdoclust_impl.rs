use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use std::collections::{HashMap, HashSet};
use std::f64;

use crate::sdo_impl::SDO;
use crate::utils::data_to_matrix;

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

        if self.sdo.observers.is_empty() {
            return Python::with_gil(|py| {
                if rows == 1 {
                    let tuple = PyTuple::new_bound(
                        py,
                        [(-1i32).into_py(py), f64::NAN.into_py(py)],
                    );
                    Ok(tuple.into_py(py))
                } else {
                    let empty_results: Vec<Py<PyAny>> = (0..rows)
                        .map(|_| {
                            let tuple = PyTuple::new_bound(
                                py,
                                [(-1i32).into_py(py), f64::NAN.into_py(py)],
                            );
                            tuple.into_py(py)
                        })
                        .collect();
                    let list = PyList::new_bound(py, empty_results);
                    Ok(list.into_py(py))
                }
            });
        }

        let results = self.predict_impl(&points_vec, None);

        // Wenn nur ein Punkt: Rückgabe als Tupel, sonst als Liste von Tupeln
        Python::with_gil(|py| {
            if rows == 1 {
                let (label, score) = results[0];
                let tuple = PyTuple::new_bound(
                    py,
                    [label.into_py(py), score.into_py(py)],
                );
                Ok(tuple.into_py(py))
            } else {
                let list: Vec<Py<PyAny>> = results
                    .iter()
                    .map(|(label, score)| {
                        let tuple = PyTuple::new_bound(
                            py,
                            [label.into_py(py), score.into_py(py)],
                        );
                        tuple.into_py(py)
                    })
                    .collect();
                let py_list = PyList::new_bound(py, list);
                Ok(py_list.into_py(py))
            }
        })
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
    pub fn predict_impl(
        &self,
        points: &Vec<Vec<f64>>,
        learn: Option<bool>,
    ) -> Vec<(i32, f64)> {
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
