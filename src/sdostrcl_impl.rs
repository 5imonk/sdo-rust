use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::obs::NeighborInfo;
use crate::sdostream_impl::SDOstream;
use crate::utils::{data_to_matrix, times_to_vec_batch};

/// SDOstreamclust Algorithm - Streaming-Version von SDOclust
/// Baut auf SDOstream auf und fügt Clustering-Logik hinzu
#[pyclass]
#[allow(clippy::upper_case_acronyms)]
pub struct SDOstreamclust {
    sdostream: SDOstream, // Basis SDOstream-Implementierung
    chi: usize,
    zeta: f64,
    min_cluster_size: usize,
}

#[pymethods]
#[allow(clippy::too_many_arguments)]
impl SDOstreamclust {
    #[new]
    #[pyo3(signature = (k, x, t_fading, t_sampling = None, chi_min = 1, chi_prop = 0.05, zeta = 0.6, min_cluster_size = 2, distance = "euclidean".to_string(), minkowski_p = None, rho = 0.1, dimension = None, data = None, time = None))]
    pub fn new(
        k: usize,
        x: usize,
        t_fading: f64,
        t_sampling: Option<f64>,
        chi_min: usize,
        chi_prop: f64,
        zeta: f64,
        min_cluster_size: usize,
        distance: String,
        minkowski_p: Option<f64>,
        rho: f64,
        dimension: Option<usize>,
        data: Option<PyReadonlyArray2<f64>>,
        time: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Self> {
        // Validate parameters
        if !(0.0..=1.0).contains(&chi_prop) {
            return Err(PyErr::new::<PyValueError, _>(
                "chi_prop must be between 0.0 and 1.0",
            ));
        }

        if !(0.0..=1.0).contains(&zeta) {
            return Err(PyErr::new::<PyValueError, _>(
                "zeta must be between 0.0 and 1.0",
            ));
        }

        Ok(Self {
            sdostream: SDOstream::new(
                k,
                x,
                t_fading,
                t_sampling,
                distance,
                minkowski_p,
                rho,
                dimension,
                data,
                time,
            )?,
            chi: ((chi_min as f64).max(chi_prop * k as f64) as usize).max(1),
            zeta,
            min_cluster_size,
        })
    }

    /// Verarbeitet einen oder mehrere Datenpunkte aus dem Stream (Batch-Verarbeitung).
    /// Ein einzelner Punkt wird als Batch der Größe 1 behandelt.
    #[pyo3(signature = (points, *, time = None))]
    pub fn learn(
        &mut self,
        points: PyReadonlyArray2<f64>,
        time: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<PyObject> {
        let (points_vec, rows) = data_to_matrix(points);

        // Bestimme Zeiten für alle Punkte
        let times_vec = times_to_vec_batch(
            time,
            rows,
            self.sdostream.get_use_explicit_time(),
            self.sdostream.get_data_points_processed(),
        )?;

        let results = self.learn_impl(&points_vec, &times_vec);

        Python::with_gil(|py| {
            if rows == 1 {
                let t = pyo3::types::PyTuple::new_bound(
                    py,
                    [results[0].0.into_py(py), results[0].1.into_py(py)],
                );
                Ok(t.into_py(py))
            } else {
                let list: Vec<Py<PyAny>> = results
                    .iter()
                    .map(|(l, s)| {
                        let t = pyo3::types::PyTuple::new_bound(py, [l.into_py(py), s.into_py(py)]);
                        t.into_py(py)
                    })
                    .collect();
                let py_list = pyo3::types::PyList::new_bound(py, list);
                Ok(py_list.into_py(py))
            }
        })
    }

    /// Berechnet Cluster-Label und Outlier-Score für einen oder mehrere Datenpunkte (Batch-Verarbeitung).
    #[pyo3(signature = (points))]
    pub fn predict(&self, points: PyReadonlyArray2<f64>) -> PyResult<PyObject> {
        let (points_vec, rows) = data_to_matrix(points);

        let results = self.predict_impl(&points_vec, None);

        Python::with_gil(|py| {
            if rows == 1 {
                let t = pyo3::types::PyTuple::new_bound(
                    py,
                    [results[0].0.into_py(py), results[0].1.into_py(py)],
                );
                Ok(t.into_py(py))
            } else {
                let list: Vec<Py<PyAny>> = results
                    .iter()
                    .map(|(l, s, _)| {
                        let t = pyo3::types::PyTuple::new_bound(py, [l.into_py(py), s.into_py(py)]);
                        t.into_py(py)
                    })
                    .collect();
                let py_list = pyo3::types::PyList::new_bound(py, list);
                Ok(py_list.into_py(py))
            }
        })
    }

    /// Gibt die Positionen der aktiven Observer als NumPy-Array zurück (Modell für Label-Vorhersage).
    pub fn get_active_observers(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        self.sdostream.get_sdo().get_active_observers(py)
    }

    /// Gibt die Cluster-Labels der aktiven Observer zurück (-1 = kein Label/Outlier).
    pub fn get_observer_labels(&self) -> Vec<i32> {
        self.sdostream
            .get_sdo()
            .observers
            .iter_observers(true)
            .map(|obs| obs.get_label().map(|l| l as i32).unwrap_or(-1))
            .collect()
    }
}

impl SDOstreamclust {
    /// Verarbeitet einen einzelnen Datenpunkt (Rust-intern).
    pub fn learn_point(&mut self, point: &Vec<f64>, time: f64) -> (i32, f64) {
        let (median, active_neighbors, _all_neighbors_opt) =
            self.sdostream.learn_point(point, time);
        let nearest_active_indices: Vec<usize> = active_neighbors.iter().map(|n| n.index).collect();

        let fading = self.sdostream.get_fading();
        let _cluster_map = self.sdostream.get_sdo_mut().observers.learn_clustering(
            self.chi,
            self.zeta,
            self.min_cluster_size,
            Some(fading),
            Some(time),
        );

        let predicted_label = self.compute_label(&nearest_active_indices);
        (predicted_label, median)
    }

    /// Verarbeitet einen Batch (Rust-intern).
    pub fn learn_impl(&mut self, points: &[Vec<f64>], times: &[f64]) -> Vec<(i32, f64)> {
        assert_eq!(points.len(), times.len());
        let mut results = Vec::with_capacity(points.len());
        for (point, time) in points.iter().zip(times.iter()) {
            results.push(self.learn_point(point, *time));
        }
        results
    }

    /// Vorhersage für einen einzelnen Punkt (Rust-intern).
    pub fn predict_point(
        &self,
        point: &[f64],
        learn: Option<bool>,
    ) -> (i32, f64, Option<Vec<NeighborInfo>>) {
        let point_vec: Vec<f64> = point.to_vec();
        let (median, active_neighbors, all_neighbors_opt) =
            self.sdostream.predict_point(&point_vec, learn);
        let nearest_active_indices: Vec<usize> = active_neighbors.iter().map(|n| n.index).collect();
        let predicted_label = self.compute_label(&nearest_active_indices);
        (predicted_label, median, all_neighbors_opt)
    }

    /// Batch-Vorhersage (Rust-intern).
    pub fn predict_impl(
        &self,
        points: &[Vec<f64>],
        learn: Option<bool>,
    ) -> Vec<(i32, f64, Option<Vec<NeighborInfo>>)> {
        let batch = self.sdostream.predict_impl(points, learn);
        batch
            .into_iter()
            .map(|(median, active_neighbors, all_neighbors_opt)| {
                let indices: Vec<usize> = active_neighbors.iter().map(|n| n.index).collect();
                let label = self.compute_label(&indices);
                (label, median, all_neighbors_opt)
            })
            .collect()
    }
}

impl SDOstreamclust {
    pub fn compute_label(&self, indices: &[usize]) -> i32 {
        if indices.is_empty() {
            panic!("No active observers found during prediction!");
        }

        // Berechne normalisierte Cluster-Scores der x-nächsten Observer
        let label_scores = self
            .sdostream
            .get_sdo()
            .observers
            .get_normalized_cluster_scores(indices);

        // Finde Label mit maximalem Score (konvertiere zu i32 für Python-API)
        let predicted_label = label_scores
            .iter()
            .max_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&label, _)| label as i32)
            .unwrap_or(-1); // -1 wenn keine Beobachtungen
        predicted_label
    }

    /// Gibt eine Referenz auf das interne SDOstream-Objekt zurück
    pub fn get_sdostream(&self) -> &SDOstream {
        &self.sdostream
    }

    /// Gibt eine mutable Referenz auf das interne SDOstream-Objekt zurück
    pub fn get_sdostream_mut(&mut self) -> &mut SDOstream {
        &mut self.sdostream
    }
}

impl Default for SDOstreamclust {
    fn default() -> Self {
        // Don't unwrap here - return a proper default or handle error
        Self {
            sdostream: SDOstream::default(),
            chi: 10,
            zeta: 0.6,
            min_cluster_size: 2,
        }
    }
}
