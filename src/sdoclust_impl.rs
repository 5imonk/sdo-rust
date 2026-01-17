use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::f64;

use crate::sdo_impl::SDO;

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
        let data_slice = data.as_array();
        let rows = data_slice.nrows();

        if rows == 0 || self.k == 0 {
            return Ok(());
        }

        // Verwende SDO für Modell-Erstellung (Sample, Observe, Clean)
        self.sdo.learn(data, None)?;

        // Führe Clustering durch (schreibe Labels in obs.label)
        self.sdo
            .observers
            .learn_cluster(self.chi, self.zeta, self.min_cluster_size, true);

        Ok(())
    }

    /// Berechnet den Outlier-Score für einen Datenpunkt (wie SDO)
    pub fn predict_outlier_score(&self, point: PyReadonlyArray2<f64>) -> PyResult<f64> {
        self.sdo.predict(point)
    }

    /// Berechnet das Cluster-Label für einen Datenpunkt
    pub fn predict(&self, point: PyReadonlyArray2<f64>) -> PyResult<i32> {
        if self.sdo.observers.is_empty() {
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

        // Finde die x nächsten Nachbarn unter den aktiven Observers
        let nearest_indices = self
            .sdo
            .observers
            .search_k_nearest_indices(&point_vec, self.sdo.x, true);

        // Zähle die Häufigkeit der Labels
        let mut label_counts: HashMap<i32, usize> = HashMap::new();
        for idx in nearest_indices {
            if let Some(obs) = self.sdo.observers.get(idx) {
                if let Some(label) = obs.label {
                    if label >= 0 {
                        *label_counts.entry(label).or_insert(0) += 1;
                    }
                }
            }
        }

        // Gib das häufigste Label zurück
        if let Some((&most_common_label, _)) = label_counts.iter().max_by_key(|(_, &count)| count) {
            Ok(most_common_label)
        } else {
            Ok(-1) // Kein Label gefunden (Outlier)
        }
    }

    /// Gibt die Anzahl der Cluster zurück
    pub fn n_clusters(&self) -> usize {
        // Stelle sicher, dass Clustering durchgeführt wurde
        let unique_labels: HashSet<i32> = self
            .sdo
            .observers
            .iter_observers(true)
            .filter_map(|obs| obs.label)
            .filter(|&l| l >= 0)
            .collect();
        unique_labels.len()
    }

    /// Gibt x zurück (Anzahl der nächsten Nachbarn)
    #[getter]
    pub fn x(&self) -> usize {
        self.sdo.x
    }

    /// Gibt die Labels der Observer zurück
    pub fn get_observer_labels(&self) -> Vec<i32> {
        // Stelle sicher, dass Clustering durchgeführt wurde
        self.sdo
            .observers
            .iter_observers(true)
            .map(|obs| obs.label.unwrap_or(-1))
            .collect()
    }

    /// Konvertiert active_observers zu NumPy-Array für Python
    pub fn get_active_observers(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        self.sdo.get_active_observers(py)
    }

    /// Calculate Mahalanobis distance uniformity score for a specific cluster
    /// Returns convexity score where lower values indicate more convex (uniform) distribution
    pub fn get_cluster_convexity_score(&self, cluster_label: i32) -> f64 {
        // Find all observers belonging to the specified cluster
        let cluster_observers: Vec<usize> = self
            .sdo
            .observers
            .iter_observers(true)
            .enumerate()
            .filter_map(|(_i, obs)| {
                if obs.label == Some(cluster_label) {
                    Some(obs.index)
                } else {
                    None
                }
            })
            .collect();

        if cluster_observers.is_empty() {
            return f64::INFINITY; // No observers for this cluster
        }

        // Calculate Mahalanobis score for cluster observers
        self.sdo
            .observers
            .mahalanobis_uniformity_score(Some(&cluster_observers))
    }

    /// Calculate convexity scores for all clusters
    /// Returns HashMap mapping cluster labels to their convexity scores
    pub fn get_all_cluster_convexity_scores(&self) -> HashMap<i32, f64> {
        let mut scores = HashMap::new();
        let cluster_labels: std::collections::HashSet<i32> = self
            .sdo
            .observers
            .iter_observers(true)
            .filter_map(|obs| obs.label)
            .collect();

        for &label in &cluster_labels {
            scores.insert(label, self.get_cluster_convexity_score(label));
        }

        scores
    }
}

impl Default for SDOclust {
    fn default() -> Self {
        Self::new(200, 5, 0.2, 4, 0.5, 2, "euclidean".to_string(), None)
    }
}
