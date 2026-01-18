use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::f64;

use crate::sdo_impl::SDO;
use crate::utils::{data_to_matrix, point_to_vec};

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
        let data_vec = data_to_matrix(data);

        // Anzahl der Datenpunkte
        let rows = data_vec.len();

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

    /// Berechnet das Cluster-Label für einen Datenpunkt
    pub fn predict(
        &self,
        point: PyReadonlyArray2<f64>,
        outlier_score_flag: bool,
    ) -> PyResult<(i32, f64)> {
        if self.sdo.observers.is_empty() {
            return Ok((-1, f64::NAN)); // Kein Label (Outlier)
        }

        // Konvertiere Punkt zu Vec<f64>
        let point_vec = point_to_vec(point);

        // Rufe die interne Vorhersagemethode auf
        let label = self.predict_impl(&point_vec);

        let (outlier_score, _nearest_active_indices) = match outlier_score_flag {
            true => self.sdo.predict_impl(&point_vec),
            false => (f64::NAN, Vec::new()),
        };
        Ok((label, outlier_score))
    }

    /// Konvertiert active_observers zu NumPy-Array für Python
    pub fn get_active_observers(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        self.sdo.get_active_observers(py)
    }
    /// Gibt die Anzahl der Cluster zurück
    pub fn n_clusters(&self) -> usize {
        // Stelle sicher, dass Clustering durchgeführt wurde
        // Gehe durch alle aktiven Observer und sammle eindeutige Labels
        let unique_labels: HashSet<i32> = self
            .sdo
            .observers
            .iter_observers(true)
            .filter_map(|obs| obs.label)
            .filter(|&label| label >= 0) // Filter out -1 labels (outliers)
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

        // Führe Clustering durch (schreibe Labels in obs.label)
        self.sdo
            .observers
            .learn_cluster(self.chi, self.zeta, self.min_cluster_size, true);
    }

    pub fn predict_impl(&self, point: &Vec<f64>) -> i32 {
        if self.sdo.observers.is_empty() {
            return -1; // Kein Label (Outlier)
        }

        // Finde die x nächsten Nachbarn unter den aktiven Observers (using optimized unified search)
        let (active_neighbors, _) = self
            .sdo
            .observers
            .search_neighbors_unified(point, self.sdo.x, true);
        let nearest_indices: Vec<usize> = active_neighbors.iter().map(|n| n.index).collect();

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

        // Gib das häufigste Label zurück (optimiert mit vorgefilterten aktiven Observers)
        if let Some((&most_common_label, _)) = label_counts.iter().max_by_key(|(_, &count)| count) {
            most_common_label
        } else {
            -1 // Kein Label gefunden (Outlier)
        }
    }

    /// Gibt die Labels der Observer zurück (optimiert mit aktiven Observer-Info)
    pub fn get_observer_labels(&self) -> Vec<i32> {
        // Stelle sicher, dass Clustering durchgeführt wurde
        // Gehe einfach durch alle aktiven Observer
        self.sdo
            .observers
            .iter_observers(true)
            .map(|obs| obs.label.unwrap_or(-1))
            .collect()
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
