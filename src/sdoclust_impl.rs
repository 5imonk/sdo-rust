use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::f64;

use crate::sdo_impl::{SDOParams, SDO};

/// Parameter-Struktur für SDOclust
#[pyclass]
#[derive(Clone)]
pub struct SDOclustParams {
    #[pyo3(get, set)]
    pub k: usize,
    #[pyo3(get, set)]
    pub x: usize,
    #[pyo3(get, set)]
    pub rho: f64,
    #[pyo3(get, set)]
    pub chi: usize,
    #[pyo3(get, set)]
    pub zeta: f64,
    #[pyo3(get, set)]
    pub min_cluster_size: usize,
    #[pyo3(get, set)]
    pub distance: String, // "euclidean", "manhattan", "chebyshev", "minkowski"
    #[pyo3(get, set)]
    pub minkowski_p: Option<f64>, // Für Minkowski-Distanz
}

#[pymethods]
#[allow(clippy::too_many_arguments)]
impl SDOclustParams {
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
            k,
            x,
            rho,
            chi,
            zeta,
            min_cluster_size,
            distance,
            minkowski_p,
        }
    }
}

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
impl SDOclust {
    #[new]
    pub fn new() -> Self {
        Self {
            sdo: SDO::new(),
            chi: 4,
            zeta: 0.5,
            min_cluster_size: 2,
            k: 100,
        }
    }

    /// Initialisiert das Modell mit Parametern und einem leeren ObserverSet
    pub fn initialize(&mut self, params: &SDOclustParams) -> PyResult<()> {
        self.chi = params.chi;
        self.zeta = params.zeta;
        self.min_cluster_size = params.min_cluster_size;
        self.k = params.k;

        // Initialisiere SDO mit Parametern
        let sdo_params = SDOParams::new(
            params.k,
            params.x,
            params.rho,
            params.distance.clone(),
            params.minkowski_p,
        );
        self.sdo.initialize(&sdo_params)?;
        Ok(())
    }

    /// Lernt das Modell aus den Daten und führt Clustering durch
    pub fn learn(&mut self, data: PyReadonlyArray2<f64>) -> PyResult<()> {
        let data_slice = data.as_array();
        let rows = data_slice.nrows();

        if rows == 0 || self.k == 0 {
            return Ok(());
        }

        // Verwende SDO für Modell-Erstellung (Sample, Observe, Clean)
        self.sdo.learn(data)?;

        // Führe Clustering durch
        self.sdo
            .observers
            .learn_cluster(self.chi, self.zeta, self.min_cluster_size);

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
        let x = self.sdo.get_x_internal();
        let nearest_indices = self
            .sdo
            .observers
            .search_k_nearest_indices(&point_vec, x, true);

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

    /// Gibt x (Anzahl der Nachbarn) zurück
    #[getter]
    pub fn x(&self) -> usize {
        self.sdo.get_x_internal()
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
}

impl Default for SDOclust {
    fn default() -> Self {
        Self::new()
    }
}
