use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

use crate::sdostream_impl::{SDOstream, SDOstreamParams};

/// Parameter-Struktur für SDOstreamclust
/// Erweitert SDOstreamParams um Clustering-Parameter
#[pyclass]
#[derive(Clone)]
pub struct SDOstreamclustParams {
    #[pyo3(get, set)]
    pub k: usize,
    #[pyo3(get, set)]
    pub x: usize,
    #[pyo3(get, set)]
    pub t_fading: f64,
    #[pyo3(get, set)]
    pub t_sampling: f64,
    #[pyo3(get, set)]
    pub distance: String,
    #[pyo3(get, set)]
    pub minkowski_p: Option<f64>,
    #[pyo3(get, set)]
    pub rho: f64,
    #[pyo3(get, set)]
    pub chi: usize, // Anzahl der nächsten Observer für lokale Thresholds
    #[pyo3(get, set)]
    pub zeta: f64, // Mixing-Parameter für globale/lokale Thresholds
    #[pyo3(get, set)]
    pub min_cluster_size: usize, // Minimale Clustergröße
}

#[pymethods]
#[allow(clippy::too_many_arguments)]
impl SDOstreamclustParams {
    #[new]
    #[pyo3(signature = (k, x, t_fading, t_sampling, chi = 4, zeta = 0.5, min_cluster_size = 2, distance = "euclidean".to_string(), minkowski_p = None, rho = 0.1))]
    pub fn new(
        k: usize,
        x: usize,
        t_fading: f64,
        t_sampling: f64,
        chi: usize,
        zeta: f64,
        min_cluster_size: usize,
        distance: String,
        minkowski_p: Option<f64>,
        rho: f64,
    ) -> Self {
        Self {
            k,
            x,
            t_fading,
            t_sampling,
            distance,
            minkowski_p,
            rho,
            chi,
            zeta,
            min_cluster_size,
        }
    }

    /// Konvertiert zu SDOstreamParams
    fn to_sdostream_params(&self) -> SDOstreamParams {
        SDOstreamParams {
            k: self.k,
            x: self.x,
            t_fading: self.t_fading,
            t_sampling: self.t_sampling,
            distance: self.distance.clone(),
            minkowski_p: self.minkowski_p,
            rho: self.rho,
        }
    }
}

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
impl SDOstreamclust {
    #[new]
    pub fn new() -> Self {
        Self {
            sdostream: SDOstream::new(),
            chi: 4,
            zeta: 0.5,
            min_cluster_size: 2,
        }
    }

    /// Initialisiert das Modell mit Parametern
    #[pyo3(signature = (params, dimension = None, *, data = None, time = None))]
    pub fn initialize(
        &mut self,
        params: &SDOstreamclustParams,
        dimension: Option<usize>,
        data: Option<PyReadonlyArray2<f64>>,
        time: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<()> {
        // Setze Clustering-spezifische Parameter
        self.chi = params.chi;
        self.zeta = params.zeta;
        self.min_cluster_size = params.min_cluster_size;

        // Setze last_cluster_time basierend auf Initialisierung (vor dem move)
        if let Some(time_array) = &time {
            let time_slice = time_array.as_array();
            if time_slice.len() == 1 {
                self.sdostream.get_sdo_mut().observers.last_cluster_time = time_slice[[0]];
            }
        }

        // Initialisiere SDOstream mit Parametern (delegiert an SDOstream.initialize)
        let stream_params = params.to_sdostream_params();
        self.sdostream
            .initialize(&stream_params, dimension, data, time)?;

        Ok(())
    }

    /// Verarbeitet einen einzelnen Datenpunkt aus dem Stream (Algorithmus 3.2)
    #[pyo3(signature = (point, *, time = None))]
    pub fn learn(
        &mut self,
        point: PyReadonlyArray2<f64>,
        time: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<()> {
        // Bestimme aktuelle Zeit vor dem Aufruf von learn()
        let current_time = if self.sdostream.get_use_explicit_time() {
            if let Some(time_array) = &time {
                let time_slice = time_array.as_array();
                if time_slice.len() != 1 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Zeit muss ein 1D-Array mit einem Wert sein",
                    ));
                }
                time_slice[[0]]
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "time-Parameter ist erforderlich",
                ));
            }
        } else {
            (self.sdostream.get_data_points_processed() + 1) as f64
        };

        // Schritt 1: SDOstream (Algorithmus 11)
        self.sdostream.learn(point, time)?;

        // Schritt 2: Cluster (Algorithmus 3.3) - verwende learn_cluster direkt
        let cluster_map = self.sdostream.get_sdo_mut().observers.learn_cluster(
            self.chi,
            self.zeta,
            self.min_cluster_size,
            false,
        );

        // Konvertiere HashMap zu Vec<HashSet<usize>> für label_clusters
        let clusters: Vec<HashSet<usize>> = cluster_map.into_values().collect();

        // Schritt 3: Label (Algorithmus 3.5)
        let cluster_labels = self.sdostream.get_sdo().observers.label_clusters(&clusters);

        // Schritt 4 & 5: Update Cluster-Beobachtungen (Fading und/oder Cluster-Updates)
        let fading = self.sdostream.get_fading();
        self.sdostream
            .get_sdo_mut()
            .observers
            .update_cluster_observations_with_fading_and_clusters(
                fading,
                current_time,
                &clusters,
                &cluster_labels,
            );

        Ok(())
    }

    /// Berechnet das Cluster-Label für einen Datenpunkt (Gleichung 3.4)
    pub fn predict(&self, point: PyReadonlyArray2<f64>) -> PyResult<i32> {
        let point_slice = point.as_array();
        if point_slice.nrows() != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Punkt muss ein 1D-Array oder 2D-Array mit einer Zeile sein",
            ));
        }

        let point_vec: Vec<f64> = (0..point_slice.ncols())
            .map(|j| point_slice[[0, j]])
            .collect();

        // Finde x-nächste aktive Observer
        let x = self.sdostream.x();
        let nearest_indices = self
            .sdostream
            .get_sdo()
            .observers
            .search_k_nearest_indices(&point_vec, x, true);

        // Summiere Lω Vektoren der x-nächsten Observer
        let mut label_scores: HashMap<i32, f64> = HashMap::new();

        for &obs_idx in &nearest_indices {
            if let Some(observer) = self.sdostream.get_sdo().observers.get(obs_idx) {
                let l_omega = &observer.cluster_observations;
                for (label_idx, &value) in l_omega.iter().enumerate() {
                    let label = label_idx as i32;
                    *label_scores.entry(label).or_insert(0.0) += value;
                }
            }
        }

        // Finde Label mit maximalem Score: argmax_c Σ_ω l_ω^c
        let predicted_label = label_scores
            .iter()
            .max_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&label, _)| label)
            .unwrap_or(-1); // -1 wenn keine Beobachtungen

        Ok(predicted_label)
    }

    /// Gibt x zurück (Anzahl der nächsten Nachbarn)
    #[getter]
    pub fn x(&self) -> usize {
        self.sdostream.x()
    }
}

impl Default for SDOstreamclust {
    fn default() -> Self {
        Self::new()
    }
}
