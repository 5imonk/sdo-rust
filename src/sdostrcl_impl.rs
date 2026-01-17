use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use std::collections::HashSet;

use crate::sdostream_impl::SDOstream;

/// SDOstreamclust Algorithm - Streaming-Version von SDOclust
/// Baut auf SDOstream auf und fügt Clustering-Logik hinzu
#[pyclass]
#[allow(clippy::upper_case_acronyms)]
pub struct SDOstreamclust {
    sdostream: SDOstream, // Basis SDOstream-Implementierung
    chi_min: usize,       // Minimum chi value
    chi_prop: f64,        // Proportion of k for chi calculation
    k: usize,             // Anzahl der Observer (für chi Berechnung)
    zeta: f64,
    min_cluster_size: usize,

    // Cluster-specific caching for predict() optimization
    cached_cluster_scores: Option<std::collections::HashMap<i32, f64>>, // Cached cluster scores
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
        // Check if initialization is needed before moving values
        let needs_init = data.is_some() || dimension.is_some() || time.is_some();

        let mut instance = Self {
            sdostream: SDOstream::new(
                k,
                x,
                t_fading,
                t_sampling,
                distance,
                minkowski_p,
                rho,
                dimension,
                data.clone(),
                time.clone(),
            )?,
            chi_min,
            chi_prop,
            k,
            zeta,
            min_cluster_size,

            // Initialize cluster cache
            cached_cluster_scores: None,
        };

        // Initialisiere mit Parametern (wenn Daten/Dimension/Zeit angegeben)
        if needs_init {
            // Setze last_cluster_time basierend auf Initialisierung
            if let Some(time_array) = &time {
                let time_slice = time_array.as_array();
                if time_slice.len() == 1 {
                    instance.sdostream.get_sdo_mut().observers.last_cluster_time = time_slice[[0]];
                }
            }

            instance.sdostream.initialize(dimension, data, time)?;

            // Invalidiere Cluster-Cache nach Initialisierung
            instance.cached_cluster_scores = None;
        }

        Ok(instance)
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
        // Berechne chi: chi = max(chi_min, chi_prop * k)
        let chi = (self.chi_min as f64).max(self.chi_prop * (self.k as f64)) as usize;
        let cluster_map = self.sdostream.get_sdo_mut().observers.learn_cluster(
            chi,
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

        // Cache cluster-spezifische predict Informationen
        if let Some(ref cached_indices) = self.sdostream.get_cached_nearest_active_indices() {
            let label_scores = self
                .sdostream
                .get_sdo()
                .observers
                .get_normalized_cluster_scores(cached_indices);

            self.cached_cluster_scores = Some(label_scores);
        }

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

        // Prüfe ob wir gecachte Ergebnisse verwenden können
        if let (Some(ref cached_point), Some(ref cached_scores)) = (
            self.sdostream.get_cached_point(),
            &self.cached_cluster_scores,
        ) {
            // Verifiziere dass Cache für den gleichen Punkt ist (mit Toleranz für Fließkomma)
            if self.sdostream.get_cached_nearest_active_indices().is_some()
                && crate::sdostream_impl::SDOstream::points_match(&point_vec, cached_point)
                && self.sdostream.is_cache_valid()
            {
                // Verwende gecachte Cluster-Scores - dies ist der "kostenlose" predict!
                let predicted_label = cached_scores
                    .iter()
                    .max_by(|(_, &a), (_, &b)| {
                        a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(&label, _)| label)
                    .unwrap_or(-1); // -1 wenn keine Beobachtungen

                return Ok(predicted_label);
            }
        }

        // Fallback: neu berechnen wenn Cache ungültig
        // Finde x-nächste aktive Observer (using optimized unified search mit aktiven info)
        let x = self.sdostream.x();
        let (active_neighbors, _, _) = self
            .sdostream
            .get_sdo()
            .observers
            .search_neighbors_unified(&point_vec, x, true);
        let nearest_indices: Vec<usize> = active_neighbors.iter().map(|n| n.index).collect();

        // Berechne normalisierte Cluster-Scores der x-nächsten Observer
        let label_scores = self
            .sdostream
            .get_sdo()
            .observers
            .get_normalized_cluster_scores(&nearest_indices);

        // Finde Label mit maximalem Score: argmax_c (Σ_ω l̂_ω^c)
        // wobei l̂_ω normalisiert ist
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

    /// Berechnet den Outlier-Score für einen Datenpunkt (delegiert an SDOstream)
    pub fn predict_outlier_score(&self, point: PyReadonlyArray2<f64>) -> PyResult<f64> {
        self.sdostream.predict(point)
    }
}

impl Default for SDOstreamclust {
    fn default() -> Self {
        Self::new(
            200,
            5,
            100.0,
            None, // t_sampling = t_fading
            1,    // chi_min
            0.05, // chi_prop
            0.6,
            2,
            "euclidean".to_string(),
            None,
            0.1,
            None,
            None,
            None,
        )
        .unwrap()
    }
}
