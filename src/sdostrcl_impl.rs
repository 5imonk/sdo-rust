use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashSet;

use crate::sdostream_impl::SDOstream;
use crate::utils::{point_to_vec, time_to_f64};

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
        if chi_prop < 0.0 || chi_prop > 1.0 {
            return Err(PyErr::new::<PyValueError, _>(
                "chi_prop must be between 0.0 and 1.0",
            ));
        }

        if zeta < 0.0 || zeta > 1.0 {
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

    /// Verarbeitet einen einzelnen Datenpunkt aus dem Stream (Algorithmus 3.2)
    #[pyo3(signature = (point, *, time = None))]
    pub fn learn(
        &mut self,
        point: PyReadonlyArray2<f64>,
        time: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<(i32, f64)> {
        // Extract point vector
        let point_vec = point_to_vec(point);

        // Determine current time based on initialization strategy
        let current_time = time_to_f64(
            time,
            self.sdostream.get_use_explicit_time(),
            self.sdostream.get_data_points_processed(),
        )?;

        // Call internal learn implementation
        let (predicted_label, outlier_score) = self.learn_impl(&point_vec, current_time);

        Ok((predicted_label, outlier_score))
    }

    /// Berechnet das Cluster-Label für einen Datenpunkt (Gleichung 3.4)
    pub fn predict(&self, point: PyReadonlyArray2<f64>) -> PyResult<(i32, f64)> {
        let point_vec = point_to_vec(point);

        let (predicted_label, outlier_score) = self.predict_impl(&point_vec);

        Ok((predicted_label, outlier_score))
    }

    /// Gibt x zurück (Anzahl der nächsten Nachbarn)
    #[getter]
    pub fn x(&self) -> usize {
        self.sdostream.x()
    }

    /// Gibt aktuelle Cluster-Informationen zurück
    #[getter]
    pub fn get_clusters(&mut self) -> PyResult<Vec<Vec<usize>>> {
        // Berechne aktuelle Cluster
        let cluster_map = self.sdostream.get_sdo_mut().observers.learn_cluster(
            self.chi,
            self.zeta,
            self.min_cluster_size,
            true, // read-only mode
        );

        // Convert to Vec<Vec<usize>>
        let clusters: Vec<Vec<usize>> = cluster_map
            .into_values()
            .map(|set| set.into_iter().collect())
            .collect();

        Ok(clusters)
    }

    /// Gibt die Anzahl der verarbeiteten Datenpunkte zurück
    #[getter]
    pub fn data_points_processed(&self) -> usize {
        self.sdostream.get_data_points_processed()
    }

    /// Gibt Observer-Informationen für einen bestimmten Index zurück
    #[pyo3(signature = (index))]
    pub fn get_observer_info(
        &self,
        _py: Python,
        index: usize,
    ) -> PyResult<(Vec<f64>, f64, f64, f64, bool, Option<i32>, Vec<f64>)> {
        if let Some(observer) = self.sdostream.get_sdo().observers.get(index) {
            let is_active = self.sdostream.get_sdo().observers.is_active(index);
            Ok((
                observer.data.clone(),
                observer.observations,
                observer.age,
                observer.time,
                is_active,
                observer.label,
                observer.cluster_observations.clone(),
            ))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Observer with index {} not found",
                index
            )))
        }
    }

    /// Gibt alle Observer-Cluster-Labels zurück
    #[getter]
    pub fn get_cluster_labels(&self, _py: Python) -> PyResult<Vec<Option<i32>>> {
        let labels: Vec<Option<i32>> = self
            .sdostream
            .get_sdo()
            .observers
            .iter_observers(false)
            .map(|obs| obs.label)
            .collect();
        Ok(labels)
    }

    /// Gibt alle Cluster-Beobachtungen zurück
    #[getter]
    pub fn get_cluster_observations(&self, _py: Python) -> PyResult<Vec<Vec<f64>>> {
        let cluster_obs: Vec<Vec<f64>> = self
            .sdostream
            .get_sdo()
            .observers
            .iter_observers(false)
            .map(|obs| obs.cluster_observations.clone())
            .collect();
        Ok(cluster_obs)
    }
}

impl SDOstreamclust {
    /// Interner Zugriff auf mutable SDOstream
    pub fn learn_impl(&mut self, point: &Vec<f64>, time: f64) -> (i32, f64) {
        // Verwende SDOstream für Modell-Erstellung (Sample, Observe, Clean)
        let (median, nearest_active_indices) = self.sdostream.learn_impl(&point, time);

        // Schritt 2: Cluster (Algorithmus 3.3)
        // Get mutable reference to observers
        // This requires that get_sdo_mut() returns &mut to the SDO object
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

        // Schritt 4 & 5: Update Cluster-Beobachtungen
        let fading = self.sdostream.get_fading();
        self.sdostream
            .get_sdo_mut()
            .observers
            .update_cluster_observations_with_fading_and_clusters(
                fading,
                time,
                &clusters,
                &cluster_labels,
            );

        let predicted_label = self.compute_label(&nearest_active_indices);

        (predicted_label, median)
    }

    pub fn predict_impl(&self, point: &Vec<f64>) -> (i32, f64) {
        let (median, nearest_active_indices) = self.sdostream.predict_impl(point);

        let predicted_label = self.compute_label(&nearest_active_indices);

        (predicted_label, median)
    }
}

impl SDOstreamclust {
    pub fn compute_label(&self, indices: &Vec<usize>) -> i32 {
        if indices.is_empty() {
            panic!("No active observers found during prediction!");
        }

        // Berechne normalisierte Cluster-Scores der x-nächsten Observer
        let label_scores = self
            .sdostream
            .get_sdo()
            .observers
            .get_normalized_cluster_scores(&indices);

        // Finde Label mit maximalem Score
        let predicted_label = label_scores
            .iter()
            .max_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&label, _)| label)
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
