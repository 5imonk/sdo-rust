use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::obs::{neighbor_index, NeighborInfo};
use crate::sdostream_impl::SDOstream;
use crate::utils::{data_to_matrix, label_score_results_to_py, times_to_vec_batch};

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
    #[pyo3(signature = (k, t_fading, t_sampling, x, rho = 0.1, chi_min = 1, chi_prop = 0.05, zeta = 0.6, min_cluster_size = 2, distance = "euclidean".to_string(), minkowski_p = None, dimension = None, data = None, time = None))]
    pub fn new(
        k: usize,
        t_fading: f64,
        t_sampling: f64,
        x: usize,
        rho: f64,
        chi_min: usize,
        chi_prop: f64,
        zeta: f64,
        min_cluster_size: usize,
        distance: String,
        minkowski_p: Option<f64>,
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

        if chi_min < 1 {
            return Err(PyErr::new::<PyValueError, _>("chi_min must be at least 1"));
        }

        if !(0.0..=1.0).contains(&zeta) {
            return Err(PyErr::new::<PyValueError, _>(
                "zeta must be between 0.0 and 1.0",
            ));
        }

        Ok(Self {
            sdostream: SDOstream::new(
                k,
                t_fading,
                t_sampling,
                x,
                rho,
                distance,
                minkowski_p,
                dimension,
                data,
                time,
            )?,
            chi: ((chi_min as f64).max(chi_prop * (1.0 - rho) * k as f64) as usize),
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

        Python::with_gil(|py| label_score_results_to_py(&results, rows, py))
    }

    /// Berechnet Cluster-Label und Outlier-Score für einen oder mehrere Datenpunkte (Batch-Verarbeitung).
    #[pyo3(signature = (points))]
    pub fn predict(&self, points: PyReadonlyArray2<f64>) -> PyResult<PyObject> {
        let (points_vec, rows) = data_to_matrix(points);
        let results = self.predict_impl(&points_vec, None);
        Python::with_gil(|py| label_score_results_to_py(&results, rows, py))
    }

    /// Gibt die Positionen der aktiven Observer als NumPy-Array zurück (Modell für Label-Vorhersage).
    pub fn get_active_observers(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        self.sdostream.get_sdo().get_active_observers(py)
    }

    /// Gibt die Cluster-Labels der aktiven Observer zurück (-1 = kein Label/Outlier).
    pub fn get_observer_labels(&self) -> Vec<i32> {
        let observers = &self.sdostream.get_sdo().observers;
        // Bestimme maximale Zeit aller Observer als current_time
        let max_time = observers.iter_observers(false)
            .map(|(_, _, _, time, _)| time)
            .fold(0.0, f64::max);
        
        observers
            .iter_observers(true)
            .map(|(idx, _, _, _, _)| observers.get_label(idx, max_time).map(|l| l as i32).unwrap_or(-1))
            .collect()
    }

    /// Gibt aktive Observer inkl. finaler Threshold-Radien für Visualisierung.
    /// Returns (points, labels, final_threshold_radii) mit final = zeta * local + (1-zeta) * global.
    pub fn get_active_observers_with_final_thresholds(
        &self,
        py: Python<'_>,
    ) -> PyResult<(Py<PyArray2<f64>>, Vec<i32>, Vec<f64>)> {
        let active_observers = self.sdostream.get_sdo().observers.get_observers(true);
        if active_observers.is_empty() {
            let empty = PyArray2::zeros_bound(py, (0, 0), false);
            return Ok((empty.unbind(), vec![], vec![]));
        }

        let observers = &self.sdostream.get_sdo().observers;
        let local_thresholds: Vec<f64> =
            active_observers.iter().map(|(idx, _, _, _, _)| observers.get_local_threshold(*idx).unwrap_or(f64::INFINITY)).collect();
        let global_threshold = observers.get_global_threshold();
        let zeta = self.zeta;
        let final_radii: Vec<f64> = local_thresholds
            .iter()
            .map(|h| zeta * h + (1.0 - zeta) * global_threshold)
            .collect();

        // Bestimme maximale Zeit aller Observer als current_time
        let max_time = observers.iter_observers(false)
            .map(|(_, _, _, time, _)| time)
            .fold(0.0, f64::max);
        
        let labels: Vec<i32> = active_observers
            .iter()
            .map(|(idx, _, _, _, _)| observers.get_label(*idx, max_time).map(|l| l as i32).unwrap_or(-1))
            .collect();

        let rows = active_observers.len();
        let cols = active_observers[0].1.len(); // data is second element
        let array = PyArray2::zeros_bound(py, (rows, cols), false);
        unsafe {
            let mut arr = array.as_array_mut();
            for (i, (_idx, data, _obs, _time, _age)) in active_observers.iter().enumerate() {
                for (j, &v) in data.iter().enumerate() {
                    arr[[i, j]] = v;
                }
            }
        }

        Ok((array.unbind(), labels, final_radii))
    }

    /// t_sampling des inneren SDOstream (für Tests/Reflection).
    pub fn get_t_sampling(&self) -> f64 {
        self.sdostream.t_sampling()
    }

    /// replacement_count des inneren SDOstream (für Tests/Verhalten).
    pub fn get_replacement_count(&self) -> usize {
        self.sdostream.replacement_count()
    }
}

impl SDOstreamclust {
    /// Verarbeitet einen einzelnen Datenpunkt (Rust-intern).
    pub fn learn_point(&mut self, point: &Vec<f64>, time: f64) -> (i32, f64) {
        // Schritt 1: Anzahl der Ersetzungen bestimmen
        let n_replacements = self.sdostream.n_replacements_impl(time);

        // Schritt 2: Predict (inklusive Labels)
        let (median, active_neighbors, all_neighbors_opt) =
            self.sdostream
                .get_sdo()
                .predict_point(point, Some(true), Some(n_replacements.min(1)));
        let nearest_active_indices: Vec<usize> = active_neighbors.iter().map(neighbor_index).collect();
        let predicted_label = self.compute_label(&nearest_active_indices);

        // Schritt 3: Sampling
        let replacement_pair = if n_replacements > 0 {
            let pair = self.sdostream.sample_point(point, time, None);
            // Aktualisiere pending_replacements und last_replacement_time
            if pair.is_some() {
                self.sdostream
                    .set_pending_replacements(n_replacements.saturating_sub(1));
                self.sdostream.set_last_replacement_time(time);
            }
            pair
        } else {
            None
        };

        // Schritt 4: Fitting
        let final_all_neighbors_opt =
            self.sdostream
                .fit_point(all_neighbors_opt, replacement_pair, point, time);

        // Schritt 5: Updating
        let fading = self.sdostream.get_fading();
        if let Some(ref neighbors) = final_all_neighbors_opt {
            let processed = self.sdostream.get_sdo_mut().observers.update(
                vec![neighbors.clone()],
                vec![time],
                fading,
            );
            self.sdostream.increment_data_points_processed(processed);
        }

        // Schritt 6: learn_clustering (einmal, mit batch_age)
        // Für einzelnen Punkt: batch_age = fading^0 = 1.0
        let batch_age = 1.0;
        self.sdostream
            .get_sdo_mut()
            .observers
            .learn_clustering_time(
                self.chi,
                self.zeta,
                self.min_cluster_size,
                fading,
                time, // batch_start_time = time für einzelnen Punkt
                time, // batch_end_time = time für einzelnen Punkt
                batch_age,
            );

        (predicted_label, median)
    }

    /// Verarbeitet einen Batch (Rust-intern).
    pub fn learn_impl(&mut self, points: &[Vec<f64>], times: &[f64]) -> Vec<(i32, f64)> {
        assert_eq!(points.len(), times.len());

        if points.is_empty() {
            return Vec::new();
        }

        let batch_size = points.len();
        let reference_time = *times.last().unwrap();

        // Schritt 1: Set n_replacements
        let n_replacements_total = self.sdostream.n_replacements_impl(reference_time);
        let n_replacements = n_replacements_total.min(batch_size);
        if n_replacements_total > batch_size {
            self.sdostream
                .set_pending_replacements(n_replacements_total - batch_size);
        } else {
            self.sdostream.set_pending_replacements(0);
        }

        // Schritt 2: Predict für alle Punkte (inklusive Labels)
        let predict_results: Vec<(f64, Vec<NeighborInfo>, Option<Vec<NeighborInfo>>)> = points
            .iter()
            .map(|point| {
                self.sdostream
                    .get_sdo()
                    .predict_point(point, Some(true), Some(n_replacements))
            })
            .collect();

        // Berechne Labels für alle Punkte
        let predicted_labels: Vec<i32> = predict_results
            .iter()
            .map(|(_, active_neighbors, _)| {
                let indices: Vec<usize> = active_neighbors.iter().map(neighbor_index).collect();
                self.compute_label(&indices)
            })
            .collect();

        // Schritt 3: Sampling
        let replacement_pairs = if n_replacements > 0 {
            let pairs = self.sdostream.sample_impl(points, times, n_replacements);
            self.sdostream.set_last_replacement_time(reference_time);
            pairs
        } else {
            vec![None; batch_size]
        };

        // Schritt 4: fit_impl
        let final_all_neighbors_batch = self.sdostream.fit_impl(
            predict_results
                .iter()
                .map(|(_, _, opt)| opt.as_ref())
                .collect(),
            &replacement_pairs,
            points,
        );

        // Schritt 5: Update
        let all_neighbors_for_update: Vec<Vec<NeighborInfo>> = final_all_neighbors_batch
            .iter()
            .filter_map(|opt| opt.as_ref().map(|v| v.clone()))
            .collect();
        let observation_times_for_update: Vec<f64> = final_all_neighbors_batch
            .iter()
            .enumerate()
            .filter_map(|(i, opt)| if opt.is_some() { Some(times[i]) } else { None })
            .collect();

        let fading = self.sdostream.get_fading();
        if !all_neighbors_for_update.is_empty() {
            let processed = self.sdostream.get_sdo_mut().observers.update(
                all_neighbors_for_update,
                observation_times_for_update,
                fading,
            );
            self.sdostream.increment_data_points_processed(processed);
        }

        // Schritt 6: learn_clustering (einmal für den Batch, mit batch_age)
        let reference_start_time = *times.first().unwrap();
        let batch_age = times
            .iter()
            .map(|&t| fading.powf(reference_time - t))
            .sum::<f64>()
            * fading.powf(reference_start_time - reference_time);

        self.sdostream
            .get_sdo_mut()
            .observers
            .learn_clustering_time(
                self.chi,
                self.zeta,
                self.min_cluster_size,
                fading,
                reference_start_time, // batch_start_time
                reference_time,       // batch_end_time
                batch_age,
            );

        // Erstelle Ergebnisse: (label, median)
        predict_results
            .into_iter()
            .zip(predicted_labels.into_iter())
            .map(|((median, _, _), label)| (label, median))
            .collect()
    }

    /// Vorhersage für einen einzelnen Punkt (Rust-intern).
    pub fn predict_point(&self, point: &[f64], learn: Option<bool>) -> (i32, f64) {
        let point_vec: Vec<f64> = point.to_vec();
        let (median, active_neighbors, _) = self.sdostream.predict_point(&point_vec, learn, None);
        let nearest_active_indices: Vec<usize> = active_neighbors.iter().map(neighbor_index).collect();
        let predicted_label = self.compute_label(&nearest_active_indices);
        (predicted_label, median)
    }

    /// Batch-Vorhersage (Rust-intern).
    pub fn predict_impl(&self, points: &[Vec<f64>], learn: Option<bool>) -> Vec<(i32, f64)> {
        let batch = self.sdostream.predict_impl(points, learn, None);
        batch
            .into_iter()
            .map(|(median, active_neighbors, _)| {
                let indices: Vec<usize> = active_neighbors.iter().map(neighbor_index).collect();
                let label = self.compute_label(&indices);
                (label, median)
            })
            .collect()
    }
}

impl SDOstreamclust {
    pub fn compute_label(&self, indices: &[usize]) -> i32 {
        if indices.is_empty() {
            panic!("No active observers found during prediction!");
        }

        // Bestimme maximale Zeit aller Observer als current_time
        let observers = &self.sdostream.get_sdo().observers;
        let max_time = observers.iter_observers(false)
            .map(|(_, _, _, time, _)| time)
            .fold(0.0, f64::max);

        // Berechne normalisierte Cluster-Scores der x-nächsten Observer
        let label_scores = observers
            .get_normalized_cluster_scores(indices, max_time);

        // Finde Label mit maximalem Score (konvertiere zu i32 für Python-API)
        let predicted_label = label_scores
            .iter()
            .max_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&label, _)| label as i32)
            .unwrap_or(-1); // -1 wenn keine Beobachtungen
        predicted_label
    }

    /// Gibt eine mutable Referenz auf das interne SDOstream-Objekt zurück
    pub fn get_sdostream_mut(&mut self) -> &mut SDOstream {
        &mut self.sdostream
    }
}

impl SDOstreamclust {
    /// For benchmarks only: create SDOstreamclust with dimension (no Python).
    pub fn new_for_benchmark(
        k: usize,
        t_fading: f64,
        t_sampling: f64,
        x: usize,
        rho: f64,
        chi_min: usize,
        chi_prop: f64,
        zeta: f64,
        min_cluster_size: usize,
        dimension: usize,
    ) -> Self {
        Self {
            sdostream: SDOstream::new_for_benchmark(k, t_fading, t_sampling, x, rho, dimension),
            chi: ((chi_min as f64).max(chi_prop * (1.0 - rho) * k as f64) as usize),
            zeta,
            min_cluster_size,
        }
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
