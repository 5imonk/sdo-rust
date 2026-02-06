use core::panic;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use std::collections::HashMap;
use std::f64;

use crate::obs::{NeighborInfo, Observer};
use crate::sdo_impl::SDO;
use crate::utils::{
    data_to_matrix, sample_random_matrix_uniform_unit, scores_single_or_list_to_py, time_to_f64,
    times_to_vec_batch,
};

impl SDOstream {
    /// Berechnet den Fading-Parameter f = exp(-T_fading^-1)
    fn get_fading_static(t_fading: f64) -> f64 {
        (-1.0 / t_fading).exp()
    }
}

/// SDOstream Algorithm - Streaming-Version von SDO
/// Baut auf SDO auf und fügt nur Streaming-spezifische Funktionalität hinzu
#[pyclass]
#[allow(clippy::upper_case_acronyms)]
pub struct SDOstream {
    sdo: SDO,                     // Basis SDO-Implementierung
    fading: f64,                  // f = exp(-T^-1)
    sampling_rate: f64, // Sampling-Rate (durchschnittliches Intervall zwischen Ersetzungen)
    data_points_processed: usize, // Zähler für Sampling
    use_explicit_time: bool, // Wenn true, erwartet learn() time-Parameter; sonst auto-increment
    last_replacement_time: f64, // Zeit der letzten Prüfung/Ersetzung (für Lazy Replacement)
    pending_replacements: usize, // Anzahl der ausstehenden Ersetzungen (wenn num_replacements > 1)
    replacement_count: usize, // Anzahl durchgeführter Ersetzungen (für Tests / Beobachtung)
    next_observer_index: usize,   // Nächster freier Index für neu eingefügte Observer (eindeutig)
}

#[pymethods]
#[allow(clippy::too_many_arguments)]
impl SDOstream {
    #[new]
    #[pyo3(signature = (k, t_fading, t_sampling, x, rho = 0.1, distance = "euclidean".to_string(), minkowski_p = None, dimension = None, data = None, time = None))]
    pub fn new(
        k: usize,
        t_fading: f64,
        t_sampling: f64,
        x: usize,
        rho: f64,
        distance: String,
        minkowski_p: Option<f64>,
        dimension: Option<usize>,
        data: Option<PyReadonlyArray2<f64>>,
        time: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Self> {
        let mut instance = Self {
            sdo: SDO::new(k, x, rho, distance, minkowski_p),
            fading: Self::get_fading_static(t_fading),
            sampling_rate: t_sampling / (k as f64),
            data_points_processed: 0,
            use_explicit_time: time.is_some(), // Default: auto-increment
            last_replacement_time: 0.0,        // Startzeit für Lazy Replacement
            pending_replacements: 0,           // Keine ausstehenden Ersetzungen
            replacement_count: 0,
            next_observer_index: 0,
        };

        instance.initialize(dimension, data, time)?;

        Ok(instance)
    }

    /// Initialisiert das Modell mit gegebenen Daten oder Dimension
    /// Wenn data gegeben, wird dieses verwendet; sonst werden k zufällige Punkte generiert
    #[pyo3(signature = (*, dimension = None, data = None, time = None))]
    pub fn initialize(
        &mut self,
        dimension: Option<usize>,
        data: Option<PyReadonlyArray2<f64>>,
        time: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<()> {
        // Prüfe, ob sowohl data als auch dimension gegeben sind (Fehler)
        if data.is_some() && dimension.is_some() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Entweder 'data' oder 'dimension' muss angegeben werden, nicht beide",
            ));
        }

        if !data.is_some() && !dimension.is_some() {
            // Kein data und keine dimension → leeres Modell initialisieren
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Entweder 'data' oder 'dimension' muss angegeben werden",
            ));
        }

        let start_time = time_to_f64(time, self.use_explicit_time, 0)?;
        let data_vec = match data {
            Some(data_array) => {
                let (matrix, _rows) = data_to_matrix(data_array);
                Some(matrix)
            }
            None => None,
        };

        self.initialize_impl(dimension, data_vec.as_ref(), start_time);

        Ok(())
    }

    /// Verarbeitet einen oder mehrere Datenpunkte aus dem Stream (Batch-Verarbeitung).
    /// Ein einzelner Punkt wird als Batch der Größe 1 behandelt.
    #[pyo3(signature = (points, *, time = None))]
    pub fn learn(
        &mut self,
        points: PyReadonlyArray2<f64>,
        time: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<PyObject> {
        // Convert points to Vec<Vec<f64>>
        let (points_vec, rows) = data_to_matrix(points);

        // Bestimme Zeiten für alle Punkte
        let times_vec = times_to_vec_batch(
            time,
            rows,
            self.use_explicit_time,
            self.data_points_processed,
        )?;

        let results = self.learn_impl(&points_vec, &times_vec);
        let scores: Vec<f64> = results.iter().map(|(median, _, _)| *median).collect();

        // Wenn nur ein Punkt: Rückgabe als einzelner Wert, sonst als Liste
        Python::with_gil(|py| scores_single_or_list_to_py(&scores, rows, py))
    }

    /// Berechnet den Outlier-Score für einen oder mehrere Datenpunkte (Batch-Verarbeitung).
    /// Ein einzelner Punkt wird als Batch der Größe 1 behandelt.
    #[pyo3(signature = (points))]
    pub fn predict(&self, points: PyReadonlyArray2<f64>) -> PyResult<PyObject> {
        let (points_vec, rows) = data_to_matrix(points);

        let results = self.predict_impl(&points_vec, Some(false), None);
        let scores: Vec<f64> = results.iter().map(|(median, _, _)| *median).collect();

        // Wenn nur ein Punkt: Rückgabe als einzelner Wert, sonst als Liste
        Python::with_gil(|py| scores_single_or_list_to_py(&scores, rows, py))
    }

    /// Gibt x zurück (Anzahl der nächsten Nachbarn)
    #[getter]
    pub fn x(&self) -> usize {
        self.sdo.x
    }

    /// Gibt k zurück (Anzahl der Observer)
    #[getter]
    pub fn k(&self) -> usize {
        self.sdo.get_k()
    }

    // gib rho zurück
    #[getter]
    pub fn rho(&self) -> f64 {
        self.sdo.get_rho()
    }

    /// Gibt t_sampling zurück
    #[getter]
    pub fn t_sampling(&self) -> f64 {
        self.sampling_rate * (self.sdo.get_k() as f64)
    }

    // Gibt t_fading zurück
    #[getter]
    pub fn t_fading(&self) -> f64 {
        -1.0 / self.fading.ln()
    }

    /// Gibt Anzahl der Observer zurück
    #[getter]
    pub fn observer_count(&self) -> usize {
        self.sdo.observers.len()
    }

    /// Gibt Anzahl der aktiven Observer zurück
    #[getter]
    pub fn num_active(&self) -> usize {
        self.sdo.observers.get_num_active()
    }

    /// Gibt Anzahl der verarbeiteten Datenpunkte zurück
    #[getter]
    pub fn data_points_processed(&self) -> usize {
        self.data_points_processed
    }

    /// Anzahl durchgeführter Observer-Ersetzungen (für Tests / Beobachtung).
    #[getter]
    pub fn replacement_count(&self) -> usize {
        self.replacement_count
    }

    /// Gibt Observer-Informationen für einen bestimmten Index zurück
    #[pyo3(signature = (index))]
    pub fn get_observer_info(
        &self,
        _py: Python,
        index: usize,
    ) -> PyResult<(f64, f64, f64, bool, Option<i32>)> {
        if let Some(observer) = self.sdo.observers.get(index) {
            let is_active = self.sdo.observers.is_active(index);
            Ok((
                observer.observations,
                observer.age,
                observer.time,
                is_active,
                observer.get_label().map(|l| l as i32),
            ))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Observer with index {} not found",
                index
            )))
        }
    }

    /// Gibt alle Observer-Informationen zurück
    #[getter]
    pub fn get_all_observer_info(
        &self,
        _py: Python,
    ) -> PyResult<Vec<(Vec<f64>, f64, f64, f64, bool, Option<i32>)>> {
        let mut result = Vec::new();
        for observer in self.sdo.observers.iter_observers(false) {
            let is_active = self.sdo.observers.is_active(observer.index);
            result.push((
                observer.data.clone(),
                observer.observations,
                observer.age,
                observer.time,
                is_active,
                observer.get_label().map(|l| l as i32),
            ));
        }
        Ok(result)
    }
}

impl SDOstream {
    /// For benchmarks only: create SDOstream with dimension (no Python).
    pub fn new_for_benchmark(
        k: usize,
        t_fading: f64,
        t_sampling: f64,
        x: usize,
        rho: f64,
        dimension: usize,
    ) -> Self {
        let mut instance = Self {
            sdo: SDO::new(k, x, rho, "euclidean".to_string(), None),
            fading: Self::get_fading_static(t_fading),
            sampling_rate: t_sampling / (k as f64),
            data_points_processed: 0,
            use_explicit_time: true,
            last_replacement_time: 0.0,
            pending_replacements: 0,
            replacement_count: 0,
            next_observer_index: 0,
        };
        instance.initialize_impl(Some(dimension), None, 0.0);
        instance
    }

    pub(crate) fn initialize_impl(
        &mut self,
        dimension: Option<usize>,
        data: Option<&Vec<Vec<f64>>>,
        time: f64,
    ) {
        // Determine what data to use
        let data_points = match (data, dimension) {
            // Case 1: User provided data
            (Some(existing_data), _) => {
                // Validate dimension if provided
                if let Some(dim) = dimension {
                    // Check if all points have correct dimension
                    for (i, point) in existing_data.iter().enumerate() {
                        if point.len() != dim {
                            panic!(
                                "Point {} has dimension {} but expected {}",
                                i,
                                point.len(),
                                dim,
                            );
                        }
                    }
                }
                existing_data.clone() // Clone the user's data
            }
            // Case 2: No data provided but dimension specified → uniform in unit square [0,1]^d
            (None, Some(dim)) => sample_random_matrix_uniform_unit(dim, self.k()),
            // Case 3: No data and no dimension → initialize empty
            (None, None) => {
                self.sdo.observers.set_num_active(0);
                self.last_replacement_time = time;
                self.pending_replacements = 0;
                self.data_points_processed = 0;
                return;
            }
        };

        for (idx, point_data) in data_points.iter().enumerate() {
            let observer = crate::obs::Observer {
                data: point_data.clone(),
                observations: 1.0, // Start mit 1 observation
                time: time,
                age: 1.0,
                index: idx,
                local_threshold: 0.0,
                label_observations: HashMap::new(),
                label_time: time,
            };
            self.sdo.observers.insert(observer);
        }

        // Setze num_active basierend auf rho
        self.sdo.observers.set_num_active(
            ((self.sdo.observers.len() as f64) * (1.0 - self.rho())).ceil() as usize,
        );

        // Initialisiere Lazy Replacement: Startzeit setzen
        self.last_replacement_time = time;
        self.pending_replacements = 0; // Keine ausstehenden Ersetzungen bei Initialisierung
        self.data_points_processed = data_points.len();
        self.next_observer_index = self.sdo.observers.len();
    }

    /// Verarbeitet einen einzelnen Datenpunkt (Rust-intern).
    #[deprecated(since = "0.1.0", note = "Use learn_impl instead")]
    #[allow(unused)]
    pub fn learn_point(
        &mut self,
        point: &Vec<f64>,
        time: f64,
    ) -> (f64, Vec<NeighborInfo>, Option<Vec<NeighborInfo>>) {
        // Schritt 1: Anzahl der Ersetzungen bestimmen (Poisson-basiert)
        let n_replacements = self.n_replacements_impl(time);

        // Schritt 2: Predict mit k_learn = x + n_replacements
        let k_learn = n_replacements.min(1);
        let (median, active_neighbors, all_neighbors_opt) =
            self.sdo.predict_point(point, Some(true), Some(k_learn));

        // Schritt 3: Ersetzung durchführen (nur eine, wenn n_replacements > 0)
        let replacement_pair = if n_replacements > 0 {
            let pair = self.sample_point(point, time, None);
            // Aktualisiere pending_replacements und last_replacement_time
            if pair.is_some() {
                self.pending_replacements = n_replacements.saturating_sub(1);
                self.last_replacement_time = time;
            }
            pair
        } else {
            None
        };

        // Schritt 4: Fit – Passt all_neighbors_opt an und gibt finale all_neighbors zurück
        let final_all_neighbors_opt =
            self.fit_point(all_neighbors_opt, replacement_pair, point, time);

        // Schritt 5: Update observations mit Fading (konsistent mit Batch-Verarbeitung)
        if let Some(ref neighbors) = final_all_neighbors_opt {
            let processed =
                self.sdo
                    .observers
                    .update(vec![neighbors.clone()], vec![time], self.fading);
            self.data_points_processed += processed;
        }

        (median, active_neighbors, final_all_neighbors_opt)
    }

    /// Verarbeitet einen Batch von (point, time) sequentiell (Rust-intern).
    pub(crate) fn learn_impl(
        &mut self,
        points: &[Vec<f64>],
        times: &[f64],
    ) -> Vec<(f64, Vec<NeighborInfo>, Option<Vec<NeighborInfo>>)> {
        assert_eq!(
            points.len(),
            times.len(),
            "points und times müssen gleiche Länge haben"
        );

        if points.is_empty() {
            return Vec::new();
        }

        let batch_size = points.len();
        let reference_time = *times.last().unwrap();

        // Schritt 1: Anzahl der Ersetzungen bestimmen (Poisson-basiert)
        let n_replacements_total = self.n_replacements_impl(reference_time);

        // Wenn n_replacements größer als Batch-Größe, füge überschüssige zu pending_replacements hinzu
        let n_replacements = n_replacements_total.min(batch_size);
        if n_replacements_total > batch_size {
            self.pending_replacements = n_replacements_total - batch_size;
        } else {
            self.pending_replacements = 0;
        }

        // Schritt 2: Predict für alle Punkte im Batch
        let predict_results: Vec<(f64, Vec<NeighborInfo>, Option<Vec<NeighborInfo>>)> = points
            .iter()
            .map(|point| {
                self.sdo
                    .predict_point(point, Some(true), Some(n_replacements))
            })
            .collect();

        // Schritt 3: Sampling/Replacing von n_replacements Punkten iterativ
        let replacement_pairs = if n_replacements > 0 {
            let pairs = self.sample_impl(points, times, n_replacements);
            // Aktualisiere last_replacement_time wenn Ersetzungen durchgeführt wurden
            // (sample_impl ruft sample_point auf, das bereits last_replacement_time aktualisiert,
            // aber wir setzen es auf reference_time für Konsistenz)
            self.last_replacement_time = reference_time;
            pairs
        } else {
            vec![None; batch_size]
        };

        // Schritt 4: fit_impl - Passt alle all_neighbors_opt an
        let final_all_neighbors_batch = self.fit_impl(
            predict_results
                .iter()
                .map(|(_, _, opt)| opt.as_ref())
                .collect(),
            &replacement_pairs,
            points,
        );

        // Schritt 5: Update observations mit Fading
        // Verwende die neue update() Funktion mit allen final_all_neighbors
        let all_neighbors_for_update: Vec<Vec<NeighborInfo>> = final_all_neighbors_batch
            .iter()
            .filter_map(|opt| opt.as_ref().map(|v| v.clone()))
            .collect();

        let observation_times_for_update: Vec<f64> = final_all_neighbors_batch
            .iter()
            .enumerate()
            .filter_map(|(i, opt)| if opt.is_some() { Some(times[i]) } else { None })
            .collect();

        if !all_neighbors_for_update.is_empty() {
            let processed = self.sdo.observers.update(
                all_neighbors_for_update,
                observation_times_for_update,
                self.fading,
            );
            self.data_points_processed += processed;
        }

        // Erstelle Ergebnisse: (median, active_neighbors, final_all_neighbors_opt)
        predict_results
            .into_iter()
            .zip(final_all_neighbors_batch.into_iter())
            .map(|((median, active_neighbors, _), final_neighbors_opt)| {
                (median, active_neighbors, final_neighbors_opt)
            })
            .collect()
    }

    /// Bestimmt die Anzahl der Ersetzungen basierend auf Poisson-Verteilung.
    pub(crate) fn n_replacements_impl(&self, time: f64) -> usize {
        let elapsed = time - self.last_replacement_time;
        if elapsed < 0.0 {
            panic!(
                "Ungültige Zeit: current time muss größer oder gleich last_replacement_time sein"
            );
        }
        let lambda_events = elapsed / self.sampling_rate;
        let num_replacements = self.sample_poisson(lambda_events);
        num_replacements + self.pending_replacements
    }

    /// Batch-Variante von fit_point: Passt alle all_neighbors_opt an und gibt finale all_neighbors zurück.
    /// Aktualisiert keine observations - das wird später in update() gemacht.
    pub(crate) fn fit_impl(
        &mut self,
        all_neighbors_batch: Vec<Option<&Vec<NeighborInfo>>>,
        replacement_pairs: &[Option<(usize, usize)>],
        points: &[Vec<f64>],
    ) -> Vec<Option<Vec<NeighborInfo>>> {
        assert_eq!(
            all_neighbors_batch.len(),
            replacement_pairs.len(),
            "all_neighbors_batch und replacement_pairs müssen gleiche Länge haben"
        );
        assert_eq!(
            all_neighbors_batch.len(),
            points.len(),
            "all_neighbors_batch und points müssen gleiche Länge haben"
        );

        let mut results = Vec::with_capacity(all_neighbors_batch.len());

        for ((all_neighbors_opt, replacement_pair), point) in all_neighbors_batch
            .iter()
            .zip(replacement_pairs.iter())
            .zip(points.iter())
        {
            let final_neighbors = self.fit_point(
                all_neighbors_opt.cloned(),
                *replacement_pair,
                point,
                0.0, // time wird nicht verwendet
            );
            results.push(final_neighbors);
        }

        results
    }

    /// Führt den Fit-Schritt durch: Passt all_neighbors_opt an und gibt finale all_neighbors zurück.
    /// Aktualisiert keine observations - das wird später in update() gemacht.
    pub(crate) fn fit_point(
        &mut self,
        mut all_neighbors_opt: Option<Vec<NeighborInfo>>,
        replacement_pair: Option<(usize, usize)>,
        point: &[f64],
        _time: f64,
    ) -> Option<Vec<NeighborInfo>> {
        // Extrahiere replace_idx und new_idx aus replacement_pair
        let replace_idx = replacement_pair.map(|(_, r)| r);
        let new_idx = replacement_pair.map(|(n, _)| n);

        // Passt all_neighbors_opt an: entfernt replace_idx, fügt new_idx hinzu
        let mut candidates: Vec<(usize, f64)> = if let Some(ref mut neighbors) = all_neighbors_opt {
            // Entferne replace_idx falls vorhanden
            if let Some(ridx) = replace_idx {
                neighbors.retain(|n| n.index != ridx);
            }
            neighbors.iter().map(|n| (n.index, n.distance)).collect()
        } else {
            Vec::new()
        };

        // Füge new_idx hinzu wenn vorhanden und nah genug
        if let Some(ni) = new_idx {
            if let Some(d) = self.sdo.observers.distance_from_point(point, ni) {
                candidates.push((ni, d));
            }
        }

        // Sortiere nach Distanz (aufsteigend) - keine Update hier, wird später in update() gemacht
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Erstelle finale all_neighbors_opt mit angepassten Werten (sortiert nach Distanz)
        let final_neighbors: Vec<NeighborInfo> = candidates
            .into_iter()
            .map(|(idx, dist)| NeighborInfo {
                index: idx,
                distance: dist,
                is_active: self.sdo.observers.is_active(idx),
            })
            .collect();

        Some(final_neighbors)
    }

    /// Vorhersage für einen einzelnen Punkt (Rust-intern).
    pub(crate) fn predict_point(
        &self,
        point: &Vec<f64>,
        learn: Option<bool>,
        k_learn: Option<usize>,
    ) -> (f64, Vec<NeighborInfo>, Option<Vec<NeighborInfo>>) {
        let (median, active_neighbors, all_neighbors_opt) =
            self.sdo.predict_point(point, learn, k_learn);
        (median, active_neighbors, all_neighbors_opt)
    }

    /// Batch-Vorhersage (Rust-intern).
    pub(crate) fn predict_impl(
        &self,
        points: &[Vec<f64>],
        learn: Option<bool>,
        k_learn: Option<usize>,
    ) -> Vec<(f64, Vec<NeighborInfo>, Option<Vec<NeighborInfo>>)> {
        points
            .iter()
            .map(|point| self.predict_point(point, learn, k_learn))
            .collect()
    }

    /// Gibt fading zurück (für interne Verwendung)
    pub(crate) fn get_fading(&self) -> f64 {
        self.fading
    }

    /// Gibt use_explicit_time zurück (für interne Verwendung)
    pub(crate) fn get_use_explicit_time(&self) -> bool {
        self.use_explicit_time
    }

    /// Gibt data_points_processed zurück (für interne Verwendung)
    pub(crate) fn get_data_points_processed(&self) -> usize {
        self.data_points_processed
    }

    /// Gibt pending_replacements zurück (für interne Verwendung)
    #[allow(unused)]
    pub(crate) fn get_pending_replacements(&self) -> usize {
        self.pending_replacements
    }

    /// Setzt pending_replacements (für interne Verwendung)
    #[allow(unused)]
    pub(crate) fn set_pending_replacements(&mut self, value: usize) {
        self.pending_replacements = value;
    }

    /// Gibt last_replacement_time zurück (für interne Verwendung)
    #[allow(unused)]
    pub(crate) fn get_last_replacement_time(&self) -> f64 {
        self.last_replacement_time
    }

    /// Setzt last_replacement_time (für interne Verwendung)
    pub(crate) fn set_last_replacement_time(&mut self, value: f64) {
        self.last_replacement_time = value;
    }

    /// Erhöht data_points_processed (für interne Verwendung)
    pub(crate) fn increment_data_points_processed(&mut self, value: usize) {
        self.data_points_processed += value;
    }

    /// Gibt sdo zurück (für interne Verwendung)
    pub(crate) fn get_sdo(&self) -> &SDO {
        &self.sdo
    }

    /// Gibt sdo mut zurück (für interne Verwendung)
    pub(crate) fn get_sdo_mut(&mut self) -> &mut SDO {
        &mut self.sdo
    }
}

impl SDOstream {
    /// Generiert eine Poisson-verteilte Zufallszahl
    /// Verwendet Knuth's Algorithm für kleine λ, sonst Normal-Approximation
    fn sample_poisson(&self, lambda: f64) -> usize {
        if lambda <= 0.0 {
            return 0;
        }

        let mut rng = thread_rng();

        if lambda < 30.0 {
            // Knuth's Algorithm für kleine λ
            let l = (-lambda).exp();
            let mut k = 0;
            let mut p = 1.0;

            loop {
                k += 1;
                p *= rng.gen::<f64>();
                if p <= l {
                    break;
                }
            }
            k - 1
        } else {
            // Normal-Approximation für große λ (Box-Muller Transformation)
            let mean = lambda;
            let std_dev = lambda.sqrt();
            // Generiere zwei normalverteilte Werte
            let u1: f64 = rng.gen();
            let u2: f64 = rng.gen();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let sample = mean + std_dev * z;
            sample.max(0.0) as usize
        }
    }

    /// Führt eine Ersetzung durch (inkrementell) und gibt (new_idx, replace_idx) zurück.
    pub(crate) fn sample_point(
        &mut self,
        point: &[f64],
        time: f64,
        worst_idx: Option<usize>,
    ) -> Option<(usize, usize)> {
        // Bestimme replace_idx (schlechtester Observer)
        let replace_idx = worst_idx.unwrap_or_else(|| {
            self.sdo
                .observers
                .find_k_worst(Some(1))
                .first()
                .cloned()
                .unwrap()
        });

        // Führe Ersetzung durch (eindeutiger Index pro neuem Observer)
        self.replacement_count += 1;
        let new_index = self.next_observer_index;
        self.next_observer_index += 1;
        let new_observer = Observer {
            data: point.to_vec(),
            observations: 0.0,
            time,
            age: 1.0,
            index: new_index,
            local_threshold: 0.0,
            label_observations: HashMap::new(),
            label_time: time,
        };
        let new_observer_clone = Observer {
            data: new_observer.data.clone(),
            observations: new_observer.observations,
            time: new_observer.time,
            age: new_observer.age,
            index: new_observer.index,
            local_threshold: new_observer.local_threshold,
            label_observations: new_observer.label_observations.clone(),
            label_time: new_observer.label_time,
        };
        let success = self.sdo.replace_observer(replace_idx, new_observer);
        if !success {
            self.sdo.observers.insert(new_observer_clone);
        }

        // Aktualisiere Zeit und pending_replacements
        self.last_replacement_time = time;
        // pending_replacements wird in learn_point aktualisiert basierend auf n_replacements

        Some((new_index, replace_idx))
    }

    /// Batch-Variante: Führt genau n_replacements viele Ersetzungen durch, zufällig auf Punkte verteilt.
    pub(crate) fn sample_impl(
        &mut self,
        points: &[Vec<f64>],
        times: &[f64],
        n_replacements: usize,
    ) -> Vec<Option<(usize, usize)>> {
        assert_eq!(points.len(), times.len());
        let mut results = vec![None; points.len()];

        if n_replacements == 0 || points.is_empty() {
            return results;
        }

        let worst_idxs = self.sdo.observers.find_k_worst(Some(n_replacements));
        if worst_idxs.is_empty() {
            return results;
        }

        // Tatsächliche Ersetzungen = min(gewünscht, verfügbare worst, Punkte)
        let actual_replacements = n_replacements.min(worst_idxs.len()).min(points.len());

        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..points.len()).collect();
        indices.shuffle(&mut rng);

        for (worst_i, &idx) in indices.iter().take(actual_replacements).enumerate() {
            results[idx] = self.sample_point(&points[idx], times[idx], Some(worst_idxs[worst_i]));
        }

        results
    }
}

impl Default for SDOstream {
    fn default() -> Self {
        Self::new(
            200,
            100.0,
            100.0,
            5,
            0.1,
            "euclidean".to_string(),
            None,
            Some(2),
            None,
            None,
        )
        .expect("SDOstream::default()")
    }
}
