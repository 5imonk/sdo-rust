use core::panic;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::{thread_rng, Rng};
use std::collections::HashMap;
use std::f64;

use crate::obs::{NeighborInfo, Observer};
use crate::sdo_impl::SDO;
use crate::utils::{
    data_to_matrix, sample_random_matrix_uniform_unit, time_to_f64, times_to_vec_batch,
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
    data_points_processed: usize, // Zähler für Sampling
    k: usize,                     // Anzahl der Observer (Modellgröße)
    t_fading: f64,                // T-Parameter für fading: f = exp(-T_fading^-1)
    t_sampling: f64, // T-Parameter für Sampling-Rate (durchschnittliches Intervall zwischen Ersetzungen)
    rho: f64,        // Rho-Parameter (für num_active Berechnung)
    use_explicit_time: bool, // Wenn true, erwartet learn() time-Parameter; sonst auto-increment
    last_replacement_time: f64, // Zeit der letzten Prüfung/Ersetzung (für Lazy Replacement)
    pending_replacements: usize, // Anzahl der ausstehenden Ersetzungen (wenn num_replacements > 1)
}

#[pymethods]
#[allow(clippy::too_many_arguments)]
impl SDOstream {
    #[new]
    #[pyo3(signature = (k, x, t_fading, t_sampling = None, distance = "euclidean".to_string(), minkowski_p = None, rho = 0.1, dimension = None, data = None, time = None))]
    pub fn new(
        k: usize,
        x: usize,
        t_fading: f64,
        t_sampling: Option<f64>,
        distance: String,
        minkowski_p: Option<f64>,
        rho: f64,
        dimension: Option<usize>,
        data: Option<PyReadonlyArray2<f64>>,
        time: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Self> {
        // t_sampling default ist t_fading wenn nicht angegeben
        let t_sampling_value = t_sampling.unwrap_or(t_fading);

        let mut instance = Self {
            sdo: SDO::new(k, x, rho, distance, minkowski_p),
            fading: Self::get_fading_static(t_fading),
            data_points_processed: 0,
            k,
            t_fading,
            t_sampling: t_sampling_value,
            rho,
            use_explicit_time: time.is_some(), // Default: auto-increment
            last_replacement_time: 0.0,        // Startzeit für Lazy Replacement
            pending_replacements: 0,           // Keine ausstehenden Ersetzungen
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
        Python::with_gil(|py| {
            if rows == 1 {
                Ok(scores[0].into_py(py))
            } else {
                Ok(scores.into_py(py))
            }
        })
    }

    /// Berechnet den Outlier-Score für einen oder mehrere Datenpunkte (Batch-Verarbeitung).
    /// Ein einzelner Punkt wird als Batch der Größe 1 behandelt.
    #[pyo3(signature = (points))]
    pub fn predict(&self, points: PyReadonlyArray2<f64>) -> PyResult<PyObject> {
        let (points_vec, rows) = data_to_matrix(points);

        let results = self.predict_impl(&points_vec, Some(false));
        let scores: Vec<f64> = results.iter().map(|(median, _, _)| *median).collect();

        // Wenn nur ein Punkt: Rückgabe als einzelner Wert, sonst als Liste
        Python::with_gil(|py| {
            if rows == 1 {
                Ok(scores[0].into_py(py))
            } else {
                Ok(scores.into_py(py))
            }
        })
    }

    /// Gibt x zurück (Anzahl der nächsten Nachbarn)
    #[getter]
    pub fn x(&self) -> usize {
        self.sdo.x
    }

    /// Gibt k zurück (Anzahl der Observer)
    #[getter]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Gibt t_fading zurück
    #[getter]
    pub fn t_fading(&self) -> f64 {
        self.t_fading
    }

    /// Gibt t_sampling zurück
    #[getter]
    pub fn t_sampling(&self) -> f64 {
        self.t_sampling
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
            (None, Some(dim)) => sample_random_matrix_uniform_unit(dim, self.k),
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
            };
            self.sdo.observers.insert(observer);
        }

        // Setze num_active basierend auf rho
        self.sdo
            .observers
            .set_num_active(((self.sdo.observers.len() as f64) * (1.0 - self.rho)).ceil() as usize);

        // Initialisiere Lazy Replacement: Startzeit setzen
        self.last_replacement_time = time;
        self.pending_replacements = 0; // Keine ausstehenden Ersetzungen bei Initialisierung
        self.data_points_processed = data_points.len();
    }

    /// Verarbeitet einen einzelnen Datenpunkt (Rust-intern).
    pub(crate) fn learn_point(
        &mut self,
        point: &Vec<f64>,
        time: f64,
    ) -> (f64, Vec<NeighborInfo>, Option<Vec<NeighborInfo>>) {
        let x = self.sdo.x;

        // Schritt 1: Ersetzungen festlegen (nur markieren, noch nicht ausführen)
        let mark = self.mark_replacements_impl(time);

        // Schritt 2: Predict – x nächste aktive Observer, Median
        let (median, active_neighbors, all_neighbors_opt) =
            self.sdo.predict_point(point, Some(true));

        // Schritt 3: Ersetzungen ausführen (markierte Observer ersetzen)
        let new_index = if let Some((replace_idx, total_replacements)) = mark {
            let ni = self.replace_observer_at(replace_idx, point, time);
            self.last_replacement_time = time;
            self.pending_replacements = total_replacements - 1;
            ni
        } else {
            None
        };

        let replace_idx = mark.map(|(idx, _)| idx);

        // Schritt 4: Fit – x nächste für Update (Wiederverwendung von all_neighbors aus Schritt 2)
        let nearest_observer_indices = self.build_nearest_for_fit(
            all_neighbors_opt.as_deref(),
            replace_idx,
            new_index,
            point,
            x,
        );
        self.sdo.observers.update_observations_with_fading(
            &nearest_observer_indices,
            self.fading,
            time,
        );
        self.data_points_processed += 1;

        (median, active_neighbors, all_neighbors_opt)
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
        let mut results = Vec::with_capacity(points.len());
        for (point, time) in points.iter().zip(times.iter()) {
            let r = self.learn_point(point, *time);
            results.push(r);
        }
        results
    }

    /// Schritt 1: Liefert (replace_idx, total_replacements) wenn eine Ersetzung fällig ist.
    fn mark_replacements_impl(&self, time: f64) -> Option<(usize, usize)> {
        let elapsed = time - self.last_replacement_time;
        if elapsed < 0.0 {
            panic!(
                "Ungültige Zeit: current time muss größer oder gleich last_replacement_time sein"
            );
        }
        let effective_sampling_interval = self.t_sampling / (self.k as f64);
        let lambda_events = elapsed / effective_sampling_interval;
        let num_replacements = self.sample_poisson(lambda_events);
        let total_replacements = num_replacements + self.pending_replacements;
        if total_replacements == 0 {
            return None;
        }
        let worst_scores = self.sdo.observers.find_k_worst_normalized_scores(Some(1));
        let (replace_idx, _) = worst_scores.first()?;
        Some((*replace_idx, total_replacements))
    }

    /// Schritt 4-Hilfe: x nächste Indizes für Update aus all_neighbors (ohne Ersetzte, mit neuen).
    fn build_nearest_for_fit(
        &self,
        all_neighbors: Option<&[crate::obs::NeighborInfo]>,
        replace_idx: Option<usize>,
        new_index: Option<usize>,
        point: &[f64],
        x: usize,
    ) -> Vec<usize> {
        let mut candidates: Vec<(usize, f64)> = all_neighbors
            .unwrap_or(&[])
            .iter()
            .filter(|n| replace_idx != Some(n.index))
            .map(|n| (n.index, n.distance))
            .collect();
        if let Some(ni) = new_index {
            if let Some(d) = self.sdo.observers.distance_from_point(point, ni) {
                candidates.push((ni, d));
            }
        }
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.into_iter().take(x).map(|(idx, _)| idx).collect()
    }

    /// Vorhersage für einen einzelnen Punkt (Rust-intern).
    pub(crate) fn predict_point(
        &self,
        point: &Vec<f64>,
        learn: Option<bool>,
    ) -> (f64, Vec<NeighborInfo>, Option<Vec<NeighborInfo>>) {
        let (median, active_neighbors, all_neighbors_opt) = self.sdo.predict_point(point, learn);
        (median, active_neighbors, all_neighbors_opt)
    }

    /// Batch-Vorhersage (Rust-intern).
    pub(crate) fn predict_impl(
        &self,
        points: &[Vec<f64>],
        learn: Option<bool>,
    ) -> Vec<(f64, Vec<NeighborInfo>, Option<Vec<NeighborInfo>>)> {
        points
            .iter()
            .map(|point| self.predict_point(point, learn))
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
    /// Prüft und führt Ersetzungen basierend auf verstrichener Zeit durch (Lazy Replacement).
    /// Wird nicht mehr von learn_point aufgerufen (4-Schritte-Struktur); für Batch/API behalten.
    #[allow(dead_code)]
    fn sample_impl(&mut self, point: &[f64], time: f64) -> Option<usize> {
        let elapsed = time - self.last_replacement_time;

        if elapsed < 0.0 {
            panic!(
                "Ungültige Zeit: current time muss größer oder gleich last_replacement_time sein"
            );
        }

        // Erwartete Anzahl von Ersetzungen in elapsed Zeit: λ_events = elapsed / t_sampling
        // t_sampling ist das durchschnittliche Intervall zwischen Ersetzungen
        let effective_sampling_interval = self.t_sampling / (self.k as f64);
        let lambda_events = elapsed / effective_sampling_interval;

        // Simuliere Poisson-Anzahl von Ersetzungen
        let num_replacements = self.sample_poisson(lambda_events);

        // Berücksichtige ausstehende Ersetzungen von vorherigen Aufrufen
        let total_replacements = num_replacements + self.pending_replacements;

        // Führe nur eine Ersetzung durch (auch wenn total_replacements > 1)
        if total_replacements == 0 {
            return None; // Keine Ersetzungen
        }

        let idx = self.replace_observer(point, time)?;

        if Some(idx).is_none() {
            return None; // Keine Ersetzungen möglich
        }

        self.last_replacement_time = time;

        // Speichere verbleibende Ersetzungen für nächste Aufrufe
        self.pending_replacements = total_replacements - 1;

        // Wenn total_replacements == 0: last_replacement_time bleibt unverändert
        // (Zeit wird beim nächsten Aufruf akkumuliert)

        Some(idx)
    }

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

    /// Ersetzt einen Observer an einem gegebenen Index (Schritt 3 der 4-Schritte-Struktur).
    fn replace_observer_at(
        &mut self,
        replace_idx: usize,
        point: &[f64],
        time: f64,
    ) -> Option<usize> {
        let new_index = self.data_points_processed;
        let new_observer = Observer {
            data: point.to_vec(),
            observations: 1.0,
            time,
            age: 1.0,
            index: new_index,
            local_threshold: 0.0,
            label_observations: HashMap::new(),
        };
        let new_observer_clone = Observer {
            data: new_observer.data.clone(),
            observations: new_observer.observations,
            time: new_observer.time,
            age: new_observer.age,
            index: new_observer.index,
            local_threshold: new_observer.local_threshold,
            label_observations: new_observer.label_observations.clone(),
        };
        let success = self.sdo.replace_observer(replace_idx, new_observer);
        if !success {
            self.sdo.observers.insert(new_observer_clone);
        }
        Some(new_index)
    }

    /// Ersetzt einen Observer basierend auf normalisierter Qualitätsmetrik P̃ω = Pω / Hω
    /// (Legacy: findet den schlechtesten und ruft replace_observer_at auf.)
    #[allow(dead_code)]
    fn replace_observer(&mut self, point: &[f64], time: f64) -> Option<usize> {
        let worst_scores = self.sdo.observers.find_k_worst_normalized_scores(Some(1));
        let (replace_idx, _) = worst_scores.first()?;
        self.replace_observer_at(*replace_idx, point, time)
    }
}

impl Default for SDOstream {
    fn default() -> Self {
        Self::new(
            200,
            5,
            100.0,
            None, // t_sampling = t_fading
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
