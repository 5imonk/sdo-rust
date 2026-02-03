use core::panic;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::{thread_rng, Rng};
use std::collections::HashMap;
use std::f64;

use crate::obs::Observer;
use crate::sdo_impl::SDO;
use crate::utils::{
    compute_median, data_to_matrix, point_to_vec, sample_random_matrix_uniform_unit, time_to_f64,
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
            Some(data_array) => Some(data_to_matrix(data_array)),
            None => None,
        };

        self.initialize_impl(dimension, data_vec.as_ref(), start_time);

        Ok(())
    }

    /// Verarbeitet einen einzelnen Datenpunkt aus dem Stream
    #[pyo3(signature = (point, *, time = None))]
    pub fn learn(
        &mut self,
        point: PyReadonlyArray2<f64>,
        time: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<f64> {
        let point_vec: Vec<f64> = point_to_vec(point);

        // Bestimme Zeit basierend auf Initialisierungs-Strategie
        let current_time = time_to_f64(time, self.use_explicit_time, self.data_points_processed)?;

        let (median, _nearest_active_indices) = self.learn_impl(&point_vec, current_time);

        Ok(median)
    }

    /// Berechnet den Outlier-Score für einen Datenpunkt (delegiert an SDO)
    #[pyo3(signature = (point))]
    pub fn predict(&self, point: PyReadonlyArray2<f64>) -> PyResult<f64> {
        let point_vec: Vec<f64> = point_to_vec(point);

        let (median, _nearest_active_indices) = self.predict_impl(&point_vec);

        Ok(median)
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

    pub(crate) fn learn_impl(&mut self, point: &Vec<f64>, time: f64) -> (f64, Vec<usize>) {
        // Schritt 1: Finde x-nächste Observer (verwende optimierte unified search mit active count)
        let x = self.sdo.x;

        // Schritt 1a: Finde alle Observer für learning
        let (all_neighbors, active_neighbors) = self
            .sdo
            .observers
            .search_neighbors_unified(&point, x, false);
        let nearest_observer_indices: Vec<usize> = all_neighbors.iter().map(|n| n.index).collect();

        let nearest_active_distances: Vec<f64> =
            active_neighbors.iter().map(|n| n.distance).collect();
        let nearest_active_indices: Vec<usize> = active_neighbors.iter().map(|n| n.index).collect();
        let median = if !nearest_active_distances.is_empty() {
            compute_median(&nearest_active_distances)
        } else {
            f64::INFINITY
        };

        // Schritt 2: Update Pω und Hω für alle Observer mit zeitbasiertem Exponential Moving Average
        // Hω ← f^(ti - ti-1) · Hω + 1, Pω ← f^(ti - ti-1) · Pω + 1 (wenn nearest) bzw. f^(ti - ti-1) · Pω
        self.sdo.observers.update_observations_with_fading(
            &nearest_observer_indices,
            self.fading,
            time,
        );
        // Increment data_points_processed für auto-increment Modus
        self.data_points_processed += 1;

        // Schritt 3: Sampling - Lazy Replacement basierend auf verstrichener Zeit (Poisson-basiert)
        self.sample_impl(&point, time);

        (median, nearest_active_indices)
    }

    pub(crate) fn predict_impl(&self, point: &Vec<f64>) -> (f64, Vec<usize>) {
        let (median, nearest_active_indices) = self.sdo.predict_impl(point);
        (median, nearest_active_indices)
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
    /// Prüft und führt Ersetzungen basierend auf verstrichener Zeit durch (Lazy Replacement)
    /// Verwendet Poisson-Verteilung für die Anzahl der Ersetzungen
    /// Funktioniert sowohl für einzelne Punkte als auch für Batches
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

    /// Ersetzt einen Observer basierend auf normalisierter Qualitätsmetrik P̃ω = Pω / Hω
    fn replace_observer(&mut self, point: &[f64], time: f64) -> Option<usize> {
        // Verwende die optimierte find_k_worst_normalized_scores Methode - O(1) statt O(n)
        let worst_scores = self.sdo.observers.find_k_worst_normalized_scores(Some(1));
        let (replace_idx, _score) = match worst_scores.first() {
            Some((idx, score)) => (*idx, *score),
            None => return None, // Keine Observer vorhanden
        };

        // Erstelle neuen Observer
        // Für neue Observer: time sollte auf die aktuelle Zeit gesetzt werden
        // Da wir hier keine Zeit haben, verwenden wir 0.0 (wird beim nächsten Update korrigiert)
        let new_index = self.data_points_processed;
        let new_observer = Observer {
            data: point.to_vec(),
            observations: 1.0, // Neuer Observer startet mit Pω = 1
            time: time,        // Setze time auf aktuelle Zeit
            age: 1.0,          // Neuer Observer startet mit Hω = 1
            index: new_index as usize,
            local_threshold: 0.0,
            label_observations: HashMap::new(),
        };

        // Verwende SDO's replace_observer Methode - O(log n)
        // replace() sollte immer erfolgreich sein, wenn remove() erfolgreich war
        // Falls nicht, füge den neuen Observer trotzdem hinzu
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
            // Fallback: Füge den neuen Observer hinzu, auch wenn replace fehlgeschlagen ist
            self.sdo.observers.insert(new_observer_clone);
        }

        Some(new_index)
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
