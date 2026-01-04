use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use rand::{thread_rng, Rng};
use std::f64;

use crate::sdo_impl::{SDOParams, SDO};
use crate::utils::Observer;

/// Parameter-Struktur für SDOstream
#[pyclass]
#[derive(Clone)]
pub struct SDOstreamParams {
    #[pyo3(get, set)]
    pub k: usize, // Anzahl der Observer (Modellgröße)
    #[pyo3(get, set)]
    pub x: usize, // Anzahl der nächsten Observer für Prediction
    #[pyo3(get, set)]
    pub t: f64, // T-Parameter für fading: f = exp(-T^-1)
    #[pyo3(get, set)]
    pub distance: String, // "euclidean", "manhattan", "chebyshev", "minkowski"
    #[pyo3(get, set)]
    pub minkowski_p: Option<f64>, // Für Minkowski-Distanz
    #[pyo3(get, set)]
    pub tree_type: String, // "vptree" (default) oder "kdtree"
}

#[pymethods]
impl SDOstreamParams {
    #[new]
    #[pyo3(signature = (k, x, t, distance = "euclidean".to_string(), minkowski_p = None, tree_type = "vptree".to_string()))]
    pub fn new(
        k: usize,
        x: usize,
        t: f64,
        distance: String,
        minkowski_p: Option<f64>,
        tree_type: String,
    ) -> Self {
        Self {
            k,
            x,
            t,
            distance,
            minkowski_p,
            tree_type,
        }
    }
}

impl SDOstreamParams {
    /// Berechnet den Fading-Parameter f = exp(-T^-1)
    fn get_fading(&self) -> f64 {
        (-1.0 / self.t).exp()
    }

    /// Berechnet die Sampling-Rate T_k = -k · ln(f)
    fn get_sampling_rate(&self) -> f64 {
        let f = self.get_fading();
        -(self.k as f64) * f.ln()
    }

    /// Konvertiert zu SDOParams (mit rho=0, da keine Entfernung in SDOstream)
    fn to_sdo_params(&self) -> SDOParams {
        SDOParams {
            k: self.k,
            x: self.x,
            rho: 0.0, // In SDOstream werden keine Observer entfernt
            distance: self.distance.clone(),
            minkowski_p: self.minkowski_p,
            tree_type: self.tree_type.clone(),
        }
    }
}

/// SDOstream Algorithm - Streaming-Version von SDO
/// Baut auf SDO auf und fügt nur Streaming-spezifische Funktionalität hinzu
#[pyclass]
#[allow(clippy::upper_case_acronyms)]
pub struct SDOstream {
    sdo: SDO,                     // Basis SDO-Implementierung
    fading: f64,                  // f = exp(-T^-1)
    sampling_rate: f64,           // T_k = -k · ln(f)
    data_points_processed: usize, // Zähler für Sampling
}

#[pymethods]
impl SDOstream {
    #[new]
    pub fn new() -> Self {
        Self {
            sdo: SDO::new(),
            fading: 0.99,
            sampling_rate: 0.01,
            data_points_processed: 0,
        }
    }

    /// Initialisiert das Modell mit initialen Daten (optional)
    pub fn initialize(
        &mut self,
        data: PyReadonlyArray2<f64>,
        params: &SDOstreamParams,
    ) -> PyResult<()> {
        let data_slice = data.as_array();
        let rows = data_slice.nrows();

        if rows == 0 || params.k == 0 {
            return Ok(());
        }

        // Konvertiere zu SDOParams (mit rho=0)
        let sdo_params = params.to_sdo_params();

        // Verwende SDO's learn-Methode
        self.sdo.learn(data, &sdo_params)?;

        // Initialisiere Streaming-spezifische Parameter
        self.fading = params.get_fading();
        self.sampling_rate = params.get_sampling_rate();

        // Initialisiere Alter für alle Observer (starten mit 1.0)
        // Hole alle Observer aus SDO (über get_active_observers_with_indices)
        let observers = self.sdo.get_active_observers_with_indices();
        for observer in observers {
            // Aktualisiere age direkt in SDO's ObserverSet
            self.sdo.update_observer_age(observer.index, 1.0);
        }

        Ok(())
    }

    /// Verarbeitet einen einzelnen Datenpunkt aus dem Stream
    pub fn learn(
        &mut self,
        point: PyReadonlyArray2<f64>,
        params: &SDOstreamParams,
    ) -> PyResult<()> {
        let point_slice = point.as_array();
        if point_slice.nrows() != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Punkt muss ein 1D-Array oder 2D-Array mit einer Zeile sein",
            ));
        }

        let point_vec: Vec<f64> = (0..point_slice.ncols())
            .map(|j| point_slice[[0, j]])
            .collect();

        // Aktualisiere Streaming-Parameter
        self.fading = params.get_fading();
        self.sampling_rate = params.get_sampling_rate();

        // Schritt 1: Finde x-nächste Observer (verwende SDO's Tree)
        let nearest_observer_indices = self.find_x_nearest_observers(&point_vec, params.x)?;

        // Schritt 2: Update Pω und Hω für alle Observer mit Exponential Moving Average
        // (Pω und Hω werden beide gefadet: Pω ← f · Pω + 1 bzw. f · Pω, Hω ← f · Hω + 1)
        self.update_observations_with_fading(&nearest_observer_indices)?;

        // Schritt 3: Sampling - ersetze Observer basierend auf normalisierter Qualität
        self.data_points_processed += 1;
        if self.should_sample() {
            self.replace_observer(&point_vec, params)?;
        }

        // Schritt 5: Invalidate Tree in SDO (wird lazy neu erstellt)
        *self.sdo.tree_active_observers.borrow_mut() = None;

        Ok(())
    }

    /// Berechnet den Outlier-Score für einen Datenpunkt (delegiert an SDO)
    pub fn predict(&self, point: PyReadonlyArray2<f64>) -> PyResult<f64> {
        // SDOstream verwendet rho=0, also alle Observer sind aktiv
        // SDO's predict verwendet bereits die aktiven Observer
        self.sdo.predict(point)
    }

    /// Gibt die Anzahl der Observer zurück
    pub fn n_observers(&self) -> usize {
        // Hole über SDO's interne Observer
        self.sdo.get_active_observers_with_indices().len()
    }

    /// Gibt x zurück (für Kompatibilität)
    #[getter]
    pub fn x(&self) -> usize {
        self.sdo.x
    }
}

impl SDOstream {
    /// Findet die x-nächsten Observer zu einem Punkt (verwendet SDO's Tree)
    fn find_x_nearest_observers(&self, point: &[f64], x: usize) -> PyResult<Vec<usize>> {
        // Verwende SDO's Tree für k-nearest neighbors
        self.sdo.ensure_tree_active_observers();

        let observers = self.sdo.get_active_observers_with_indices();
        if observers.is_empty() {
            return Ok(Vec::new());
        }

        // Verwende SDO's compute_distance für Brute-Force Suche
        // (kann später mit Tree optimiert werden)
        let mut distances: Vec<(usize, f64)> = observers
            .iter()
            .map(|obs| {
                let dist = self.sdo.compute_distance(point, &obs.data);
                (obs.index, dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        Ok(distances.into_iter().take(x).map(|(idx, _)| idx).collect())
    }

    /// Aktualisiert Pω für alle Observer mit Exponential Moving Average
    /// Pω ← f · Pω + 1 wenn ω unter den x-nächsten, sonst Pω ← f · Pω
    fn update_observations_with_fading(&mut self, nearest_indices: &[usize]) -> PyResult<()> {
        let observers = self.sdo.get_active_observers_with_indices();
        let nearest_set: std::collections::HashSet<usize> =
            nearest_indices.iter().cloned().collect();

        // Aktualisiere jeden Observer
        for observer in observers {
            let new_observations = if nearest_set.contains(&observer.index) {
                // Pω ← f · Pω + 1
                self.fading * observer.observations + 1.0
            } else {
                // Pω ← f · Pω
                self.fading * observer.observations
            };

            // Aktualisiere Observer in SDO's ObserverSet
            self.sdo
                .update_observer_observations(observer.index, new_observations);
        }

        Ok(())
    }

    /// Erhöht das Alter Hω für alle Observer
    #[allow(dead_code)]
    fn increment_ages(&mut self) {
        let observers = self.sdo.get_active_observers_with_indices();
        for observer in observers {
            let new_age = observer.age + 1.0;
            self.sdo.update_observer_age(observer.index, new_age);
        }
    }

    /// Prüft, ob ein Observer ersetzt werden sollte basierend auf Sampling-Rate
    fn should_sample(&self) -> bool {
        if self.sampling_rate <= 0.0 {
            return false;
        }
        let probability = 1.0 / self.sampling_rate;
        thread_rng().gen::<f64>() < probability
    }

    /// Ersetzt einen Observer basierend auf normalisierter Qualitätsmetrik P̃ω = Pω / Hω
    fn replace_observer(&mut self, new_point: &[f64], _params: &SDOstreamParams) -> PyResult<()> {
        let observers = self.sdo.get_active_observers_with_indices();

        if observers.is_empty() {
            return Ok(());
        }

        // Finde Observer mit niedrigster normalisierter Qualität
        let mut min_normalized_quality = f64::INFINITY;
        let mut replace_idx = 0;
        let mut old_index = 0;

        for observer in &observers {
            let age = observer.age;
            let normalized_quality = if age > 0.0 {
                observer.observations / age
            } else {
                f64::INFINITY
            };

            if normalized_quality < min_normalized_quality {
                min_normalized_quality = normalized_quality;
                replace_idx = observer.index;
                old_index = observer.index;
            }
        }

        // Erstelle neuen Observer
        let new_observer = Observer {
            data: new_point.to_vec(),
            observations: 0.0, // Neuer Observer startet mit Pω = 0
            age: 1.0,          // Neuer Observer startet mit Hω = 1
            index: old_index,
        };

        // Verwende SDO's replace_observer Methode
        self.sdo.replace_observer(replace_idx, new_observer);

        Ok(())
    }
}

impl Default for SDOstream {
    fn default() -> Self {
        Self::new()
    }
}
