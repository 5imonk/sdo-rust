use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::{thread_rng, Rng};
use std::f64;

use crate::observer::Observer;
use crate::sdo_impl::{SDOParams, SDO};

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
    pub rho: f64, // Anteil der inaktiven Observer (0.0 = alle aktiv, 1.0 = alle inaktiv)
}

#[pymethods]
impl SDOstreamParams {
    #[new]
    #[pyo3(signature = (k, x, t, distance = "euclidean".to_string(), minkowski_p = None, rho = 0.1))]
    pub fn new(
        k: usize,
        x: usize,
        t: f64,
        distance: String,
        minkowski_p: Option<f64>,
        rho: f64,
    ) -> Self {
        Self {
            k,
            x,
            t,
            distance,
            minkowski_p,
            rho,
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

    /// Konvertiert zu SDOParams (mit rho für inaktive Observer)
    fn to_sdo_params(&self) -> SDOParams {
        SDOParams {
            k: self.k,
            x: self.x,
            rho: self.rho, // In SDOstream werden Observer inaktiv statt entfernt
            distance: self.distance.clone(),
            minkowski_p: self.minkowski_p,
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
    k: usize,                     // Anzahl der Observer (Modellgröße)
    t: f64,                       // T-Parameter für fading
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
            k: 100,
            t: 100.0,
        }
    }

    /// Initialisiert das Modell mit Parametern und einem leeren ObserverSet
    pub fn initialize(&mut self, params: &SDOstreamParams) -> PyResult<()> {
        self.k = params.k;
        self.t = params.t;

        // Initialisiere Streaming-spezifische Parameter
        self.fading = params.get_fading();
        self.sampling_rate = params.get_sampling_rate();

        // Initialisiere SDO mit Parametern
        let sdo_params = params.to_sdo_params();
        self.sdo.initialize(&sdo_params)?;

        Ok(())
    }

    /// Verarbeitet einen einzelnen Datenpunkt aus dem Stream
    #[pyo3(signature = (point, *, time = None))]
    pub fn learn(
        &mut self,
        point: PyReadonlyArray2<f64>,
        time: Option<PyReadonlyArray1<f64>>,
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

        // Wenn time nicht angegeben, verwende auto-increment basierend auf data_points_processed
        let time_was_provided = time.is_some();
        let current_time = if let Some(time_array) = time {
            let time_slice = time_array.as_array();
            if time_slice.len() != 1 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Zeit muss ein 1D-Array mit einem Wert sein",
                ));
            }
            time_slice[[0]]
        } else {
            // Auto-increment: verwende data_points_processed als Zeit
            self.data_points_processed as f64
        };

        // Schritt 1: Finde x-nächste Observer (verwende SDO's Tree)
        let x = self.sdo.x;
        let nearest_observer_indices = self.find_x_nearest_observers(&point_vec, x)?;

        // Schritt 2: Update Pω und Hω für alle Observer mit zeitbasiertem Exponential Moving Average
        // Hω ← f^(ti - ti-1) · Hω + 1, Pω ← f^(ti - ti-1) · Pω + 1 (wenn nearest) bzw. f^(ti - ti-1) · Pω
        self.sdo.observers.update_observations_with_fading(
            &nearest_observer_indices,
            self.fading,
            current_time,
        );

        // Schritt 3: Sampling - ersetze Observer basierend auf normalisierter Qualität
        // Increment data_points_processed nur wenn time nicht explizit angegeben wurde
        if !time_was_provided {
            self.data_points_processed += 1;
        }
        if self.should_sample() {
            self.replace_observer(&point_vec, current_time)?;
        }

        Ok(())
    }

    /// Berechnet den Outlier-Score für einen Datenpunkt (delegiert an SDO)
    pub fn predict(&self, point: PyReadonlyArray2<f64>) -> PyResult<f64> {
        // SDOstream verwendet rho, um aktive Observer zu bestimmen
        // SDO's predict verwendet bereits die aktiven Observer basierend auf rho
        self.sdo.predict(point)
    }

    /// Gibt x zurück (für Kompatibilität)
    #[getter]
    pub fn x(&self) -> usize {
        self.sdo.x
    }
}

impl SDOstream {
    fn find_x_nearest_observers(&self, point: &[f64], x: usize) -> PyResult<Vec<usize>> {
        let observers = self.sdo.observers.get_observers(true);
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

    /// Prüft, ob ein Observer ersetzt werden sollte basierend auf Sampling-Rate
    fn should_sample(&self) -> bool {
        if self.sampling_rate <= 0.0 {
            return false;
        }
        let probability = 1.0 / self.sampling_rate;
        thread_rng().gen::<f64>() < probability
    }

    /// Ersetzt einen Observer basierend auf normalisierter Qualitätsmetrik P̃ω = Pω / Hω
    fn replace_observer(&mut self, new_point: &[f64], current_time: f64) -> PyResult<()> {
        // Verwende die optimierte find_k_worst_normalized_scores Methode - O(1) statt O(n)
        let worst_scores = self.sdo.observers.find_k_worst_normalized_scores(Some(1));
        let (replace_idx, _score) = match worst_scores.first() {
            Some((idx, score)) => (*idx, *score),
            None => return Ok(()), // Keine Observer vorhanden
        };

        // Erstelle neuen Observer
        // Für neue Observer: time sollte auf die aktuelle Zeit gesetzt werden
        // Da wir hier keine Zeit haben, verwenden wir 0.0 (wird beim nächsten Update korrigiert)
        let new_observer = Observer {
            data: new_point.to_vec(),
            observations: 1.0,  // Neuer Observer startet mit Pω = 0
            time: current_time, // Setze time auf aktuelle Zeit
            age: 1.0,           // Neuer Observer startet mit Hω = 1
            index: replace_idx, // Verwende den Index des ersetzten Observers
            label: None,
        };

        // Verwende SDO's replace_observer Methode - O(log n)
        self.sdo.replace_observer(replace_idx, new_observer);

        Ok(())
    }
}

impl Default for SDOstream {
    fn default() -> Self {
        Self::new()
    }
}
