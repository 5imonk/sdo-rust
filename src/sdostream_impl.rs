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
    pub t_fading: f64, // T-Parameter für fading: f = exp(-T_fading^-1)
    #[pyo3(get, set)]
    pub t_sampling: f64, // T-Parameter für Sampling-Rate (durchschnittliche Zeit zwischen Ersetzungen)
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
    #[pyo3(signature = (k, x, t_fading, t_sampling, distance = "euclidean".to_string(), minkowski_p = None, rho = 0.1))]
    pub fn new(
        k: usize,
        x: usize,
        t_fading: f64,
        t_sampling: f64,
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
        }
    }
}

impl SDOstreamParams {
    /// Berechnet den Fading-Parameter f = exp(-T_fading^-1)
    fn get_fading(&self) -> f64 {
        (-1.0 / self.t_fading).exp()
    }

    /// Berechnet die Sampling-Rate T_k = -k · ln(f) (für Kompatibilität, nicht mehr verwendet)
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
    sampling_rate: f64,           // T_k = -k · ln(f) (für Kompatibilität, nicht mehr verwendet)
    data_points_processed: usize, // Zähler für Sampling
    k: usize,                     // Anzahl der Observer (Modellgröße)
    t_fading: f64,                // T-Parameter für fading: f = exp(-T_fading^-1)
    t_sampling: f64, // T-Parameter für Sampling-Rate (durchschnittliche Zeit zwischen Ersetzungen)
    rho: f64,        // Rho-Parameter (für num_active Berechnung)
    use_explicit_time: bool, // Wenn true, erwartet learn() time-Parameter; sonst auto-increment
    last_replacement_time: f64, // Zeit der letzten Prüfung/Ersetzung (für Lazy Replacement)
    pending_replacements: usize, // Anzahl der ausstehenden Ersetzungen (wenn num_replacements > 1)
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
            t_fading: 100.0,
            t_sampling: 100.0,
            rho: 0.1,
            use_explicit_time: false,   // Default: auto-increment
            last_replacement_time: 0.0, // Startzeit für Lazy Replacement
            pending_replacements: 0,    // Keine ausstehenden Ersetzungen
        }
    }

    /// Initialisiert das Modell mit Parametern
    /// Kann optional mit einem Datensatz oder zufälligen Punkten initialisiert werden
    /// Wenn time nicht angegeben, wird time=0 verwendet
    ///
    /// Verwendung:
    /// - initialize(params) - nur Parameter setzen (auto-increment Zeit)
    /// - initialize(params, data=data, time=time) - mit Datensatz
    ///   - Wenn time angegeben: erwartet learn() time-Parameter für jeden Punkt
    ///   - Wenn time nicht angegeben: verwendet auto-increment (Zähler)
    /// - initialize(params, dimension, time=time) - mit zufälligen Punkten
    #[pyo3(signature = (params, dimension = None, *, data = None, time = None))]
    pub fn initialize(
        &mut self,
        params: &SDOstreamParams,
        dimension: Option<usize>,
        data: Option<PyReadonlyArray2<f64>>,
        time: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<()> {
        // Setze Parameter
        self.k = params.k;
        self.t_fading = params.t_fading;
        self.t_sampling = params.t_sampling;
        self.rho = params.rho;

        // Initialisiere Streaming-spezifische Parameter
        self.fading = params.get_fading();
        self.sampling_rate = params.get_sampling_rate();

        // Entscheide Zeit-Strategie: Wenn time bei Initialisierung angegeben, erwarte time bei learn()
        // Sonst verwende auto-increment basierend auf data_points_processed
        self.use_explicit_time = time.is_some();

        // Bestimme Startzeit für exponentielles Sampling
        let start_time = if let Some(time_array) = &time {
            let time_slice = time_array.as_array();
            if time_slice.len() != 1 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Zeit muss ein 1D-Array mit einem Wert sein",
                ));
            }
            time_slice[[0]]
        } else {
            0.0
        };

        // Initialisiere Lazy Replacement: Startzeit setzen
        self.last_replacement_time = start_time;
        self.pending_replacements = 0; // Keine ausstehenden Ersetzungen bei Initialisierung

        // Initialisiere SDO mit Parametern
        let sdo_params = params.to_sdo_params();
        self.sdo.initialize(&sdo_params)?;

        // Prüfe, ob sowohl data als auch dimension gegeben sind (Fehler)
        if data.is_some() && dimension.is_some() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Entweder 'data' oder 'dimension' muss angegeben werden, nicht beide",
            ));
        }

        // Prüfe, ob data oder dimension gegeben ist
        if let Some(data_array) = data {
            // Verwende SDO's learn-Methode für das initiale Training
            // time wird direkt an learn() übergeben (wird auf t0 gesetzt wenn angegeben)
            self.sdo.learn(data_array, time)?;
        } else if let Some(dim) = dimension {
            // Initialisierung mit zufälligen normalverteilten Punkten
            if dim == 0 || self.k == 0 {
                return Ok(());
            }

            // Bestimme Zeit für alle Punkte
            let t0 = if let Some(time_array) = &time {
                let time_slice = time_array.as_array();
                if time_slice.len() != 1 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Zeit muss ein 1D-Array mit einem Wert sein",
                    ));
                }
                time_slice[[0]]
            } else {
                0.0 // Default: time = 0
            };

            // Generiere k normalverteilte Punkte mit Box-Muller Transformation
            let mut random_data_normal: Vec<Vec<f64>> = Vec::new();
            let mut rng = thread_rng();
            for _ in 0..self.k {
                let mut point: Vec<f64> = Vec::new();
                let mut i = 0;
                while i < dim {
                    if i + 1 < dim {
                        // Box-Muller: generiere zwei normalverteilte Werte
                        let u1: f64 = rng.gen();
                        let u2: f64 = rng.gen();
                        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                        let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).sin();
                        point.push(z0);
                        point.push(z1);
                        i += 2;
                    } else {
                        // Letzte Dimension: generiere einen normalverteilten Wert
                        let u1: f64 = rng.gen();
                        let u2: f64 = rng.gen();
                        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                        point.push(z0);
                        i += 1;
                    }
                }
                random_data_normal.push(point);
            }

            // Hole distance_metric und minkowski_p aus ObserverSet (wurden oben gesetzt)
            let distance_metric = self.sdo.observers.get_distance_metric();
            let minkowski_p = self.sdo.observers.get_minkowski_p();

            // Erstelle ObserverSet mit zufälligen Punkten
            self.sdo.observers = crate::observer::ObserverSet::new();
            self.sdo
                .observers
                .set_tree_params(distance_metric, minkowski_p);

            for (idx, point_data) in random_data_normal.iter().enumerate() {
                let observer = crate::observer::Observer {
                    data: point_data.clone(),
                    observations: 1.0, // Start mit 1 observation
                    time: t0,
                    age: 1.0,
                    index: idx,
                    label: None,
                    cluster_observations: Vec::new(),
                };
                self.sdo.observers.insert(observer);
            }

            // Setze num_active basierend auf rho
            self.sdo.observers.set_num_active(
                ((self.sdo.observers.len() as f64) * (1.0 - self.rho)).ceil() as usize,
            );
        }
        // Wenn weder data noch dimension gegeben ist, bleibt ObserverSet leer (nur Parameter gesetzt)

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

        // Bestimme Zeit basierend auf Initialisierungs-Strategie
        let current_time = if self.use_explicit_time {
            // Erwarte time-Parameter wenn bei Initialisierung time angegeben wurde
            if let Some(time_array) = time {
                let time_slice = time_array.as_array();
                if time_slice.len() != 1 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Zeit muss ein 1D-Array mit einem Wert sein",
                    ));
                }
                time_slice[[0]]
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "time-Parameter ist erforderlich (wurde bei Initialisierung mit time initialisiert)",
                ));
            }
        } else {
            // Auto-increment: verwende data_points_processed als Zeit
            // Ignoriere time-Parameter wenn gegeben (aber nicht erforderlich)
            (self.data_points_processed + 1) as f64
        };

        // Schritt 1: Finde x-nächste Observer (verwende SDO's Tree)
        let x = self.sdo.x;
        let nearest_observer_indices = self
            .sdo
            .observers
            .search_k_nearest_indices(&point_vec, x, false);

        // Schritt 2: Update Pω und Hω für alle Observer mit zeitbasiertem Exponential Moving Average
        // Hω ← f^(ti - ti-1) · Hω + 1, Pω ← f^(ti - ti-1) · Pω + 1 (wenn nearest) bzw. f^(ti - ti-1) · Pω
        self.sdo.observers.update_observations_with_fading(
            &nearest_observer_indices,
            self.fading,
            current_time,
        );

        // Increment data_points_processed für auto-increment Modus
        self.data_points_processed += 1;

        // Schritt 3: Sampling - Lazy Replacement basierend auf verstrichener Zeit (Poisson-basiert)
        self.check_and_replace(current_time, &point_vec)?;

        Ok(())
    }

    /// Berechnet den Outlier-Score für einen Datenpunkt (delegiert an SDO)
    pub fn predict(&self, point: PyReadonlyArray2<f64>) -> PyResult<f64> {
        // SDOstream verwendet rho, um aktive Observer zu bestimmen
        // SDO's predict verwendet bereits die aktiven Observer basierend auf rho
        self.sdo.predict(point)
    }

    /// Gibt x zurück (Anzahl der nächsten Nachbarn)
    #[getter]
    pub fn x(&self) -> usize {
        self.sdo.x
    }
}

impl SDOstream {
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
    fn check_and_replace(&mut self, current_time: f64, new_point: &[f64]) -> PyResult<()> {
        let elapsed = current_time - self.last_replacement_time;

        if elapsed <= 0.0 {
            return Ok(()); // Keine Zeit vergangen oder Zeit geht zurück
        }

        // Erwartete Anzahl von Ersetzungen in elapsed Zeit: λ_events = elapsed / T_sampling
        let lambda_events = elapsed / self.t_sampling;

        // Simuliere Poisson-Anzahl von Ersetzungen
        let num_replacements = self.sample_poisson(lambda_events);

        // Berücksichtige ausstehende Ersetzungen von vorherigen Aufrufen
        let total_replacements = num_replacements + self.pending_replacements;

        // Führe nur eine Ersetzung durch (auch wenn total_replacements > 1)
        if total_replacements > 0 {
            self.replace_observer(new_point, current_time)?;
            self.last_replacement_time = current_time;

            // Speichere verbleibende Ersetzungen für nächste Aufrufe
            self.pending_replacements = total_replacements - 1;
        }
        // Wenn total_replacements == 0: last_replacement_time bleibt unverändert
        // (Zeit wird beim nächsten Aufruf akkumuliert)

        Ok(())
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
            observations: 1.0,  // Neuer Observer startet mit Pω = 1
            time: current_time, // Setze time auf aktuelle Zeit
            age: 1.0,           // Neuer Observer startet mit Hω = 1
            index: self.data_points_processed,
            label: None,
            cluster_observations: Vec::new(),
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
