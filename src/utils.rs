use acap::chebyshev::chebyshev_distance;
use acap::coords::Coordinates;
use acap::distance::{Distance, Proximity};
use acap::euclid::{euclidean_distance, EuclideanDistance};
use acap::lp::lp_distance;
use acap::taxi::taxicab_distance;
use std::sync::Arc;

/// Lp/Minkowski Wrapper für ArcVec (analog zu acap's Euclidean, Taxicab, Chebyshev)
/// Verwendet lp_distance aus acap
#[derive(Clone, Debug)]
pub struct Lp {
    pub data: ArcVec,
    p: f64,
}

impl Lp {
    pub fn new(data: ArcVec, p: f64) -> Self {
        Self { data, p }
    }
}

impl Coordinates for Lp {
    type Value = f64;

    fn dims(&self) -> usize {
        self.data.dims()
    }

    fn coord(&self, i: usize) -> Self::Value {
        self.data.coord(i)
    }
}

impl Proximity for Lp {
    type Distance = f64;

    fn distance(&self, other: &Lp) -> Self::Distance {
        lp_distance(self.p, self, other)
    }
}

/// Wrapper für Arc<Vec<f64>> um Coordinates zu implementieren
#[derive(Clone, Debug)]
pub struct ArcVec(pub Arc<Vec<f64>>);

impl Coordinates for ArcVec {
    type Value = f64;

    fn dims(&self) -> usize {
        self.0.len()
    }

    fn coord(&self, i: usize) -> Self::Value {
        self.0[i]
    }
}

/// Observer-Struktur mit Daten, Beobachtungen und Index
#[derive(Clone, Debug)]
#[allow(dead_code)] // Felder werden in Zukunft verwendet
pub struct Observer {
    pub data: Vec<f64>,
    pub observations: f64,
    pub age: f64,
    pub index: usize,
}

/// Wrapper für Observer, der nach observations sortiert wird (für BTreeSet)
/// Sortiert nach observations (aufsteigend), dann nach index für Stabilität
#[derive(Clone, Debug)]
pub struct ObserverSorted {
    pub observer: Observer,
}

impl PartialEq for ObserverSorted {
    fn eq(&self, other: &Self) -> bool {
        self.observer.observations == other.observer.observations
            && self.observer.index == other.observer.index
    }
}

impl Eq for ObserverSorted {}

impl PartialOrd for ObserverSorted {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ObserverSorted {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Sortiere nach observations (aufsteigend), dann nach index
        self.observer
            .observations
            .partial_cmp(&other.observer.observations)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.observer.index.cmp(&other.observer.index))
    }
}

/// Datenstruktur für Observer, sortiert nach observations
/// Ermöglicht schnelles Filtern nach aktiven Observers und später schnelle Inserts/Replaces
pub struct ObserverSet {
    observers: std::collections::BTreeSet<ObserverSorted>,
}

impl ObserverSet {
    pub fn new() -> Self {
        Self {
            observers: std::collections::BTreeSet::new(),
        }
    }

    /// Fügt einen Observer hinzu
    pub fn insert(&mut self, observer: Observer) {
        self.observers.insert(ObserverSorted { observer });
    }

    /// Ersetzt einen Observer (entfernt alten, fügt neuen hinzu)
    #[allow(dead_code)] // Für zukünftige Verwendung
    pub fn replace(&mut self, old_index: usize, new_observer: Observer) -> bool {
        // Finde und entferne den alten Observer
        let to_remove: Vec<_> = self
            .observers
            .iter()
            .filter(|obs| obs.observer.index == old_index)
            .cloned()
            .collect();
        for obs in to_remove {
            self.observers.remove(&obs);
        }
        // Füge neuen Observer hinzu
        self.observers.insert(ObserverSorted {
            observer: new_observer,
        });
        true
    }

    /// Gibt die aktiven Observer zurück (obere N nach observations)
    pub fn get_active(&self, num_active: usize) -> Vec<Observer> {
        self.observers
            .iter()
            .rev() // Größte observations zuerst
            .take(num_active)
            .map(|obs| obs.observer.clone())
            .collect()
    }

    /// Gibt alle Observer zurück, sortiert nach observations (aufsteigend)
    #[allow(dead_code)] // Für zukünftige Verwendung
    pub fn all(&self) -> Vec<Observer> {
        self.observers
            .iter()
            .map(|obs| obs.observer.clone())
            .collect()
    }

    /// Gibt die Anzahl der Observer zurück
    #[allow(dead_code)] // Für zukünftige Verwendung
    pub fn len(&self) -> usize {
        self.observers.len()
    }

    /// Prüft, ob leer
    #[allow(dead_code)] // Für zukünftige Verwendung
    pub fn is_empty(&self) -> bool {
        self.observers.is_empty()
    }

    /// Aktualisiert die observations eines Observers
    #[allow(dead_code)] // Für zukünftige Verwendung
    pub fn update_observations(&mut self, index: usize, new_observations: f64) -> bool {
        // Finde den Observer
        let observer_opt = self
            .observers
            .iter()
            .find(|obs| obs.observer.index == index)
            .map(|obs| obs.observer.clone());

        if let Some(mut observer) = observer_opt {
            // Entferne alten
            self.observers.remove(&ObserverSorted {
                observer: observer.clone(),
            });
            // Aktualisiere und füge wieder hinzu
            observer.observations = new_observations;
            self.observers.insert(ObserverSorted { observer });
            true
        } else {
            false
        }
    }

    /// Aktualisiert observations und age eines Observers
    pub fn update_observer(&mut self, index: usize, new_observations: f64, new_age: f64) -> bool {
        // Finde den Observer
        let observer_opt = self
            .observers
            .iter()
            .find(|obs| obs.observer.index == index)
            .map(|obs| obs.observer.clone());

        if let Some(mut observer) = observer_opt {
            // Entferne alten
            self.observers.remove(&ObserverSorted {
                observer: observer.clone(),
            });
            // Aktualisiere und füge wieder hinzu
            observer.observations = new_observations;
            observer.age = new_age;
            self.observers.insert(ObserverSorted { observer });
            true
        } else {
            false
        }
    }

    /// Findet den Observer mit dem schlechtesten normalisierten Score (observations/age)
    /// Gibt (index, normalized_score) zurück, oder None wenn keine Observer vorhanden sind
    #[allow(dead_code)]
    pub fn find_worst_normalized_score(&self) -> Option<(usize, f64)> {
        if self.observers.is_empty() {
            return None;
        }

        let mut worst_index = 0;
        let mut worst_score = f64::INFINITY;

        for obs in self.observers.iter() {
            let normalized_score = if obs.observer.age > 0.0 {
                obs.observer.observations / obs.observer.age
            } else {
                f64::INFINITY
            };

            if normalized_score < worst_score {
                worst_score = normalized_score;
                worst_index = obs.observer.index;
            }
        }

        Some((worst_index, worst_score))
    }
}

impl Default for ObserverSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Wrapper für Vec<f64> um acap Traits zu implementieren (für Kompatibilität)
#[derive(Clone, Debug)]
pub struct Point(pub Vec<f64>);

impl Coordinates for Point {
    type Value = f64;

    fn dims(&self) -> usize {
        self.0.len()
    }

    fn coord(&self, i: usize) -> Self::Value {
        self.0[i]
    }
}

/// Berechnet die euklidische Distanz (L²) zwischen zwei Punkten
pub fn compute_euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    let point_a = Point(a.to_vec());
    let point_b = Point(b.to_vec());
    euclidean_distance(&point_a, &point_b).value()
}

/// Berechnet die Manhattan-Distanz (L¹) zwischen zwei Punkten
pub fn compute_manhattan_distance(a: &[f64], b: &[f64]) -> f64 {
    let point_a = Point(a.to_vec());
    let point_b = Point(b.to_vec());
    taxicab_distance(&point_a, &point_b)
}

/// Berechnet die Chebyshev-Distanz (L∞) zwischen zwei Punkten
pub fn compute_chebyshev_distance(a: &[f64], b: &[f64]) -> f64 {
    let point_a = Point(a.to_vec());
    let point_b = Point(b.to_vec());
    chebyshev_distance(&point_a, &point_b)
}

/// Berechnet die Minkowski-Distanz (Lᵖ) zwischen zwei Punkten
pub fn compute_minkowski_distance(a: &[f64], b: &[f64], p: f64) -> f64 {
    let point_a = Point(a.to_vec());
    let point_b = Point(b.to_vec());
    lp_distance(p, &point_a, &point_b)
}

/// Berechnet die Distanz zwischen zwei Punkten basierend auf der gewählten Metrik
pub fn compute_distance(
    a: &[f64],
    b: &[f64],
    metric: crate::sdo_impl::DistanceMetric,
    minkowski_p: Option<f64>,
) -> f64 {
    match metric {
        crate::sdo_impl::DistanceMetric::Euclidean => compute_euclidean_distance(a, b),
        crate::sdo_impl::DistanceMetric::Manhattan => compute_manhattan_distance(a, b),
        crate::sdo_impl::DistanceMetric::Chebyshev => compute_chebyshev_distance(a, b),
        crate::sdo_impl::DistanceMetric::Minkowski => {
            let p = minkowski_p.unwrap_or(3.0);
            compute_minkowski_distance(a, b, p)
        }
    }
}

/// Observer-Wrapper für verschiedene Metriken (für Tree-basierte Suche)
/// Diese Wrapper ermöglichen es, Observer direkt in Trees zu speichern
/// und bei k-NN-Suchen direkt die Observer-Objekte zurückzubekommen

#[derive(Clone, Debug)]
pub struct ObserverEuclidean(pub Observer);

#[derive(Clone, Debug)]
pub struct ObserverManhattan(pub Observer);

#[derive(Clone, Debug)]
pub struct ObserverChebyshev(pub Observer);

#[derive(Clone, Debug)]
pub struct ObserverMinkowski {
    pub observer: Observer,
    pub p: f64,
}

// Implementiere Coordinates für ObserverEuclidean
impl Coordinates for ObserverEuclidean {
    type Value = f64;
    fn dims(&self) -> usize {
        self.0.data.len()
    }
    fn coord(&self, i: usize) -> Self::Value {
        self.0.data[i]
    }
}

// Implementiere Proximity für ObserverEuclidean
impl Proximity for ObserverEuclidean {
    type Distance = EuclideanDistance<f64>;
    fn distance(&self, other: &Self) -> Self::Distance {
        euclidean_distance(self, other)
    }
}

// Implementiere Coordinates für ObserverManhattan
impl Coordinates for ObserverManhattan {
    type Value = f64;
    fn dims(&self) -> usize {
        self.0.data.len()
    }
    fn coord(&self, i: usize) -> Self::Value {
        self.0.data[i]
    }
}

// Implementiere Proximity für ObserverManhattan
impl Proximity for ObserverManhattan {
    type Distance = f64;
    fn distance(&self, other: &Self) -> Self::Distance {
        taxicab_distance(self, other)
    }
}

// Implementiere Coordinates für ObserverChebyshev
impl Coordinates for ObserverChebyshev {
    type Value = f64;
    fn dims(&self) -> usize {
        self.0.data.len()
    }
    fn coord(&self, i: usize) -> Self::Value {
        self.0.data[i]
    }
}

// Implementiere Proximity für ObserverChebyshev
impl Proximity for ObserverChebyshev {
    type Distance = f64;
    fn distance(&self, other: &Self) -> Self::Distance {
        chebyshev_distance(self, other)
    }
}

// Implementiere Coordinates für ObserverMinkowski
impl Coordinates for ObserverMinkowski {
    type Value = f64;
    fn dims(&self) -> usize {
        self.observer.data.len()
    }
    fn coord(&self, i: usize) -> Self::Value {
        self.observer.data[i]
    }
}

// Implementiere Proximity für ObserverMinkowski
impl Proximity for ObserverMinkowski {
    type Distance = f64;
    fn distance(&self, other: &Self) -> Self::Distance {
        lp_distance(self.p, self, other)
    }
}
