use acap::chebyshev::chebyshev_distance;
use acap::coords::Coordinates;
use acap::distance::{Distance, Proximity};
use acap::euclid::euclidean_distance;
use acap::lp::lp_distance;
use acap::taxi::taxicab_distance;

/// Minkowski/Lp Wrapper für Vec<f64>
#[derive(Clone, Debug)]
pub struct Minkowski {
    data: Vec<f64>,
    p: f64,
}

impl Minkowski {
    pub fn new(data: Vec<f64>, p: f64) -> Self {
        Self { data, p }
    }
}

impl Coordinates for Minkowski {
    type Value = f64;

    fn dims(&self) -> usize {
        self.data.len()
    }

    fn coord(&self, i: usize) -> Self::Value {
        self.data[i]
    }
}

impl Proximity for Minkowski {
    type Distance = f64;

    fn distance(&self, other: &Minkowski) -> Self::Distance {
        lp_distance(self.p, self, other)
    }
}

/// Observer-Struktur mit Daten, Beobachtungen und Index
#[derive(Clone, Debug)]
#[allow(dead_code)] // Felder werden in Zukunft verwendet
pub struct Observer {
    pub data: Vec<f64>,
    pub observations: f64,
    pub index: usize,
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
