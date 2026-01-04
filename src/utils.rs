/// Distanzfunktion für SDO
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum DistanceMetric {
    Euclidean = 0,
    Manhattan = 1,
    Chebyshev = 2,
    Minkowski = 3,
}

/// Berechnet die euklidische Distanz (L²) zwischen zwei Punkten
pub fn compute_euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Berechnet die Manhattan-Distanz (L¹) zwischen zwei Punkten
pub fn compute_manhattan_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Berechnet die Chebyshev-Distanz (L∞) zwischen zwei Punkten
pub fn compute_chebyshev_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Berechnet die Minkowski-Distanz (Lᵖ) zwischen zwei Punkten
pub fn compute_minkowski_distance(a: &[f64], b: &[f64], p: f64) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs().powf(p))
        .sum::<f64>()
        .powf(1.0 / p)
}

/// Berechnet die Distanz zwischen zwei Punkten basierend auf der gewählten Metrik
pub fn compute_distance(
    a: &[f64],
    b: &[f64],
    metric: DistanceMetric,
    minkowski_p: Option<f64>,
) -> f64 {
    match metric {
        DistanceMetric::Euclidean => compute_euclidean_distance(a, b),
        DistanceMetric::Manhattan => compute_manhattan_distance(a, b),
        DistanceMetric::Chebyshev => compute_chebyshev_distance(a, b),
        DistanceMetric::Minkowski => {
            let p = minkowski_p.unwrap_or(3.0);
            compute_minkowski_distance(a, b, p)
        }
    }
}
