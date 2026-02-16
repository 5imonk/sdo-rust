//! SIMD-accelerated Euclidean distance for batch KNN (f64).
//! Uses AVX2 on x86_64 when available; falls back to scalar otherwise.

use crate::utils::compute_euclidean_distance;

/// Computes Euclidean distances from one query point to each observer.
/// `observers` is a slice of slices (each inner slice is one point); all must have same length as `query`.
/// Returns one distance per observer.
#[inline]
pub fn compute_distances_row_euclidean(query: &[f64], observers: &[&[f64]]) -> Vec<f64> {
    if observers.is_empty() {
        return Vec::new();
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { compute_distances_row_euclidean_avx2(query, observers) };
        }
    }

    compute_distances_row_euclidean_scalar(query, observers)
}

/// Scalar fallback: one euclidean distance per observer.
fn compute_distances_row_euclidean_scalar(query: &[f64], observers: &[&[f64]]) -> Vec<f64> {
    observers
        .iter()
        .map(|obs| compute_euclidean_distance(query, obs))
        .collect()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn compute_distances_row_euclidean_avx2(query: &[f64], observers: &[&[f64]]) -> Vec<f64> {
    use std::arch::x86_64::*;

    let dim = query.len();
    let mut out = Vec::with_capacity(observers.len());

    for obs in observers {
        if obs.len() != dim {
            out.push(compute_euclidean_distance(query, obs));
            continue;
        }
        let mut sum_sq = 0.0_f64;
        let mut i = 0;

        // Process 4 components at a time (AVX: 4 x f64)
        if dim >= 4 {
            let mut acc = _mm256_setzero_pd();
            while i + 4 <= dim {
                let q = _mm256_loadu_pd(query.as_ptr().add(i));
                let o = _mm256_loadu_pd(obs.as_ptr().add(i));
                let d = _mm256_sub_pd(q, o);
                let d2 = _mm256_mul_pd(d, d);
                acc = _mm256_add_pd(acc, d2);
                i += 4;
            }
            sum_sq += horizontal_sum_pd(acc);
        }

        // Remainder
        for j in i..dim {
            let t = query[j] - obs[j];
            sum_sq += t * t;
        }

        out.push(sum_sq.sqrt());
    }

    out
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn horizontal_sum_pd(v: std::arch::x86_64::__m256d) -> f64 {
    use std::arch::x86_64::*;
    let sum = _mm256_hadd_pd(v, v);
    let hi = _mm256_extractf128_pd(sum, 1);
    let lo = _mm256_castpd256_pd128(sum);
    let both = _mm_add_pd(lo, hi);
    _mm_cvtsd_f64(both)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distances_row_euclidean_matches_scalar() {
        let query = vec![1.0, 2.0, 3.0];
        let obs1 = vec![0.0, 0.0, 0.0];
        let obs2 = vec![1.0, 2.0, 3.0];
        let obs3 = vec![2.0, 4.0, 6.0];
        let observers: Vec<&[f64]> = vec![&obs1[..], &obs2[..], &obs3[..]];
        let got = compute_distances_row_euclidean(&query, &observers);
        let expected: Vec<f64> = observers
            .iter()
            .map(|o| compute_euclidean_distance(&query, o))
            .collect();
        for (a, b) in got.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-10, "{} vs {}", a, b);
        }
    }
}
