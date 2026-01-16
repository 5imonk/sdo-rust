// Optimized Threshold Computation with Caching
// This shows the caching strategy for local thresholds

use std::collections::HashMap;

pub struct ThresholdCache {
    cache: HashMap<(usize, usize), f64>, // (observer_index, chi) -> threshold
    cache_valid: bool,
}

impl ThresholdCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            cache_valid: true,
        }
    }

    pub fn get_or_compute<F>(&mut self, observer_index: usize, chi: usize, compute_fn: F) -> f64
    where
        F: FnOnce() -> f64,
    {
        let cache_key = (observer_index, chi);

        if self.cache_valid {
            if let Some(&cached_threshold) = self.cache.get(&cache_key) {
                return cached_threshold;
            }
        }

        let threshold = compute_fn();
        self.cache.insert(cache_key, threshold);
        threshold
    }

    pub fn invalidate(&mut self) {
        self.cache.clear();
        self.cache_valid = false;
    }

    pub fn validate(&mut self) {
        self.cache_valid = true;
    }
}

// Example usage
pub fn threshold_computation_example() {
    let mut cache = ThresholdCache::new();

    // First computation - computes and caches
    let threshold1 = cache.get_or_compute(5, 3, || {
        println!("Computing threshold for observer 5 with chi=3");
        2.5 // Simulated computation result
    });

    // Second computation - uses cache
    let threshold2 = cache.get_or_compute(5, 3, || {
        println!("This should not be printed!");
        2.5
    });

    assert_eq!(threshold1, threshold2);
}
