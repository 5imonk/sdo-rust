// Efficient Neighbor Finding Implementation
// This shows the optimized neighbor finding methods

pub fn efficient_neighbor_finding_example() {
    // Example of k-nearest neighbors using sorted distance list
    let distances = vec![(1, 0.5), (2, 1.2), (3, 1.8), (4, 2.5), (5, 3.1)];
    let k = 3;

    // O(k) operation since list is already sorted
    let k_nearest: Vec<(usize, f64)> = distances
        .iter()
        .take(k)
        .map(|&(idx, dist)| (idx, dist))
        .collect();

    // Example of threshold-based neighbor finding with binary search
    let threshold = 2.0;

    // Binary search to find first distance >= threshold
    let mut left = 0;
    let mut right = distances.len();

    while left < right {
        let mid = left + (right - left) / 2;
        match distances[mid]
            .1
            .partial_cmp(&threshold)
            .unwrap_or(std::cmp::Ordering::Equal)
        {
            std::cmp::Ordering::Less => left = mid + 1,
            std::cmp::Ordering::Greater | std::cmp::Ordering::Equal => right = mid,
        }
    }

    // All neighbors with distance < threshold
    let neighbors_within_threshold: Vec<(usize, f64)> = distances
        .iter()
        .take(left) // All elements before the threshold position
        .map(|&(idx, dist)| (idx, dist))
        .collect();
}
