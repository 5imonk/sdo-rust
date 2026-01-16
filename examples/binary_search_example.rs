// Binary Search Implementation for OrderedDistanceList
// This is already integrated into src/obs.rs above

pub fn binary_search_insert_example() {
    // Example of binary search insertion
    let mut distances = vec![(0, 1.0), (1, 2.0), (2, 4.0), (3, 5.0)];
    let new_distance = 3.0;

    // Binary search to find insertion point
    let mut left = 0;
    let mut right = distances.len();

    while left < right {
        let mid = left + (right - left) / 2;
        match distances[mid]
            .1
            .partial_cmp(&new_distance)
            .unwrap_or(std::cmp::Ordering::Equal)
        {
            std::cmp::Ordering::Less => left = mid + 1,
            std::cmp::Ordering::Greater | std::cmp::Ordering::Equal => right = mid,
        }
    }

    distances.insert(left, (4, new_distance));
    // Result: [(0, 1.0), (1, 2.0), (4, 3.0), (2, 4.0), (3, 5.0)]
}
