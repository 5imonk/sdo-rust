//! Sparse distance matrix: one row per observer, each row sorted by distance.
//! Symmetric (d(i,j) = d(j,i)); stored as rows for efficient threshold lookups.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

use crate::obs::Observer;
use crate::utils::{compute_distance, DistanceMetric};

/// One row of the distance matrix: (neighbor_index, distance) pairs sorted by distance ascending.
#[derive(Clone, Debug)]
pub(crate) struct DistanceRow {
    /// Pairs (target_index, distance) sorted by distance ascending.
    pub(crate) distances: Vec<(usize, f64)>,
}

impl DistanceRow {
    pub(crate) fn new() -> Self {
        Self {
            distances: Vec::new(),
        }
    }

    /// Insert or update one entry, keeping sort order. O(n) scan + insert.
    pub(crate) fn insert(&mut self, target_index: usize, distance: f64) {
        if let Some(pos) = self
            .distances
            .iter()
            .position(|(idx, _)| *idx == target_index)
        {
            self.distances.remove(pos);
        }
        let insert_pos = self.binary_search_insert_position(distance);
        self.distances.insert(insert_pos, (target_index, distance));
    }

    fn binary_search_insert_position(&self, distance: f64) -> usize {
        let mut left = 0;
        let mut right = self.distances.len();
        while left < right {
            let mid = left + (right - left) / 2;
            match self.distances[mid]
                .1
                .partial_cmp(&distance)
                .unwrap_or(Ordering::Equal)
            {
                Ordering::Less => left = mid + 1,
                Ordering::Greater | Ordering::Equal => right = mid,
            }
        }
        left
    }

    /// Index of first element with distance >= threshold.
    pub(crate) fn find_threshold_position(&self, threshold: f64) -> usize {
        let mut left = 0;
        let mut right = self.distances.len();
        while left < right {
            let mid = left + (right - left) / 2;
            match self.distances[mid]
                .1
                .partial_cmp(&threshold)
                .unwrap_or(Ordering::Equal)
            {
                Ordering::Less => left = mid + 1,
                Ordering::Greater | Ordering::Equal => right = mid,
            }
        }
        left
    }

    /// Remove the entry for `target_index`.
    pub(crate) fn remove(&mut self, target_index: usize) {
        self.distances.retain(|(idx, _)| *idx != target_index);
    }
}

/// Sparse symmetric distance matrix: one row per observer index.
#[derive(Clone)]
pub(crate) struct DistanceMatrix {
    rows: HashMap<usize, DistanceRow>,
    distance_metric: DistanceMetric,
    minkowski_p: Option<f64>,
}

impl DistanceMatrix {
    pub(crate) fn new(distance_metric: DistanceMetric, minkowski_p: Option<f64>) -> Self {
        Self {
            rows: HashMap::new(),
            distance_metric,
            minkowski_p,
        }
    }

    /// Add a new observer: new row for its index and add column to all existing rows.
    pub(crate) fn insert(
        &mut self,
        new_observer: &Observer,
        observers: &HashMap<usize, Arc<Observer>>,
    ) {
        let new_index = new_observer.index;
        let new_data = new_observer.data.as_slice();
        for (&existing_index, arc) in observers {
            if existing_index == new_index {
                continue;
            }
            let existing_data = arc.data.as_slice();
            let d = compute_distance(
                existing_data,
                new_data,
                self.distance_metric,
                self.minkowski_p,
            );
            self.rows
                .entry(existing_index)
                .or_insert_with(DistanceRow::new)
                .insert(new_index, d);
        }
        let mut new_row = DistanceRow::new();
        for (&existing_index, arc) in observers {
            if existing_index == new_index {
                continue;
            }
            let existing_data = arc.data.as_slice();
            let d = compute_distance(
                new_data,
                existing_data,
                self.distance_metric,
                self.minkowski_p,
            );
            new_row.insert(existing_index, d);
        }
        self.rows.insert(new_index, new_row);
    }

    /// Remove observer at `index`: drop its row and remove it from all other rows.
    pub(crate) fn remove(&mut self, index: usize) {
        self.rows.remove(&index);
        for row in self.rows.values_mut() {
            row.remove(index);
        }
    }

    pub(crate) fn get(&self, index: usize) -> Option<&DistanceRow> {
        self.rows.get(&index)
    }

    pub(crate) fn get_mut(&mut self, index: usize) -> Option<&mut DistanceRow> {
        self.rows.get_mut(&index)
    }

    #[allow(dead_code)]
    pub(crate) fn clear(&mut self) {
        self.rows.clear();
    }

    /// Rebuild from scratch: one row per observer, distances between all pairs.
    pub(crate) fn rebuild(&mut self, observers: &HashMap<usize, Arc<Observer>>) {
        self.rows.clear();
        let indices: Vec<usize> = observers.keys().copied().collect();
        for &i in &indices {
            let data_i = observers[&i].data.as_slice();
            let mut row = DistanceRow::new();
            for &j in &indices {
                if i != j {
                    let data_j = observers[&j].data.as_slice();
                    let d =
                        compute_distance(data_i, data_j, self.distance_metric, self.minkowski_p);
                    row.insert(j, d);
                }
            }
            self.rows.insert(i, row);
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}
