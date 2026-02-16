#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use ahash::AHashMap;
    use crate::distance_matrix::{DistanceMatrix, DistanceRow};
    use crate::obs::Observer;
    use crate::utils::DistanceMetric;

    fn observer(index: usize, data: Vec<f64>) -> Observer {
        Observer {
            data,
            observations: 1.0,
            time: 0.0,
            age: 1.0,
            index,
            local_threshold: 0.0,
            label_observations: HashMap::new(),
            label_time: 0.0,
        }
    }

    #[test]
    fn distance_row_new_is_empty() {
        let row = DistanceRow::new();
        assert!(row.distances.is_empty());
    }

    #[test]
    fn distance_row_insert_keeps_sort_order() {
        let mut row = DistanceRow::new();
        row.insert(1, 3.0);
        row.insert(0, 1.0);
        row.insert(2, 2.0);
        assert_eq!(row.distances, [(0, 1.0), (2, 2.0), (1, 3.0)]);
    }

    #[test]
    fn distance_row_insert_updates_existing_index() {
        let mut row = DistanceRow::new();
        row.insert(0, 2.0);
        row.insert(0, 1.0);
        assert_eq!(row.distances.len(), 1);
        assert_eq!(row.distances[0], (0, 1.0));
    }

    #[test]
    fn distance_row_remove() {
        let mut row = DistanceRow::new();
        row.insert(0, 1.0);
        row.insert(1, 2.0);
        row.insert(2, 3.0);
        row.remove(1);
        assert_eq!(row.distances, [(0, 1.0), (2, 3.0)]);
    }

    #[test]
    fn distance_row_find_threshold_position() {
        let mut row = DistanceRow::new();
        row.insert(0, 1.0);
        row.insert(1, 2.0);
        row.insert(2, 3.0);
        assert_eq!(row.find_threshold_position(0.5), 0);
        assert_eq!(row.find_threshold_position(1.0), 0);
        assert_eq!(row.find_threshold_position(1.5), 1);
        assert_eq!(row.find_threshold_position(2.5), 2);
        assert_eq!(row.find_threshold_position(4.0), 3);
    }

    #[test]
    fn distance_matrix_new_is_empty() {
        let m = DistanceMatrix::new(DistanceMetric::Euclidean, None);
        assert!(m.is_empty());
        assert!(m.get(0).is_none());
    }

    #[test]
    fn distance_matrix_insert_one_observer() {
        let mut m = DistanceMatrix::new(DistanceMetric::Euclidean, None);
        let obs = observer(0, vec![0.0, 0.0]);
        let map: AHashMap<usize, Arc<Observer>> =
            [(0, Arc::new(obs.clone()))].into_iter().collect();
        m.insert(&obs, &map);
        assert!(!m.is_empty());
        let row = m.get(0).unwrap();
        assert!(row.distances.is_empty());
    }

    #[test]
    fn distance_matrix_insert_two_observers() {
        let mut m = DistanceMatrix::new(DistanceMetric::Euclidean, None);
        let o0 = observer(0, vec![0.0, 0.0]);
        let o1 = observer(1, vec![3.0, 4.0]);
        let map: AHashMap<usize, Arc<Observer>> = [
            (0, Arc::new(o0.clone())),
            (1, Arc::new(o1.clone())),
        ]
        .into_iter()
        .collect();
        m.insert(&o1, &map);
        let row1 = m.get(1).unwrap();
        assert_eq!(row1.distances.len(), 1);
        assert_eq!(row1.distances[0].0, 0);
        assert!((row1.distances[0].1 - 5.0).abs() < 1e-10);
    }

    #[test]
    fn distance_matrix_remove() {
        let mut m = DistanceMatrix::new(DistanceMetric::Euclidean, None);
        let o0 = observer(0, vec![0.0, 0.0]);
        let o1 = observer(1, vec![1.0, 0.0]);
        let map: AHashMap<usize, Arc<Observer>> = [
            (0, Arc::new(o0.clone())),
            (1, Arc::new(o1.clone())),
        ]
        .into_iter()
        .collect();
        m.insert(&o1, &map);
        m.remove(1);
        assert!(m.get(1).is_none());
        let row0 = m.get(0).unwrap();
        assert!(row0.distances.is_empty());
    }

    #[test]
    fn distance_matrix_rebuild() {
        let mut m = DistanceMatrix::new(DistanceMetric::Euclidean, None);
        let observers: AHashMap<usize, Arc<Observer>> = [
            (0, Arc::new(observer(0, vec![0.0, 0.0]))),
            (1, Arc::new(observer(1, vec![1.0, 0.0]))),
            (2, Arc::new(observer(2, vec![0.0, 1.0]))),
        ]
        .into_iter()
        .collect();
        m.rebuild(&observers);
        assert_eq!(m.get(0).unwrap().distances.len(), 2);
        assert_eq!(m.get(1).unwrap().distances.len(), 2);
        assert_eq!(m.get(2).unwrap().distances.len(), 2);
    }
}
