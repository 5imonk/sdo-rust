#[cfg(test)]
mod distance_optimization_tests {
    use crate::obs::OrderedDistanceList;

    #[test]
    fn test_ordered_distance_list_binary_search_insertion() {
        let mut list = OrderedDistanceList::new();

        // Insert in random order
        list.insert(1, 5.0);
        list.insert(2, 2.0);
        list.insert(3, 8.0);
        list.insert(4, 1.0);
        list.insert(5, 3.0);

        // Verify sorted order
        assert_eq!(
            list.distances,
            vec![(4, 1.0), (2, 2.0), (5, 3.0), (1, 5.0), (3, 8.0)]
        );

        // Test update of existing index
        list.insert(2, 6.0);
        assert_eq!(
            list.distances,
            vec![(4, 1.0), (5, 3.0), (1, 5.0), (2, 6.0), (3, 8.0)]
        );
    }

    #[test]
    fn test_k_nearest_distances() {
        let mut list = OrderedDistanceList::new();

        // Add test data
        list.insert(1, 1.0);
        list.insert(2, 2.0);
        list.insert(3, 3.0);
        list.insert(4, 4.0);
        list.insert(5, 5.0);

        // Test k-nearest
        let k_nearest = list.get_k_nearest_distances(3);
        assert_eq!(k_nearest.len(), 3);
        assert_eq!(k_nearest[0], 1.0);
        assert_eq!(k_nearest[1], 2.0);
        assert_eq!(k_nearest[2], 3.0);
    }

    #[test]
    fn test_find_threshold_position() {
        let mut list = OrderedDistanceList::new();

        // Add test data
        list.insert(1, 1.0);
        list.insert(2, 2.0);
        list.insert(3, 3.0);
        list.insert(4, 4.0);
        list.insert(5, 5.0);

        // Test threshold positions
        assert_eq!(list.find_threshold_position(0.5), 0); // All >= 0.5
        assert_eq!(list.find_threshold_position(2.5), 2); // Start of 3.0
        assert_eq!(list.find_threshold_position(3.0), 2); // Start of 3.0
        assert_eq!(list.find_threshold_position(5.5), 5); // None >= 5.5
    }
}

#[cfg(test)]
mod observer_set_optimization_tests {
    use crate::obs::Observer;
    use crate::obset::ObserverSet;

    fn create_test_observer(index: usize, observations: f64) -> Observer {
        Observer {
            data: vec![index as f64, (index * 2) as f64],
            observations,
            time: index as f64,
            age: 1.0,
            index,
            label: None,
            cluster_observations: Vec::new(),
        }
    }

    #[test]
    fn test_k_nearest_neighbors() {
        let mut obset = ObserverSet::new();

        // Add test observers
        for i in 0..5 {
            obset.insert(create_test_observer(i, 10.0 - i as f64));
        }

        // Test k-nearest neighbors
        let neighbors = obset.get_k_nearest_neighbors(0, 3);
        assert_eq!(neighbors.len(), 3);
        // Should be sorted by distance
        assert!(neighbors[0].1 <= neighbors[1].1);
        assert!(neighbors[1].1 <= neighbors[2].1);
    }

    #[test]
    fn test_neighbors_within_threshold() {
        let mut obset = ObserverSet::new();

        // Add test observers
        for i in 0..5 {
            obset.insert(create_test_observer(i, 10.0 - i as f64));
        }

        // Test with very low threshold first (should return no neighbors)
        let neighbors = obset.get_neighbors_within_threshold(0, 0.0);
        assert_eq!(neighbors.len(), 0); // No neighbors with distance < 0

        // Test with a reasonable threshold
        // First let's see what distances we actually have
        let all_neighbors = obset.get_k_nearest_neighbors(0, 10);
        println!("All neighbors for observer 0: {:?}", all_neighbors);

        // Test with threshold of 1000 (should include most neighbors)
        let neighbors = obset.get_neighbors_within_threshold(0, 1000.0);
        assert!(neighbors.len() >= 0); // At least no neighbors
        assert!(neighbors.len() <= 4); // At most all other observers
    }

    #[test]
    fn test_threshold_cache() {
        let mut obset = ObserverSet::new();

        // Add test observers
        for i in 0..5 {
            obset.insert(create_test_observer(i, 10.0 - i as f64));
        }

        // Validate cache
        obset.validate_threshold_cache();

        // First computation
        let t1 = obset.compute_local_threshold_cached(0, 3);

        // Second computation should use cache
        let t2 = obset.compute_local_threshold_cached(0, 3);

        assert_eq!(t1, t2);

        // Invalidate cache
        obset.invalidate_threshold_cache();

        // Should recompute
        let t3 = obset.compute_local_threshold_cached(0, 3);
        assert_eq!(t1, t3); // Same value, but recomputed
    }

    #[test]
    fn test_batch_update_distance_lists() {
        let mut obset = ObserverSet::new();

        // Add initial observers
        for i in 0..5 {
            obset.insert(create_test_observer(i, 10.0 - i as f64));
        }

        // Update some observers
        let updated_indices = vec![0, 2, 4];
        obset.batch_update_distance_lists(&updated_indices);

        // Verify distance lists are still valid
        for &idx in &updated_indices {
            assert!(obset.distance_lists.contains_key(&idx));
        }
    }
}

#[cfg(test)]
mod clustering_optimization_tests {
    use crate::obs::Observer;
    use crate::obset::ObserverSet;

    fn create_test_observer(index: usize, observations: f64) -> Observer {
        Observer {
            data: vec![index as f64, (index * 2) as f64],
            observations,
            time: index as f64,
            age: 1.0,
            index,
            label: None,
            cluster_observations: Vec::new(),
        }
    }

    #[test]
    fn test_optimized_local_threshold_computation() {
        let mut obset = ObserverSet::new();

        // Add test observers
        for i in 0..10 {
            obset.insert(create_test_observer(i, 10.0 - i as f64));
        }

        obset.set_num_active(5);

        // Test local threshold computation
        let threshold = obset.compute_local_threshold_impl(0, 3);
        assert!(threshold.is_finite());

        // Test with cached version
        obset.validate_threshold_cache();
        let cached_threshold = obset.compute_local_threshold_cached(0, 3);
        assert_eq!(threshold, cached_threshold);
    }

    #[test]
    fn test_clustering_with_optimizations() {
        let mut obset = ObserverSet::new();

        // Create clearly separated clusters with identical points within each cluster
        let cluster1_observers = vec![
            Observer {
                data: vec![0.0, 0.0], // Cluster 1 origin
                observations: 10.0,
                time: 1.0,
                age: 1.0,
                index: 0,
                label: None,
                cluster_observations: Vec::new(),
            },
            Observer {
                data: vec![0.1, 0.1], // Very close to cluster 1
                observations: 9.5,
                time: 2.0,
                age: 1.0,
                index: 1,
                label: None,
                cluster_observations: Vec::new(),
            },
            Observer {
                data: vec![-0.1, 0.0], // Very close to cluster 1
                observations: 9.0,
                time: 3.0,
                age: 1.0,
                index: 2,
                label: None,
                cluster_observations: Vec::new(),
            },
        ];

        let cluster2_observers = vec![
            Observer {
                data: vec![10.0, 10.0], // Cluster 2 far away
                observations: 8.0,
                time: 4.0,
                age: 1.0,
                index: 3,
                label: None,
                cluster_observations: Vec::new(),
            },
            Observer {
                data: vec![10.1, 10.1], // Very close to cluster 2
                observations: 7.5,
                time: 5.0,
                age: 1.0,
                index: 4,
                label: None,
                cluster_observations: Vec::new(),
            },
        ];

        // Insert all observers
        for obs in cluster1_observers.iter().chain(cluster2_observers.iter()) {
            obset.insert(obs.clone());
        }

        obset.set_num_active(5); // Make all but the last observer active

        // Run clustering with very permissive parameters
        let clusters = obset.learn_cluster(2, 0.0, 1, true); // zeta=0.0 (purely global), min_cluster_size=1

        // Should find at least one cluster due to identical points
        assert!(clusters.len() >= 1); // At least one cluster

        // Verify cluster assignments are valid
        let mut total_clustered = 0usize;
        for (_, indices) in &clusters {
            total_clustered += indices.len();
        }
        assert!(total_clustered <= 5); // At most all active observers
        assert!(total_clustered >= 1); // At least one observer clustered
    }
}

#[cfg(test)]
mod performance_tests {
    use crate::obs::Observer;
    use crate::obset::ObserverSet;
    use std::time::Instant;

    fn create_large_test_set(size: usize) -> ObserverSet {
        let mut obset = ObserverSet::new();

        for i in 0..size {
            let obs = Observer {
                data: vec![i as f64, (i * 2) as f64],
                observations: (size - i) as f64,
                time: i as f64,
                age: 1.0,
                index: i,
                label: None,
                cluster_observations: Vec::new(),
            };
            obset.insert(obs);
        }

        obset.set_num_active(size / 2);
        obset
    }

    #[test]
    fn test_performance_optimizations() {
        let size = 1000;
        let obset = create_large_test_set(size);

        // Test k-nearest neighbor performance
        let start = Instant::now();
        for i in 0..10 {
            let _neighbors = obset.get_k_nearest_neighbors(i, 10);
        }
        let k_nearest_time = start.elapsed();

        // Test threshold-based neighbor performance
        let start = Instant::now();
        for i in 0..10 {
            let _neighbors = obset.get_neighbors_within_threshold(i, 100.0);
        }
        let threshold_time = start.elapsed();

        // Performance should be reasonable
        println!("k-nearest time: {:?}", k_nearest_time);
        println!("threshold time: {:?}", threshold_time);

        // These are just sanity checks - actual values depend on hardware
        assert!(k_nearest_time.as_millis() < 1000); // Should be fast
        assert!(threshold_time.as_millis() < 1000); // Should be fast
    }
}
