use crate::obset::ObserverSet;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obs::Observer;
    use crate::utils::DistanceMetric;
    use std::collections::HashMap;

    fn create_test_observers() -> (ObserverSet, Vec<usize>) {
        let mut obs_set = ObserverSet::new(DistanceMetric::Euclidean, None);

        // Create uniform distribution (single cluster)
        let uniform_observers = vec![
            Observer {
                data: vec![0.0, 0.0],
                observations: 10.0,
                time: 1.0,
                age: 1.0,
                index: 0,
                local_threshold: 0.0,
                label_observations: HashMap::from([(0, 1.0)]),
            },
            Observer {
                data: vec![0.1, 0.1],
                observations: 9.0,
                time: 1.0,
                age: 1.0,
                index: 1,
                local_threshold: 0.0,
                label_observations: HashMap::from([(0, 1.0)]),
            },
            Observer {
                data: vec![-0.1, 0.1],
                observations: 8.0,
                time: 1.0,
                age: 1.0,
                index: 2,
                local_threshold: 0.0,
                label_observations: HashMap::from([(0, 1.0)]),
            },
            Observer {
                data: vec![0.1, -0.1],
                observations: 7.0,
                time: 1.0,
                age: 1.0,
                index: 3,
                local_threshold: 0.0,
                label_observations: HashMap::from([(0, 1.0)]),
            },
        ];

        // Create non-uniform distribution (two separate clusters)
        let non_uniform_observers = vec![
            Observer {
                data: vec![-3.0, -3.0],
                observations: 6.0,
                time: 1.0,
                age: 1.0,
                index: 4,
                local_threshold: 0.0,
                label_observations: HashMap::from([(1, 1.0)]),
            },
            Observer {
                data: vec![3.0, 3.0],
                observations: 5.0,
                time: 1.0,
                age: 1.0,
                index: 5,
                local_threshold: 0.0,
                label_observations: HashMap::from([(1, 1.0)]),
            },
            Observer {
                data: vec![-3.1, -2.9],
                observations: 4.0,
                time: 1.0,
                age: 1.0,
                index: 6,
                local_threshold: 0.0,
                label_observations: HashMap::from([(1, 1.0)]),
            },
            Observer {
                data: vec![3.1, 2.9],
                observations: 3.0,
                time: 1.0,
                age: 1.0,
                index: 7,
                local_threshold: 0.0,
                label_observations: HashMap::from([(1, 1.0)]),
            },
        ];

        let all_observers = uniform_observers
            .into_iter()
            .chain(non_uniform_observers)
            .collect::<Vec<_>>();

        let indices: Vec<usize> = all_observers.iter().map(|obs| obs.index).collect();

        for obs in all_observers {
            obs_set.insert(obs);
        }
        obs_set.set_num_active(obs_set.len());

        (obs_set, indices)
    }

    #[test]
    fn test_mahalanobis_uniformity_score() {
        let (obs_set, _indices) = create_test_observers();

        // Test uniform cluster (indices 0-3)
        let uniform_indices = vec![0, 1, 2, 3];
        let uniform_score = obs_set.mahalanobis_uniformity_score(Some(&uniform_indices));

        // Test non-uniform cluster (indices 4-7)
        let non_uniform_indices = vec![4, 5, 6, 7];
        let non_uniform_score = obs_set.mahalanobis_uniformity_score(Some(&non_uniform_indices));

        println!("Uniform score: {}", uniform_score);
        println!("Non-uniform score: {}", non_uniform_score);

        // Uniform cluster should have lower score (more convex)
        assert!(uniform_score < non_uniform_score);

        // Test with all observers
        let all_score = obs_set.mahalanobis_uniformity_score(None);
        println!("All observers score: {}", all_score);

        // All observers should give a positive score (combined set)
        assert!(all_score > 0.0);
    }

    #[test]
    fn test_edge_cases() {
        let mut obs_set = ObserverSet::new(DistanceMetric::Euclidean, None);

        // Test with empty set
        let score = obs_set.mahalanobis_uniformity_score(None);
        assert_eq!(score, 0.0);

        // Test with single observer
        let single_obs = Observer {
            data: vec![1.0, 2.0],
            observations: 10.0,
            time: 1.0,
            age: 1.0,
            index: 0,
            local_threshold: 0.0,
            label_observations: HashMap::from([(0, 1.0)]),
        };
        obs_set.insert(single_obs);

        let score = obs_set.mahalanobis_uniformity_score(None);
        assert_eq!(score, 0.0);

        // Test with non-existent indices
        let score = obs_set.mahalanobis_uniformity_score(Some(&[999]));
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_mixed_indices() {
        let (obs_set, _) = create_test_observers();

        // Test with mixed indices from both clusters
        let mixed_indices = vec![0, 1, 4, 5];
        let mixed_score = obs_set.mahalanobis_uniformity_score(Some(&mixed_indices));

        println!("Mixed cluster score: {}", mixed_score);

        // Should be between uniform and non-uniform scores
        assert!(mixed_score > 0.0);
    }
}
