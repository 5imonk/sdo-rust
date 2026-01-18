#[cfg(feature = "testing")]
pub mod sdostream_testing {
    use crate::sdostrcl_impl::SDOstreamclust;
    use crate::sdostream_impl::SDOstream;

    impl SDOstream {
        /// Get all normalized scores for observers
        pub fn get_normalized_scores(&self) -> Vec<f64> {
            self.sdo
                .observers
                .find_k_worst_normalized_scores(None)
                .into_iter()
                .map(|(_, score)| score)
                .collect()
        }

        /// Get last replacement time for testing
        pub fn get_last_replacement_time(&self) -> f64 {
            self.last_replacement_time
        }

        /// Get pending replacements for testing  
        pub fn get_pending_replacements(&self) -> usize {
            self.pending_replacements
        }

        /// Force a specific seed for reproducible testing
        pub fn set_testing_seed(&mut self, seed: u64) {
            // Note: This would require modifying the random number generation
            // For now, this is a placeholder for the interface
        }
    }

    impl SDOstreamclust {
        /// Get all normalized cluster scores for observers
        pub fn get_normalized_cluster_scores_all(
            &self,
        ) -> Vec<std::collections::HashMap<i32, f64>> {
            let all_indices: Vec<usize> = self
                .sdostream
                .get_sdo()
                .observers
                .iter_observers(false)
                .map(|obs| obs.index)
                .collect();

            self.sdostream
                .get_sdo()
                .observers
                .get_normalized_cluster_scores(&all_indices)
        }
    }
}
