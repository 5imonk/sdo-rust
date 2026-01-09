use std::collections::HashSet;
use std::sync::Arc;

use crate::observer::{NormalizedScoreKey, ObservationKey, Observer, ObserverSet, OrderedFloat};

/// Streaming-Erweiterungen für ObserverSet
/// Enthält Funktionen für zeitbasiertes Fading und Observation-Updates
impl ObserverSet {
    /// Aktualisiert Pω und Hω für alle Observer mit zeitbasiertem Exponential Moving Average
    /// Hω ← f^(ti - ti-1) · Hω + 1
    /// Pω ← f^(ti - ti-1) · Pω + 1 wenn ω unter den x-nächsten, sonst Pω ← f^(ti - ti-1) · Pω
    ///
    /// # Arguments
    /// * `nearest_indices` - Indizes der Observer, die zu den x-nächsten gehören (bekommen +1)
    /// * `fading` - Fading-Parameter f = exp(-T^-1)
    /// * `current_time` - Aktuelle Zeit ti
    pub fn update_observations_with_fading(
        &mut self,
        nearest_indices: &[usize],
        fading: f64,
        current_time: f64,
    ) {
        // Sammle Observer-Daten in separatem Scope, um Borrow-Konflikte zu vermeiden
        let updates: Vec<(usize, f64, f64)> = {
            let nearest_set: HashSet<usize> = nearest_indices.iter().cloned().collect();

            // Verwende iter_observers für effizienten Zugriff ohne Kopie
            self.iter_observers(true)
                .map(|observer| {
                    // Berechne Zeitdifferenz: ti - ti-1
                    let time_diff = current_time - observer.time;
                    // Berechne fading-Faktor für diese Zeitdifferenz: f^(ti - ti-1)
                    let fading_factor = fading.powf(time_diff);

                    // Update observations: Pω ← f^(ti - ti-1) · Pω + 1 (wenn nearest) bzw. f^(ti - ti-1) · Pω
                    let new_observations = if nearest_set.contains(&observer.index) {
                        fading_factor * observer.observations + 1.0
                    } else {
                        fading_factor * observer.observations
                    };

                    // Update age: Hω ← f^(ti - ti-1) · Hω + 1
                    let new_age = fading_factor * observer.age + 1.0;

                    (observer.index, new_observations, new_age)
                })
                .collect()
        };

        // Aktualisiere jeden Observer mit observations, age und time
        for (index, new_observations, new_age) in updates {
            self.update_observer_with_time(index, new_observations, new_age, current_time);
        }
    }

    /// Update observations, age, and time - O(log n)
    /// Wichtig für zeitbasierte Updates in SDOstream
    pub fn update_observer_with_time(
        &mut self,
        index: usize,
        new_observations: f64,
        new_age: f64,
        new_time: f64,
    ) -> bool {
        // Get the current observer Arc
        let observer_arc = match self.observers_by_index.get(&index) {
            Some(arc) => arc.clone(),
            None => return false,
        };

        // Remove old entries from secondary indices using old values
        let old_obs_key = ObservationKey {
            observations: OrderedFloat(observer_arc.observations),
            index,
        };
        let old_normalized_score = if observer_arc.age > 0.0 {
            observer_arc.observations / observer_arc.age
        } else {
            f64::INFINITY
        };
        let old_score_key = NormalizedScoreKey {
            score: OrderedFloat(old_normalized_score),
            index,
        };

        self.indices_by_obs.remove(&old_obs_key);
        self.indices_by_score.remove(&old_score_key);

        // Update the observer - try to update in place if we have exclusive access
        let updated_observer = {
            // Get mutable reference to the Arc in the HashMap
            let arc_mut = self.observers_by_index.get_mut(&index).unwrap();
            if let Some(mut_observer) = Arc::get_mut(arc_mut) {
                // Exclusive access - update in place (no clone!)
                mut_observer.observations = new_observations;
                mut_observer.age = new_age;
                mut_observer.time = new_time;
                Arc::clone(arc_mut) // Clone the Arc reference, not the Observer
            } else {
                // Shared - create new Arc with updated values
                Arc::new(Observer {
                    data: observer_arc.data.clone(),
                    observations: new_observations,
                    time: new_time,
                    age: new_age,
                    index: observer_arc.index,
                    label: observer_arc.label,
                })
            }
        };

        // Update HashMap with new Arc
        self.observers_by_index.insert(index, updated_observer);

        // Re-insert with updated values
        let new_obs_key = ObservationKey {
            observations: OrderedFloat(new_observations),
            index,
        };
        let new_normalized_score = if new_age > 0.0 {
            new_observations / new_age
        } else {
            f64::INFINITY
        };
        let new_score_key = NormalizedScoreKey {
            score: OrderedFloat(new_normalized_score),
            index,
        };

        self.indices_by_obs.insert(new_obs_key, index);
        self.indices_by_score.insert(new_score_key, index);

        true
    }
}
