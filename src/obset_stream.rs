use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::obs::{NeighborInfo, NormalizedScoreKey, ObservationKey, Observer, OrderedFloat};
use crate::obset::ObserverSet;

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
            self.iter_observers(false)
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
                    local_threshold: observer_arc.local_threshold,
                    label_observations: observer_arc.label_observations.clone(),
                    label_time: observer_arc.label_time,
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

        self.indices_by_obs.insert(new_obs_key);
        self.indices_by_score.insert(new_score_key);

        true
    }

    /// Batch-Update von observations mit zeitbasiertem Fading
    /// Zusammenfasst mehrere Beobachtungen (NeighborInfo) und aktualisiert alle Observer
    ///
    /// # Arguments
    /// * `neighbor_info_batch` - Vektor von NeighborInfo-Vektoren (eine pro Beobachtung)
    /// * `observation_times` - Zeitpunkte für jede Beobachtung (gleiche Länge wie neighbor_info_batch)
    /// * `fading` - Fading-Parameter f = exp(-T^-1)
    ///
    /// # Returns
    /// Anzahl der verarbeiteten Beobachtungen
    pub fn update(
        &mut self,
        neighbor_info_batch: Vec<Vec<NeighborInfo>>,
        observation_times: Vec<f64>,
        fading: f64,
    ) -> usize {
        // Validierung
        assert_eq!(
            neighbor_info_batch.len(),
            observation_times.len(),
            "neighbor_info_batch und observation_times müssen gleiche Länge haben"
        );

        let num_observations = neighbor_info_batch.len();
        if num_observations == 0 {
            return 0;
        }

        // Berechne reference_start_time (min) und reference_time (max)
        let reference_start_time = observation_times
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let reference_time = observation_times
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        // Berechne batch_age: Maximum observation die ein Observer während dieses Batches erreichen könnte
        // = Summe über alle observation_times von fading^(reference_time - t)
        let batch_age: f64 = observation_times
            .iter()
            .map(|&t| fading.powf(reference_time - t))
            .sum();

        // Schritt 1: Zusammenfassen der Beobachtungen
        // HashMap: Observer-Index -> gewichteter Beitrag (summe von fading_factor)
        let mut observer_contributions: HashMap<usize, f64> = HashMap::new();

        for (neighbors, &observation_time) in
            neighbor_info_batch.iter().zip(observation_times.iter())
        {
            // Berechne Fading-Faktor für diese Beobachtung: fading^(reference_time - observation_time)
            let time_elapsed = reference_time - observation_time;
            let fading_factor = fading.powf(time_elapsed);

            // Für jede NeighborInfo in dieser Beobachtung: addiere fading_factor zum Beitrag
            for neighbor_info in neighbors {
                *observer_contributions
                    .entry(neighbor_info.index)
                    .or_insert(0.0) += fading_factor;
            }
        }

        // Schritt 2: Fade alle Observer und addiere Beiträge
        let updates: Vec<(usize, f64, f64)> = {
            self.iter_observers(false)
                .map(|observer| {
                    // Für observations: Fade zur reference_time (max)
                    let fading_factor_obs = fading.powf(reference_time - observer.time);

                    // Hole Beitrag für diesen Observer (0.0 wenn nicht vorhanden)
                    let contribution = observer_contributions
                        .get(&observer.index)
                        .copied()
                        .unwrap_or(0.0);

                    // Neue observations: gefadete observations + Beiträge (zur reference_time)
                    let new_observations = fading_factor_obs * observer.observations + contribution;

                    // Für age: Fade zur reference_start_time (min)
                    let fading_factor_age = fading.powf(reference_start_time - observer.time);

                    // Neue age: gefadete age + batch_age (beide zur reference_start_time)
                    let new_age = fading_factor_age * observer.age + batch_age * fading_factor_age;

                    (observer.index, new_observations, new_age)
                })
                .collect()
        };

        // Schritt 3: Aktualisiere jeden Observer mit observations, age und reference_time
        for (index, new_observations, new_age) in updates {
            self.update_observer_with_time(index, new_observations, new_age, reference_time);
        }

        num_observations
    }
}
