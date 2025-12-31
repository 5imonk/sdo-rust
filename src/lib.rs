use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::{HashMap, HashSet};
use std::f64;

/// Sparse Data Observers (SDO) Algorithm
#[pyclass]
pub struct SDO {
    // Entferne #[pyo3(get, set)] um Konflikt mit get_active_observers() zu vermeiden
    active_observers: Vec<Vec<f64>>,
    kdtree: KdTree<f64, usize, Vec<f64>>,
    #[pyo3(get, set)]
    x: usize,
}

#[pymethods]
impl SDO {
    #[new]
    pub fn new() -> Self {
        Self {
            active_observers: Vec::new(),
            kdtree: KdTree::new(2),
            x: 10,
        }
    }

    /// Lernt das Modell aus den Daten
    pub fn learn(
        &mut self,
        data: PyReadonlyArray2<f64>,
        k: usize,
        x: usize,
        rho: f64,
    ) -> PyResult<()> {
        let data_slice = data.as_array();
        let rows = data_slice.nrows();
        let cols = data_slice.ncols();

        if rows == 0 || k == 0 {
            return Ok(());
        }

        // Konvertiere NumPy-Array zu Vec<Vec<f64>>
        let data_vec: Vec<Vec<f64>> = (0..rows)
            .map(|i| (0..cols).map(|j| data_slice[[i, j]]).collect())
            .collect();

        self.x = x;
        let dimension = data_vec[0].len();

        // Schritt 1: Sample
        let mut rng = thread_rng();
        let observers: Vec<Vec<f64>> = data_vec
            .choose_multiple(&mut rng, k.min(data_vec.len()))
            .cloned()
            .collect();

        // Schritt 2: Observe
        let mut observations: Vec<(usize, usize)> = observers
            .iter()
            .enumerate()
            .map(|(idx, observer)| {
                let count = self.count_points_in_neighborhood(observer, &data_vec, x);
                (idx, count)
            })
            .collect();

        // Schritt 3: Clean model
        observations.sort_by_key(|&(_, count)| count);
        let num_active = ((observers.len() as f64) * (1.0 - rho)).ceil() as usize;
        let num_active = num_active.max(1).min(observers.len());

        self.active_observers = observations
            .iter()
            .rev()
            .take(num_active)
            .map(|&(idx, _)| observers[idx].clone())
            .collect();

        // Baue kd-tree
        self.kdtree = KdTree::new(dimension);
        for (idx, observer) in self.active_observers.iter().enumerate() {
            if let Err(e) = self.kdtree.add(observer.clone(), idx) {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Fehler beim Hinzufügen zum kd-tree: {:?}",
                    e
                )));
            }
        }

        Ok(())
    }

    /// Berechnet den Outlier-Score für einen Datenpunkt
    pub fn predict(&self, point: PyReadonlyArray2<f64>) -> PyResult<f64> {
        if self.active_observers.is_empty() {
            return Ok(f64::INFINITY);
        }

        let point_slice = point.as_array();
        if point_slice.nrows() != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Punkt muss ein 1D-Array oder 2D-Array mit einer Zeile sein",
            ));
        }

        let point_vec: Vec<f64> = (0..point_slice.ncols())
            .map(|j| point_slice[[0, j]])
            .collect();

        let k = self.x.min(self.active_observers.len());

        match self.kdtree.nearest(&point_vec, k, &squared_euclidean) {
            Ok(neighbors) => {
                if neighbors.is_empty() {
                    return Ok(f64::INFINITY);
                }

                let mut distances: Vec<f64> = neighbors
                    .iter()
                    .map(|(dist_squared, _)| dist_squared.sqrt())
                    .collect();

                distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let mid = distances.len() / 2;
                let median = if distances.len() % 2 == 0 && mid > 0 {
                    (distances[mid - 1] + distances[mid]) / 2.0
                } else {
                    distances[mid]
                };

                Ok(median)
            }
            Err(_) => {
                // Fallback auf Brute-Force
                Ok(self.predict_brute_force(&point_vec))
            }
        }
    }

    /// Konvertiert active_observers zu NumPy-Array für Python
    pub fn get_active_observers(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        if self.active_observers.is_empty() {
            let array = PyArray2::zeros_bound(py, (0, 0), false);
            return Ok(array.unbind());
        }

        let rows = self.active_observers.len();
        let cols = self.active_observers[0].len();
        let array = PyArray2::zeros_bound(py, (rows, cols), false);

        unsafe {
            let mut array_mut = array.as_array_mut();
            for (i, observer) in self.active_observers.iter().enumerate() {
                for (j, &value) in observer.iter().enumerate() {
                    array_mut[[i, j]] = value;
                }
            }
        }

        Ok(array.unbind())
    }
}

impl SDO {
    fn count_points_in_neighborhood(&self, observer: &[f64], data: &[Vec<f64>], x: usize) -> usize {
        let mut distances: Vec<(usize, f64)> = data
            .iter()
            .enumerate()
            .map(|(idx, point)| {
                let dist = squared_euclidean(observer, point);
                (idx, dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut count = 0;
        for (idx, _) in distances.iter().skip(1).take(x) {
            if data[*idx] != observer {
                count += 1;
            }
        }
        count.min(x)
    }

    fn predict_brute_force(&self, point: &[f64]) -> f64 {
        let mut distances: Vec<f64> = self
            .active_observers
            .iter()
            .map(|observer| euclidean_distance(point, observer))
            .collect();

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let k = self.x.min(distances.len());
        let relevant_distances: Vec<f64> = distances.into_iter().take(k).collect();

        if relevant_distances.is_empty() {
            return f64::INFINITY;
        }

        let mid = relevant_distances.len() / 2;
        if relevant_distances.len() % 2 == 0 && mid > 0 {
            (relevant_distances[mid - 1] + relevant_distances[mid]) / 2.0
        } else {
            relevant_distances[mid]
        }
    }
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// ============================================================================
// SDOclust - Clustering Extension
// ============================================================================

/// Union-Find Datenstruktur für Connected Components
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x != root_y {
            if self.rank[root_x] < self.rank[root_y] {
                self.parent[root_x] = root_y;
            } else if self.rank[root_x] > self.rank[root_y] {
                self.parent[root_y] = root_x;
            } else {
                self.parent[root_y] = root_x;
                self.rank[root_x] += 1;
            }
        }
    }

    fn get_components(&mut self, n: usize) -> Vec<Vec<usize>> {
        // Normalisiere alle Pfade
        for i in 0..n {
            self.find(i);
        }

        // Gruppiere nach Root
        let mut components: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            components
                .entry(self.parent[i])
                .or_insert_with(Vec::new)
                .push(i);
        }

        components.into_values().collect()
    }
}

/// Sparse Data Observers Clustering (SDOclust) Algorithm
#[pyclass]
pub struct SDOclust {
    active_observers: Vec<Vec<f64>>,
    kdtree: KdTree<f64, usize, Vec<f64>>,
    observer_labels: Vec<i32>, // Label für jeden Observer (-1 = entfernt)
    #[pyo3(get, set)]
    x: usize,
    #[pyo3(get, set)]
    chi: usize, // χ - Anzahl der nächsten Observer für lokale Thresholds
    #[pyo3(get, set)]
    zeta: f64, // ζ - Mixing-Parameter für globale/lokale Thresholds
    #[pyo3(get, set)]
    min_cluster_size: usize, // e - minimale Clustergröße
}

#[pymethods]
impl SDOclust {
    #[new]
    pub fn new() -> Self {
        Self {
            active_observers: Vec::new(),
            kdtree: KdTree::new(2),
            observer_labels: Vec::new(),
            x: 10,
            chi: 4,
            zeta: 0.5,
            min_cluster_size: 2,
        }
    }

    /// Lernt das Modell aus den Daten und führt Clustering durch
    pub fn learn(
        &mut self,
        data: PyReadonlyArray2<f64>,
        k: usize,
        x: usize,
        rho: f64,
        chi: usize,
        zeta: f64,
        min_cluster_size: usize,
    ) -> PyResult<()> {
        let data_slice = data.as_array();
        let rows = data_slice.nrows();
        let cols = data_slice.ncols();

        if rows == 0 || k == 0 {
            return Ok(());
        }

        self.x = x;
        self.chi = chi;
        self.zeta = zeta;
        self.min_cluster_size = min_cluster_size;

        // Konvertiere NumPy-Array zu Vec<Vec<f64>>
        let data_vec: Vec<Vec<f64>> = (0..rows)
            .map(|i| (0..cols).map(|j| data_slice[[i, j]]).collect())
            .collect();

        let dimension = data_vec[0].len();

        // Schritt 1: Sample (wie SDO)
        let mut rng = thread_rng();
        let observers: Vec<Vec<f64>> = data_vec
            .choose_multiple(&mut rng, k.min(data_vec.len()))
            .cloned()
            .collect();

        // Schritt 2: Observe (wie SDO)
        let mut observations: Vec<(usize, usize)> = observers
            .iter()
            .enumerate()
            .map(|(idx, observer)| {
                let count = Self::count_points_in_neighborhood(observer, &data_vec, x);
                (idx, count)
            })
            .collect();

        // Schritt 3: Clean model (wie SDO)
        observations.sort_by_key(|&(_, count)| count);
        let num_active = ((observers.len() as f64) * (1.0 - rho)).ceil() as usize;
        let num_active = num_active.max(1).min(observers.len());

        self.active_observers = observations
            .iter()
            .rev()
            .take(num_active)
            .map(|&(idx, _)| observers[idx].clone())
            .collect();

        // Schritt 4: Clustering
        self.perform_clustering()?;

        // Schritt 5: Entferne kleine Cluster
        self.remove_small_clusters();

        // Baue kd-tree nur für aktive Observer mit Labels
        self.kdtree = KdTree::new(dimension);
        for (idx, observer) in self.active_observers.iter().enumerate() {
            if self.observer_labels[idx] >= 0 {
                if let Err(e) = self.kdtree.add(observer.clone(), idx) {
                    return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Fehler beim Hinzufügen zum kd-tree: {:?}",
                        e
                    )));
                }
            }
        }

        Ok(())
    }

    /// Berechnet das Cluster-Label für einen Datenpunkt
    pub fn predict(&self, point: PyReadonlyArray2<f64>) -> PyResult<i32> {
        if self.active_observers.is_empty() {
            return Ok(-1); // Kein Label (Outlier)
        }

        let point_slice = point.as_array();
        if point_slice.nrows() != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Punkt muss ein 1D-Array oder 2D-Array mit einer Zeile sein",
            ));
        }

        let point_vec: Vec<f64> = (0..point_slice.ncols())
            .map(|j| point_slice[[0, j]])
            .collect();

        let k = self.x.min(self.active_observers.len());

        match self.kdtree.nearest(&point_vec, k, &squared_euclidean) {
            Ok(neighbors) => {
                if neighbors.is_empty() {
                    return Ok(-1);
                }

                // Zähle Labels der x nächsten Observer
                let mut label_counts: HashMap<i32, usize> = HashMap::new();

                for (_, &observer_idx) in neighbors.iter() {
                    if observer_idx < self.observer_labels.len() {
                        let label = self.observer_labels[observer_idx];
                        if label >= 0 {
                            *label_counts.entry(label).or_insert(0) += 1;
                        }
                    }
                }

                // Finde das häufigste Label
                if let Some((&most_common_label, _)) =
                    label_counts.iter().max_by_key(|(_, &count)| count)
                {
                    Ok(most_common_label)
                } else {
                    Ok(-1) // Kein Label gefunden
                }
            }
            Err(_) => {
                // Fallback auf Brute-Force
                Ok(self.predict_brute_force(&point_vec))
            }
        }
    }

    /// Gibt die Anzahl der Cluster zurück
    pub fn n_clusters(&self) -> usize {
        let unique_labels: HashSet<i32> = self
            .observer_labels
            .iter()
            .filter(|&&l| l >= 0)
            .copied()
            .collect();
        unique_labels.len()
    }

    /// Gibt die Labels der Observer zurück
    pub fn get_observer_labels(&self) -> Vec<i32> {
        self.observer_labels.clone()
    }

    /// Konvertiert active_observers zu NumPy-Array für Python
    pub fn get_active_observers(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        if self.active_observers.is_empty() {
            let array = PyArray2::zeros_bound(py, (0, 0), false);
            return Ok(array.unbind());
        }

        let rows = self.active_observers.len();
        let cols = self.active_observers[0].len();
        let array = PyArray2::zeros_bound(py, (rows, cols), false);

        unsafe {
            let mut array_mut = array.as_array_mut();
            for (i, observer) in self.active_observers.iter().enumerate() {
                for (j, &value) in observer.iter().enumerate() {
                    array_mut[[i, j]] = value;
                }
            }
        }

        Ok(array.unbind())
    }
}

impl SDOclust {
    fn perform_clustering(&mut self) -> PyResult<()> {
        let n = self.active_observers.len();
        if n == 0 {
            return Ok(());
        }

        // Schritt 1: Berechne lokale Cutoff-Thresholds h_ω
        let mut local_thresholds = vec![0.0; n];
        let mut observer_kdtree = KdTree::new(self.active_observers[0].len());

        // Baue kd-tree für Observer
        for (idx, observer) in self.active_observers.iter().enumerate() {
            if let Err(e) = observer_kdtree.add(observer.clone(), idx) {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Fehler beim Hinzufügen zum Observer kd-tree: {:?}",
                    e
                )));
            }
        }

        // Berechne lokale Thresholds
        for (idx, observer) in self.active_observers.iter().enumerate() {
            let chi_actual = (self.chi + 1).min(n); // +1 weil der Observer selbst nicht zählt
            match observer_kdtree.nearest(observer, chi_actual, &squared_euclidean) {
                Ok(neighbors) => {
                    if neighbors.len() > self.chi {
                        // Der χ-te nächste Observer (skip self)
                        let (dist_squared, _) = neighbors[self.chi];
                        local_thresholds[idx] = dist_squared.sqrt();
                    } else if !neighbors.is_empty() {
                        // Fallback: verwende den letzten verfügbaren
                        let (dist_squared, _) = neighbors[neighbors.len() - 1];
                        local_thresholds[idx] = dist_squared.sqrt();
                    } else {
                        local_thresholds[idx] = f64::INFINITY;
                    }
                }
                Err(_) => {
                    local_thresholds[idx] = f64::INFINITY;
                }
            }
        }

        // Schritt 2: Berechne globalen Density-Threshold h
        let global_threshold: f64 =
            local_thresholds.iter().sum::<f64>() / local_thresholds.len() as f64;

        // Schritt 3: Berechne finale Thresholds mit Mixture-Modell: h'_ω = ζ·h_ω + (1-ζ)·h
        let final_thresholds: Vec<f64> = local_thresholds
            .iter()
            .map(|&h_omega| self.zeta * h_omega + (1.0 - self.zeta) * global_threshold)
            .collect();

        // Schritt 4: Baue Graph - zwei Observer sind verbunden wenn d(ν,ω) < h'_ω UND d(ν,ω) < h'_ν
        let mut uf = UnionFind::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                let dist = euclidean_distance(&self.active_observers[i], &self.active_observers[j]);
                if dist < final_thresholds[i] && dist < final_thresholds[j] {
                    uf.union(i, j);
                }
            }
        }

        // Schritt 5: Weise Labels zu
        let components = uf.get_components(n);
        self.observer_labels = vec![-1; n]; // -1 = kein Label/entfernt

        for (label, component) in components.iter().enumerate() {
            for &observer_idx in component {
                self.observer_labels[observer_idx] = label as i32;
            }
        }

        Ok(())
    }

    fn remove_small_clusters(&mut self) {
        // Zähle Größe jedes Clusters
        let mut cluster_sizes: HashMap<i32, usize> = HashMap::new();
        for &label in &self.observer_labels {
            if label >= 0 {
                *cluster_sizes.entry(label).or_insert(0) += 1;
            }
        }

        // Entferne Observer aus zu kleinen Clustern
        for label in self.observer_labels.iter_mut() {
            if *label >= 0 {
                if let Some(&size) = cluster_sizes.get(label) {
                    if size < self.min_cluster_size {
                        *label = -1; // Markiere als entfernt
                    }
                }
            }
        }
    }

    fn count_points_in_neighborhood(observer: &[f64], data: &[Vec<f64>], x: usize) -> usize {
        let mut distances: Vec<(usize, f64)> = data
            .iter()
            .enumerate()
            .map(|(idx, point)| {
                let dist = squared_euclidean(observer, point);
                (idx, dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut count = 0;
        for (idx, _) in distances.iter().skip(1).take(x) {
            if data[*idx] != observer {
                count += 1;
            }
        }
        count.min(x)
    }

    fn predict_brute_force(&self, point: &[f64]) -> i32 {
        let mut distances: Vec<(usize, f64)> = self
            .active_observers
            .iter()
            .enumerate()
            .map(|(idx, observer)| (idx, euclidean_distance(point, observer)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let k = self.x.min(distances.len());
        let mut label_counts: HashMap<i32, usize> = HashMap::new();

        for (idx, _) in distances.iter().take(k) {
            if *idx < self.observer_labels.len() {
                let label = self.observer_labels[*idx];
                if label >= 0 {
                    *label_counts.entry(label).or_insert(0) += 1;
                }
            }
        }

        if let Some((&most_common_label, _)) = label_counts.iter().max_by_key(|(_, &count)| count) {
            most_common_label
        } else {
            -1
        }
    }
}

/// Python-Modul
#[pymodule]
fn sdo(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<SDO>()?;
    m.add_class::<SDOclust>()?;
    Ok(())
}
