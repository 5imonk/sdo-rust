use acap::kd::KdTree;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::{HashMap, HashSet};
use std::f64;

use crate::sdo_impl::{DistanceMetric, Point};
use crate::utils::euclidean_distance;

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

/// Parameter-Struktur für SDOclust
#[pyclass]
#[derive(Clone)]
pub struct SDOclustParams {
    #[pyo3(get, set)]
    pub k: usize,
    #[pyo3(get, set)]
    pub x: usize,
    #[pyo3(get, set)]
    pub rho: f64,
    #[pyo3(get, set)]
    pub chi: usize,
    #[pyo3(get, set)]
    pub zeta: f64,
    #[pyo3(get, set)]
    pub min_cluster_size: usize,
    #[pyo3(get, set)]
    pub distance: String, // "euclidean", "manhattan", "chebyshev", "minkowski"
    #[pyo3(get, set)]
    pub minkowski_p: Option<f64>, // Für Minkowski-Distanz
}

#[pymethods]
impl SDOclustParams {
    #[new]
    #[pyo3(signature = (k, x, rho, chi, zeta, min_cluster_size, distance = "euclidean".to_string(), minkowski_p = None))]
    pub fn new(
        k: usize,
        x: usize,
        rho: f64,
        chi: usize,
        zeta: f64,
        min_cluster_size: usize,
        distance: String,
        minkowski_p: Option<f64>,
    ) -> Self {
        Self {
            k,
            x,
            rho,
            chi,
            zeta,
            min_cluster_size,
            distance,
            minkowski_p,
        }
    }
}

impl SDOclustParams {
    fn get_metric(&self) -> DistanceMetric {
        match self.distance.to_lowercase().as_str() {
            "manhattan" => DistanceMetric::Manhattan,
            "chebyshev" => DistanceMetric::Chebyshev,
            "minkowski" => DistanceMetric::Minkowski,
            _ => DistanceMetric::Euclidean,
        }
    }
}

/// Sparse Data Observers Clustering (SDOclust) Algorithm
#[pyclass]
pub struct SDOclust {
    active_observers: Vec<Vec<f64>>,
    kdtree: Option<KdTree<Point>>,
    observer_labels: Vec<i32>, // Label für jeden Observer (-1 = entfernt)
    distance_metric: DistanceMetric,
    minkowski_p: Option<f64>,
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
            kdtree: None,
            observer_labels: Vec::new(),
            distance_metric: DistanceMetric::Euclidean,
            minkowski_p: None,
            x: 10,
            chi: 4,
            zeta: 0.5,
            min_cluster_size: 2,
        }
    }

    /// Lernt das Modell aus den Daten und führt Clustering durch
    pub fn learn(&mut self, data: PyReadonlyArray2<f64>, params: &SDOclustParams) -> PyResult<()> {
        let data_slice = data.as_array();
        let rows = data_slice.nrows();
        let cols = data_slice.ncols();

        if rows == 0 || params.k == 0 {
            return Ok(());
        }

        self.x = params.x;
        self.chi = params.chi;
        self.zeta = params.zeta;
        self.min_cluster_size = params.min_cluster_size;
        self.distance_metric = params.get_metric();
        self.minkowski_p = params.minkowski_p;

        // Konvertiere NumPy-Array zu Vec<Vec<f64>>
        let data_vec: Vec<Vec<f64>> = (0..rows)
            .map(|i| (0..cols).map(|j| data_slice[[i, j]]).collect())
            .collect();

        // Schritt 1: Sample (wie SDO)
        let mut rng = thread_rng();
        let observers: Vec<Vec<f64>> = data_vec
            .choose_multiple(&mut rng, params.k.min(data_vec.len()))
            .cloned()
            .collect();

        // Schritt 2: Observe (wie SDO)
        let mut observations: Vec<(usize, usize)> = observers
            .iter()
            .enumerate()
            .map(|(idx, observer)| {
                let count = Self::count_points_in_neighborhood(observer, &data_vec, params.x);
                (idx, count)
            })
            .collect();

        // Schritt 3: Clean model (wie SDO)
        observations.sort_by_key(|&(_, count)| count);
        let num_active = ((observers.len() as f64) * (1.0 - params.rho)).ceil() as usize;
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
        let points: Vec<Point> = self
            .active_observers
            .iter()
            .enumerate()
            .filter(|(idx, _)| self.observer_labels[*idx] >= 0)
            .map(|(_, observer)| Point(observer.clone()))
            .collect();
        self.kdtree = Some(KdTree::from_iter(points));

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

        // Verwende Brute-Force mit gewählter Distanzfunktion
        Ok(self.predict_brute_force(&point_vec))
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

        // Berechne lokale Thresholds mit Brute-Force (mit gewählter Distanzfunktion)
        for (idx, observer) in self.active_observers.iter().enumerate() {
            let mut distances: Vec<f64> = self
                .active_observers
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != idx)
                .map(|(_, other)| self.compute_distance(observer, other))
                .collect();

            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let chi_actual = self.chi.min(distances.len());
            if chi_actual > 0 {
                local_thresholds[idx] = distances[chi_actual - 1];
            } else {
                local_thresholds[idx] = f64::INFINITY;
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
                let dist =
                    self.compute_distance(&self.active_observers[i], &self.active_observers[j]);
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
        // Verwende euklidische Distanz für count (kann später angepasst werden)
        let mut distances: Vec<(usize, f64)> = data
            .iter()
            .enumerate()
            .map(|(idx, point)| {
                let dist = euclidean_distance(observer, point);
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

    fn compute_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        match self.distance_metric {
            DistanceMetric::Euclidean => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt(),
            DistanceMetric::Manhattan => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .sum::<f64>(),
            DistanceMetric::Chebyshev => a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f64::max),
            DistanceMetric::Minkowski => {
                let p = self.minkowski_p.unwrap_or(3.0);
                let sum: f64 = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).abs().powf(p))
                    .sum();
                sum.powf(1.0 / p)
            }
        }
    }

    fn predict_brute_force(&self, point: &[f64]) -> i32 {
        let mut distances: Vec<(usize, f64)> = self
            .active_observers
            .iter()
            .enumerate()
            .map(|(idx, observer)| (idx, self.compute_distance(point, observer)))
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

impl Default for SDOclust {
    fn default() -> Self {
        Self::new()
    }
}
