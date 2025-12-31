use acap::kd::KdTree;
use acap::vp::VpTree;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::f64;

use crate::sdo_impl::{SDOParams, SpatialTree, TreeType, SDO};
use crate::utils::{compute_distance, Point};

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
            components.entry(self.parent[i]).or_default().push(i);
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
    #[pyo3(get, set)]
    pub tree_type: String, // "vptree" (default) oder "kdtree"
}

#[pymethods]
#[allow(clippy::too_many_arguments)]
impl SDOclustParams {
    #[new]
    #[pyo3(signature = (k, x, rho, chi = 4, zeta = 0.5, min_cluster_size = 2, distance = "euclidean".to_string(), minkowski_p = None, tree_type = "vptree".to_string()))]
    pub fn new(
        k: usize,
        x: usize,
        rho: f64,
        chi: usize,
        zeta: f64,
        min_cluster_size: usize,
        distance: String,
        minkowski_p: Option<f64>,
        tree_type: String,
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
            tree_type,
        }
    }
}

impl SDOclustParams {
    fn get_tree_type(&self) -> TreeType {
        match self.tree_type.to_lowercase().as_str() {
            "kdtree" => TreeType::KdTree,
            _ => TreeType::VpTree, // Default: VpTree
        }
    }
}

/// Sparse Data Observers Clustering (SDOclust) Algorithm
#[pyclass]
pub struct SDOclust {
    sdo: SDO, // Internes SDO-Objekt für Modell-Erstellung
    tree_active: Option<SpatialTree>,
    observer_labels: Vec<i32>, // Label für jeden Observer (-1 = entfernt)
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
            sdo: SDO::new(),
            tree_active: None,
            observer_labels: Vec::new(),
            chi: 4,
            zeta: 0.5,
            min_cluster_size: 2,
        }
    }

    /// Lernt das Modell aus den Daten und führt Clustering durch
    pub fn learn(&mut self, data: PyReadonlyArray2<f64>, params: &SDOclustParams) -> PyResult<()> {
        let data_slice = data.as_array();
        let rows = data_slice.nrows();

        if rows == 0 || params.k == 0 {
            return Ok(());
        }

        self.chi = params.chi;
        self.zeta = params.zeta;
        self.min_cluster_size = params.min_cluster_size;

        // Verwende SDO für Modell-Erstellung (Sample, Observe, Clean)
        let sdo_params = SDOParams::new(
            params.k,
            params.x,
            params.rho,
            params.distance.clone(),
            params.minkowski_p,
            params.tree_type.clone(),
        );
        self.sdo.learn(data, &sdo_params)?;

        // Hole active_observers aus SDO
        let active_observers = self.sdo.get_active_observers_internal().clone();

        // Schritt 4: Clustering
        self.perform_clustering(&active_observers)?;

        // Schritt 5: Entferne kleine Cluster
        self.remove_small_clusters();

        // Baue Tree nur für aktive Observer mit Labels
        let points: Vec<Point> = active_observers
            .iter()
            .enumerate()
            .filter(|(idx, _)| *idx < self.observer_labels.len() && self.observer_labels[*idx] >= 0)
            .map(|(_, observer)| Point(observer.clone()))
            .collect();

        self.tree_active = match params.get_tree_type() {
            TreeType::VpTree => Some(SpatialTree::Vp(VpTree::balanced(points))),
            TreeType::KdTree => Some(SpatialTree::Kd(KdTree::from_iter(points))),
        };

        Ok(())
    }

    /// Berechnet den Outlier-Score für einen Datenpunkt (wie SDO)
    pub fn predict_outlier_score(&self, point: PyReadonlyArray2<f64>) -> PyResult<f64> {
        self.sdo.predict(point)
    }

    /// Berechnet das Cluster-Label für einen Datenpunkt
    pub fn predict(&self, point: PyReadonlyArray2<f64>) -> PyResult<i32> {
        let active_observers = self.sdo.get_active_observers_internal();
        if active_observers.is_empty() {
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
        Ok(self.predict_brute_force(&point_vec, active_observers))
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

    /// Gibt x (Anzahl der Nachbarn) zurück
    #[getter]
    pub fn x(&self) -> usize {
        self.sdo.get_x_internal()
    }

    /// Gibt die Labels der Observer zurück
    pub fn get_observer_labels(&self) -> Vec<i32> {
        self.observer_labels.clone()
    }

    /// Konvertiert active_observers zu NumPy-Array für Python
    pub fn get_active_observers(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        self.sdo.get_active_observers(py)
    }
}

impl SDOclust {
    fn perform_clustering(&mut self, active_observers: &[Vec<f64>]) -> PyResult<()> {
        let n = active_observers.len();
        if n == 0 {
            return Ok(());
        }

        // Schritt 1: Berechne lokale Cutoff-Thresholds h_ω
        let mut local_thresholds = vec![0.0; n];
        let distance_metric = self.sdo.get_distance_metric_internal();
        let minkowski_p = self.sdo.get_minkowski_p_internal();

        // Berechne lokale Thresholds mit Brute-Force (mit gewählter Distanzfunktion)
        for (idx, observer) in active_observers.iter().enumerate() {
            let mut distances: Vec<f64> = active_observers
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != idx)
                .map(|(_, other)| compute_distance(observer, other, distance_metric, minkowski_p))
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
                let dist = compute_distance(
                    &active_observers[i],
                    &active_observers[j],
                    distance_metric,
                    minkowski_p,
                );
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

    fn predict_brute_force(&self, point: &[f64], active_observers: &[Vec<f64>]) -> i32 {
        let distance_metric = self.sdo.get_distance_metric_internal();
        let minkowski_p = self.sdo.get_minkowski_p_internal();
        let x = self.sdo.get_x_internal();

        let mut distances: Vec<(usize, f64)> = active_observers
            .iter()
            .enumerate()
            .map(|(idx, observer)| {
                (
                    idx,
                    compute_distance(point, observer, distance_metric, minkowski_p),
                )
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let k = x.min(distances.len());
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
