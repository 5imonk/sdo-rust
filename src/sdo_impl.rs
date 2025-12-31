use acap::chebyshev::Chebyshev;
use acap::euclid::Euclidean;
use acap::kd::KdTree;
use acap::taxi::Taxicab;
use acap::vp::VpTree;

use crate::utils::Minkowski;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::f64;

use crate::utils::{compute_distance, Observer};

/// Distanzfunktion für SDO
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum DistanceMetric {
    Euclidean = 0,
    Manhattan = 1,
    Chebyshev = 2,
    Minkowski = 3,
}

/// Tree-Typ für die räumliche Indexierung
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TreeType {
    VpTree = 0,
    KdTree = 1,
}

/// Enum für beide Tree-Typen mit verschiedenen Metriken
#[allow(dead_code)]
pub(crate) enum SpatialTree {
    VpEuclidean(VpTree<Euclidean<Vec<f64>>>),
    VpManhattan(VpTree<Taxicab<Vec<f64>>>),
    VpChebyshev(VpTree<Chebyshev<Vec<f64>>>),
    VpMinkowski(VpTree<Minkowski>),
    KdEuclidean(KdTree<Euclidean<Vec<f64>>>),
    KdManhattan(KdTree<Taxicab<Vec<f64>>>),
    KdChebyshev(KdTree<Chebyshev<Vec<f64>>>),
    KdMinkowski(KdTree<Minkowski>),
}

/// Parameter-Struktur für SDO
#[pyclass]
#[derive(Clone)]
pub struct SDOParams {
    #[pyo3(get, set)]
    pub k: usize,
    #[pyo3(get, set)]
    pub x: usize,
    #[pyo3(get, set)]
    pub rho: f64,
    #[pyo3(get, set)]
    pub distance: String, // "euclidean", "manhattan", "chebyshev", "minkowski"
    #[pyo3(get, set)]
    pub minkowski_p: Option<f64>, // Für Minkowski-Distanz
    #[pyo3(get, set)]
    pub tree_type: String, // "vptree" (default) oder "kdtree"
}

#[pymethods]
impl SDOParams {
    #[new]
    #[pyo3(signature = (k, x, rho, distance = "euclidean".to_string(), minkowski_p = None, tree_type = "vptree".to_string()))]
    pub fn new(
        k: usize,
        x: usize,
        rho: f64,
        distance: String,
        minkowski_p: Option<f64>,
        tree_type: String,
    ) -> Self {
        Self {
            k,
            x,
            rho,
            distance,
            minkowski_p,
            tree_type,
        }
    }
}

impl SDOParams {
    fn get_metric(&self) -> DistanceMetric {
        match self.distance.to_lowercase().as_str() {
            "manhattan" => DistanceMetric::Manhattan,
            "chebyshev" => DistanceMetric::Chebyshev,
            "minkowski" => DistanceMetric::Minkowski,
            _ => DistanceMetric::Euclidean,
        }
    }

    fn get_tree_type(&self) -> TreeType {
        match self.tree_type.to_lowercase().as_str() {
            "kdtree" => TreeType::KdTree,
            _ => TreeType::VpTree, // Default: VpTree
        }
    }
}

/// Sparse Data Observers (SDO) Algorithm
#[pyclass]
#[allow(clippy::upper_case_acronyms)]
pub struct SDO {
    // Entferne #[pyo3(get, set)] um Konflikt mit get_active_observers() zu vermeiden
    active_observers: Vec<Observer>,
    tree_observers: Option<SpatialTree>,
    distance_metric: DistanceMetric,
    minkowski_p: Option<f64>,
    #[pyo3(get, set)]
    x: usize,
}

#[pymethods]
impl SDO {
    #[new]
    pub fn new() -> Self {
        Self {
            active_observers: Vec::new(),
            tree_observers: None,
            distance_metric: DistanceMetric::Euclidean,
            minkowski_p: None,
            x: 10,
        }
    }

    /// Lernt das Modell aus den Daten
    pub fn learn(&mut self, data: PyReadonlyArray2<f64>, params: &SDOParams) -> PyResult<()> {
        let data_slice = data.as_array();
        let rows = data_slice.nrows();
        let cols = data_slice.ncols();

        if rows == 0 || params.k == 0 {
            return Ok(());
        }

        // Konvertiere NumPy-Array zu Vec<Vec<f64>>
        let data_vec: Vec<Vec<f64>> = (0..rows)
            .map(|i| (0..cols).map(|j| data_slice[[i, j]]).collect())
            .collect();

        self.x = params.x;
        self.distance_metric = params.get_metric();
        self.minkowski_p = params.minkowski_p;

        // Schritt 1: Sample
        let mut rng = thread_rng();
        let observers: Vec<Vec<f64>> = data_vec
            .choose_multiple(&mut rng, params.k.min(data_vec.len()))
            .cloned()
            .collect();

        // Schritt 2: Observe
        let mut observations: Vec<(usize, usize, f64)> = observers
            .iter()
            .enumerate()
            .map(|(idx, observer)| {
                let count = self.count_points_in_neighborhood(observer, &data_vec, params.x);
                (idx, count, count as f64)
            })
            .collect();

        // Schritt 3: Clean model
        observations.sort_by_key(|&(_, count, _)| count);
        let num_active = ((observers.len() as f64) * (1.0 - params.rho)).ceil() as usize;
        let num_active = num_active.max(1).min(observers.len());

        let metric = params.get_metric();
        let minkowski_p = params.minkowski_p;
        let tree_type = params.get_tree_type();

        self.active_observers = observations
            .iter()
            .rev()
            .take(num_active)
            .enumerate()
            .map(|(new_idx, &(old_idx, _, obs_count))| Observer {
                data: observers[old_idx].clone(),
                observations: obs_count,
                index: new_idx,
            })
            .collect();

        // Baue Tree mit entsprechenden Metrik-Wrappern
        self.tree_observers = match (tree_type, metric) {
            (TreeType::VpTree, DistanceMetric::Euclidean) => {
                let points: Vec<Euclidean<Vec<f64>>> = self
                    .active_observers
                    .iter()
                    .map(|obs| Euclidean(obs.data.clone()))
                    .collect();
                Some(SpatialTree::VpEuclidean(VpTree::balanced(points)))
            }
            (TreeType::VpTree, DistanceMetric::Manhattan) => {
                let points: Vec<Taxicab<Vec<f64>>> = self
                    .active_observers
                    .iter()
                    .map(|obs| Taxicab(obs.data.clone()))
                    .collect();
                Some(SpatialTree::VpManhattan(VpTree::balanced(points)))
            }
            (TreeType::VpTree, DistanceMetric::Chebyshev) => {
                let points: Vec<Chebyshev<Vec<f64>>> = self
                    .active_observers
                    .iter()
                    .map(|obs| Chebyshev(obs.data.clone()))
                    .collect();
                Some(SpatialTree::VpChebyshev(VpTree::balanced(points)))
            }
            (TreeType::VpTree, DistanceMetric::Minkowski) => {
                let p = minkowski_p.unwrap_or(3.0);
                let points: Vec<Minkowski> = self
                    .active_observers
                    .iter()
                    .map(|obs| Minkowski::new(obs.data.clone(), p))
                    .collect();
                Some(SpatialTree::VpMinkowski(VpTree::balanced(points)))
            }
            (TreeType::KdTree, DistanceMetric::Euclidean) => {
                let points: Vec<Euclidean<Vec<f64>>> = self
                    .active_observers
                    .iter()
                    .map(|obs| Euclidean(obs.data.clone()))
                    .collect();
                Some(SpatialTree::KdEuclidean(KdTree::from_iter(points)))
            }
            (TreeType::KdTree, DistanceMetric::Manhattan) => {
                let points: Vec<Taxicab<Vec<f64>>> = self
                    .active_observers
                    .iter()
                    .map(|obs| Taxicab(obs.data.clone()))
                    .collect();
                Some(SpatialTree::KdManhattan(KdTree::from_iter(points)))
            }
            (TreeType::KdTree, DistanceMetric::Chebyshev) => {
                let points: Vec<Chebyshev<Vec<f64>>> = self
                    .active_observers
                    .iter()
                    .map(|obs| Chebyshev(obs.data.clone()))
                    .collect();
                Some(SpatialTree::KdChebyshev(KdTree::from_iter(points)))
            }
            (TreeType::KdTree, DistanceMetric::Minkowski) => {
                let p = minkowski_p.unwrap_or(3.0);
                let points: Vec<Minkowski> = self
                    .active_observers
                    .iter()
                    .map(|obs| Minkowski::new(obs.data.clone(), p))
                    .collect();
                Some(SpatialTree::KdMinkowski(KdTree::from_iter(points)))
            }
        };

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

        // Verwende Brute-Force mit gewählter Distanzfunktion
        // (kd-tree wird für andere Metriken als Euclidean nicht optimal genutzt)
        let mut distances: Vec<f64> = self
            .active_observers
            .iter()
            .map(|observer| self.compute_distance(&point_vec, &observer.data))
            .collect();

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let k_actual = k.min(distances.len());
        let relevant_distances: Vec<f64> = distances.into_iter().take(k_actual).collect();

        if relevant_distances.is_empty() {
            return Ok(f64::INFINITY);
        }

        let mid = relevant_distances.len() / 2;
        let median = if relevant_distances.len().is_multiple_of(2) && mid > 0 {
            (relevant_distances[mid - 1] + relevant_distances[mid]) / 2.0
        } else {
            relevant_distances[mid]
        };

        Ok(median)
    }

    /// Konvertiert active_observers zu NumPy-Array für Python
    pub fn get_active_observers(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        if self.active_observers.is_empty() {
            let array = PyArray2::zeros_bound(py, (0, 0), false);
            return Ok(array.unbind());
        }

        let rows = self.active_observers.len();
        let cols = self.active_observers[0].data.len();
        let array = PyArray2::zeros_bound(py, (rows, cols), false);

        unsafe {
            let mut array_mut = array.as_array_mut();
            for (i, observer) in self.active_observers.iter().enumerate() {
                for (j, &value) in observer.data.iter().enumerate() {
                    array_mut[[i, j]] = value;
                }
            }
        }

        Ok(array.unbind())
    }
}

impl SDO {
    /// Interne Methode, um active_observers zu erhalten (für SDOclust)
    pub(crate) fn get_active_observers_internal(&self) -> Vec<Vec<f64>> {
        self.active_observers
            .iter()
            .map(|obs| obs.data.clone())
            .collect()
    }

    /// Interne Methode, um x zu erhalten (für SDOclust)
    pub(crate) fn get_x_internal(&self) -> usize {
        self.x
    }

    /// Interne Methode, um distance_metric zu erhalten (für SDOclust)
    pub(crate) fn get_distance_metric_internal(&self) -> DistanceMetric {
        self.distance_metric
    }

    /// Interne Methode, um minkowski_p zu erhalten (für SDOclust)
    pub(crate) fn get_minkowski_p_internal(&self) -> Option<f64> {
        self.minkowski_p
    }

    fn compute_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        compute_distance(a, b, self.distance_metric, self.minkowski_p)
    }

    fn count_points_in_neighborhood(&self, observer: &[f64], data: &[Vec<f64>], x: usize) -> usize {
        let mut distances: Vec<(usize, f64)> = data
            .iter()
            .enumerate()
            .map(|(idx, point)| {
                let dist = self.compute_distance(observer, point);
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
}

impl Default for SDO {
    fn default() -> Self {
        Self::new()
    }
}
