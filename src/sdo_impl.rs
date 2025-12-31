use acap::kd::KdTree;
use acap::Proximity;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::f64;

use crate::utils::{compute_distance, Point};

/// Distanzfunktion für SDO
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum DistanceMetric {
    Euclidean = 0,
    Manhattan = 1,
    Chebyshev = 2,
    Minkowski = 3,
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
}

#[pymethods]
impl SDOParams {
    #[new]
    #[pyo3(signature = (k, x, rho, distance = "euclidean".to_string(), minkowski_p = None))]
    pub fn new(k: usize, x: usize, rho: f64, distance: String, minkowski_p: Option<f64>) -> Self {
        Self {
            k,
            x,
            rho,
            distance,
            minkowski_p,
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
}

// Implementiere Proximity mit euklidischer Distanz (für kd-tree)
impl Proximity<Point> for Point {
    type Distance = f64;

    fn distance(&self, other: &Point) -> Self::Distance {
        // Euklidische Distanz für kd-tree (kann später durch gewählte Metrik ersetzt werden)
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// Sparse Data Observers (SDO) Algorithm
#[pyclass]
pub struct SDO {
    // Entferne #[pyo3(get, set)] um Konflikt mit get_active_observers() zu vermeiden
    active_observers: Vec<Vec<f64>>,
    kdtree: Option<KdTree<Point>>,
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
            kdtree: None,
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
        let mut observations: Vec<(usize, usize)> = observers
            .iter()
            .enumerate()
            .map(|(idx, observer)| {
                let count = self.count_points_in_neighborhood(observer, &data_vec, params.x);
                (idx, count)
            })
            .collect();

        // Schritt 3: Clean model
        observations.sort_by_key(|&(_, count)| count);
        let num_active = ((observers.len() as f64) * (1.0 - params.rho)).ceil() as usize;
        let num_active = num_active.max(1).min(observers.len());

        self.active_observers = observations
            .iter()
            .rev()
            .take(num_active)
            .map(|&(idx, _)| observers[idx].clone())
            .collect();

        // Baue kd-tree mit acap
        let points: Vec<Point> = self
            .active_observers
            .iter()
            .map(|v| Point(v.clone()))
            .collect();
        self.kdtree = Some(KdTree::from_iter(points));

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
            .map(|observer| self.compute_distance(&point_vec, observer))
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
