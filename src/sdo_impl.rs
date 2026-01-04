use acap::chebyshev::Chebyshev;
use acap::distance::Distance;
use acap::euclid::Euclidean;
use acap::kd::KdTree;
use acap::knn::NearestNeighbors;
use acap::taxi::Taxicab;
use acap::vp::VpTree;
use std::cell::RefCell;
use std::sync::Arc;

use crate::utils::Lp;
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::f64;

use crate::utils::{
    compute_distance, Observer, ObserverChebyshev, ObserverEuclidean, ObserverManhattan,
    ObserverMinkowski, ObserverSet,
};

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

use crate::utils::ArcVec;

/// Enum für beide Tree-Typen mit verschiedenen Metriken
/// Verwendet ArcVec für geteilten Besitz, damit Trees Referenzen halten können
#[allow(dead_code)]
pub(crate) enum SpatialTree {
    VpEuclidean(VpTree<Euclidean<ArcVec>>),
    VpManhattan(VpTree<Taxicab<ArcVec>>),
    VpChebyshev(VpTree<Chebyshev<ArcVec>>),
    VpMinkowski(VpTree<Lp>),
    KdEuclidean(KdTree<Euclidean<ArcVec>>),
    KdManhattan(KdTree<Taxicab<ArcVec>>),
    KdChebyshev(KdTree<Chebyshev<ArcVec>>),
    KdMinkowski(KdTree<Lp>),
}

/// Enum für Trees, die direkt auf Observer-Objekten arbeiten
/// Ermöglicht direkten Zugriff auf Observer-Index ohne Suche
#[allow(dead_code)]
pub(crate) enum SpatialTreeObserver {
    VpEuclidean(VpTree<ObserverEuclidean>),
    VpManhattan(VpTree<ObserverManhattan>),
    VpChebyshev(VpTree<ObserverChebyshev>),
    VpMinkowski(VpTree<ObserverMinkowski>),
    KdEuclidean(KdTree<ObserverEuclidean>),
    KdManhattan(KdTree<ObserverManhattan>),
    KdChebyshev(KdTree<ObserverChebyshev>),
    KdMinkowski(KdTree<ObserverMinkowski>),
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
    // Observer-Set, sortiert nach observations
    observers: ObserverSet,
    tree_observers: Option<SpatialTree>,
    #[allow(dead_code)] // Wird von SDOstream verwendet
    pub(crate) tree_active_observers: RefCell<Option<SpatialTreeObserver>>,
    distance_metric: DistanceMetric,
    minkowski_p: Option<f64>,
    tree_type: TreeType,
    rho: f64,
    #[pyo3(get, set)]
    pub(crate) x: usize,
}

#[pymethods]
impl SDO {
    #[new]
    pub fn new() -> Self {
        Self {
            observers: ObserverSet::new(),
            tree_observers: None,
            tree_active_observers: RefCell::new(None),
            distance_metric: DistanceMetric::Euclidean,
            minkowski_p: None,
            tree_type: TreeType::VpTree,
            rho: 0.1,
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
        self.tree_type = params.get_tree_type();
        self.rho = params.rho;
        // Reset tree_active_observers, da sich die Parameter geändert haben
        *self.tree_active_observers.borrow_mut() = None;

        // Schritt 1: Sample
        let mut rng = thread_rng();
        let observers_data: Vec<Vec<f64>> = data_vec
            .choose_multiple(&mut rng, params.k.min(data_vec.len()))
            .cloned()
            .collect();

        let metric = params.get_metric();
        let minkowski_p = params.minkowski_p;
        let tree_type = params.get_tree_type();

        // Schritt 2: Erstelle Tree mit allen Observers (ohne observations)
        self.tree_observers = self.build_tree(&observers_data, metric, minkowski_p, tree_type);

        // Schritt 3: Berechne observations für jeden Observer mit Nearest Neighbor Search
        let mut observer_list: Vec<Observer> = Vec::new();
        for (idx, observer_data) in observers_data.iter().enumerate() {
            let count =
                self.count_points_in_neighborhood_with_tree(observer_data, &data_vec, params.x);
            observer_list.push(Observer {
                data: observer_data.clone(),
                observations: count as f64,
                age: 1.0, // Start-Alter
                index: idx,
            });
        }

        // Schritt 4: Füge alle Observer zum ObserverSet hinzu (sortiert nach observations)
        self.observers = ObserverSet::new();
        for observer in observer_list {
            self.observers.insert(observer);
        }

        Ok(())
    }

    /// Berechnet den Outlier-Score für einen Datenpunkt
    pub fn predict(&self, point: PyReadonlyArray2<f64>) -> PyResult<f64> {
        if self.observers.is_empty() {
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

        // Lazy: Berechne aktive Observer nur wenn nötig
        let num_active = ((self.observers.len() as f64) * (1.0 - self.rho)).ceil() as usize;
        let num_active = num_active.max(1).min(self.observers.len());
        let active_observers = self.observers.get_active(num_active);

        if active_observers.is_empty() {
            return Ok(f64::INFINITY);
        }

        let k = self.x.min(active_observers.len());

        // Versuche Tree-basierte Nearest Neighbor Search zu verwenden
        // Erstelle tree_active_observers lazy, wenn noch nicht vorhanden
        self.ensure_tree_active_observers();

        let distances = {
            let tree_opt = self.tree_active_observers.borrow();
            if let Some(ref tree) = *tree_opt {
                // Verwende Tree für k-nearest neighbors (SpatialTreeObserver)
                self.predict_with_tree_observer(&point_vec, tree, k)
            } else {
                // Give a warning that the tree is not built
                eprintln!("Warning: Tree is not built. Using brute-force instead.");
                // Fallback: Brute-Force mit gewählter Distanzfunktion
                let mut distances: Vec<f64> = active_observers
                    .iter()
                    .map(|observer| self.compute_distance(&point_vec, &observer.data))
                    .collect();
                distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
                distances.into_iter().take(k).collect()
            }
        };

        if distances.is_empty() {
            return Ok(f64::INFINITY);
        }

        let mid = distances.len() / 2;
        let median = if distances.len().is_multiple_of(2) && mid > 0 {
            (distances[mid - 1] + distances[mid]) / 2.0
        } else {
            distances[mid]
        };

        Ok(median)
    }

    /// Konvertiert active_observers zu NumPy-Array für Python
    pub fn get_active_observers(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        let num_active = ((self.observers.len() as f64) * (1.0 - self.rho)).ceil() as usize;
        let num_active = num_active.max(1).min(self.observers.len());
        let active_observers = self.observers.get_active(num_active);

        if active_observers.is_empty() {
            let array = PyArray2::zeros_bound(py, (0, 0), false);
            return Ok(array.unbind());
        }

        let rows = active_observers.len();
        let cols = active_observers[0].data.len();
        let array = PyArray2::zeros_bound(py, (rows, cols), false);

        unsafe {
            let mut array_mut = array.as_array_mut();
            for (i, observer) in active_observers.iter().enumerate() {
                for (j, &value) in observer.data.iter().enumerate() {
                    array_mut[[i, j]] = value;
                }
            }
        }

        Ok(array.unbind())
    }
}

// Private Hilfsmethoden (nicht für Python)
impl SDO {
    /// Baut einen Tree aus Observer-Daten (verwendet Rc für Referenzen)
    fn build_tree(
        &self,
        observers_data: &[Vec<f64>],
        metric: DistanceMetric,
        minkowski_p: Option<f64>,
        tree_type: TreeType,
    ) -> Option<SpatialTree> {
        match (tree_type, metric) {
            (TreeType::VpTree, DistanceMetric::Euclidean) => {
                let points: Vec<Euclidean<ArcVec>> = observers_data
                    .iter()
                    .map(|data| Euclidean(ArcVec(Arc::new(data.clone()))))
                    .collect();
                Some(SpatialTree::VpEuclidean(VpTree::balanced(points)))
            }
            (TreeType::VpTree, DistanceMetric::Manhattan) => {
                let points: Vec<Taxicab<ArcVec>> = observers_data
                    .iter()
                    .map(|data| Taxicab(ArcVec(Arc::new(data.clone()))))
                    .collect();
                Some(SpatialTree::VpManhattan(VpTree::balanced(points)))
            }
            (TreeType::VpTree, DistanceMetric::Chebyshev) => {
                let points: Vec<Chebyshev<ArcVec>> = observers_data
                    .iter()
                    .map(|data| Chebyshev(ArcVec(Arc::new(data.clone()))))
                    .collect();
                Some(SpatialTree::VpChebyshev(VpTree::balanced(points)))
            }
            (TreeType::VpTree, DistanceMetric::Minkowski) => {
                let p = minkowski_p.unwrap_or(3.0);
                let points: Vec<Lp> = observers_data
                    .iter()
                    .map(|data| Lp::new(ArcVec(Arc::new(data.clone())), p))
                    .collect();
                Some(SpatialTree::VpMinkowski(VpTree::balanced(points)))
            }
            (TreeType::KdTree, DistanceMetric::Euclidean) => {
                let points: Vec<Euclidean<ArcVec>> = observers_data
                    .iter()
                    .map(|data| Euclidean(ArcVec(Arc::new(data.clone()))))
                    .collect();
                Some(SpatialTree::KdEuclidean(KdTree::from_iter(points)))
            }
            (TreeType::KdTree, DistanceMetric::Manhattan) => {
                let points: Vec<Taxicab<ArcVec>> = observers_data
                    .iter()
                    .map(|data| Taxicab(ArcVec(Arc::new(data.clone()))))
                    .collect();
                Some(SpatialTree::KdManhattan(KdTree::from_iter(points)))
            }
            (TreeType::KdTree, DistanceMetric::Chebyshev) => {
                let points: Vec<Chebyshev<ArcVec>> = observers_data
                    .iter()
                    .map(|data| Chebyshev(ArcVec(Arc::new(data.clone()))))
                    .collect();
                Some(SpatialTree::KdChebyshev(KdTree::from_iter(points)))
            }
            (TreeType::KdTree, DistanceMetric::Minkowski) => {
                let p = minkowski_p.unwrap_or(3.0);
                let points: Vec<Lp> = observers_data
                    .iter()
                    .map(|data| Lp::new(ArcVec(Arc::new(data.clone())), p))
                    .collect();
                Some(SpatialTree::KdMinkowski(KdTree::from_iter(points)))
            }
        }
    }

    /// Baut einen Tree aus Observer-Objekten (mit Index-Information)
    fn build_tree_from_observers(
        &self,
        observers: &[Observer],
        metric: DistanceMetric,
        minkowski_p: Option<f64>,
        tree_type: TreeType,
    ) -> Option<SpatialTreeObserver> {
        if observers.is_empty() {
            return None;
        }

        match (tree_type, metric) {
            (TreeType::VpTree, DistanceMetric::Euclidean) => {
                let points: Vec<ObserverEuclidean> = observers
                    .iter()
                    .map(|obs| ObserverEuclidean(obs.clone()))
                    .collect();
                Some(SpatialTreeObserver::VpEuclidean(VpTree::balanced(points)))
            }
            (TreeType::VpTree, DistanceMetric::Manhattan) => {
                let points: Vec<ObserverManhattan> = observers
                    .iter()
                    .map(|obs| ObserverManhattan(obs.clone()))
                    .collect();
                Some(SpatialTreeObserver::VpManhattan(VpTree::balanced(points)))
            }
            (TreeType::VpTree, DistanceMetric::Chebyshev) => {
                let points: Vec<ObserverChebyshev> = observers
                    .iter()
                    .map(|obs| ObserverChebyshev(obs.clone()))
                    .collect();
                Some(SpatialTreeObserver::VpChebyshev(VpTree::balanced(points)))
            }
            (TreeType::VpTree, DistanceMetric::Minkowski) => {
                let p = minkowski_p.unwrap_or(3.0);
                let points: Vec<ObserverMinkowski> = observers
                    .iter()
                    .map(|obs| ObserverMinkowski {
                        observer: obs.clone(),
                        p,
                    })
                    .collect();
                Some(SpatialTreeObserver::VpMinkowski(VpTree::balanced(points)))
            }
            (TreeType::KdTree, DistanceMetric::Euclidean) => {
                let points: Vec<ObserverEuclidean> = observers
                    .iter()
                    .map(|obs| ObserverEuclidean(obs.clone()))
                    .collect();
                Some(SpatialTreeObserver::KdEuclidean(KdTree::from_iter(points)))
            }
            (TreeType::KdTree, DistanceMetric::Manhattan) => {
                let points: Vec<ObserverManhattan> = observers
                    .iter()
                    .map(|obs| ObserverManhattan(obs.clone()))
                    .collect();
                Some(SpatialTreeObserver::KdManhattan(KdTree::from_iter(points)))
            }
            (TreeType::KdTree, DistanceMetric::Chebyshev) => {
                let points: Vec<ObserverChebyshev> = observers
                    .iter()
                    .map(|obs| ObserverChebyshev(obs.clone()))
                    .collect();
                Some(SpatialTreeObserver::KdChebyshev(KdTree::from_iter(points)))
            }
            (TreeType::KdTree, DistanceMetric::Minkowski) => {
                let p = minkowski_p.unwrap_or(3.0);
                let points: Vec<ObserverMinkowski> = observers
                    .iter()
                    .map(|obs| ObserverMinkowski {
                        observer: obs.clone(),
                        p,
                    })
                    .collect();
                Some(SpatialTreeObserver::KdMinkowski(KdTree::from_iter(points)))
            }
        }
    }

    /// Zählt Punkte in der Nachbarschaft mit Tree-basierter Nearest Neighbor Search
    fn count_points_in_neighborhood_with_tree(
        &self,
        observer: &[f64],
        _data: &[Vec<f64>],
        x: usize,
    ) -> usize {
        if let Some(ref tree) = self.tree_observers {
            // Verwende Tree für Nearest Neighbor Search
            let k = x;
            let observer_arc = ArcVec(Arc::new(observer.to_vec()));
            match tree {
                SpatialTree::VpEuclidean(ref t) => {
                    let query = Euclidean(observer_arc.clone());
                    t.k_nearest(&query, k).len()
                }
                SpatialTree::VpManhattan(ref t) => {
                    let query = Taxicab(observer_arc.clone());
                    t.k_nearest(&query, k).len()
                }
                SpatialTree::VpChebyshev(ref t) => {
                    let query = Chebyshev(observer_arc.clone());
                    t.k_nearest(&query, k).len()
                }
                SpatialTree::VpMinkowski(ref t) => {
                    let p = self.minkowski_p.unwrap_or(3.0);
                    let query = Lp::new(observer_arc.clone(), p);
                    t.k_nearest(&query, k).len()
                }
                SpatialTree::KdEuclidean(ref t) => {
                    let query = Euclidean(observer_arc.clone());
                    t.k_nearest(&query, k).len()
                }
                SpatialTree::KdManhattan(ref t) => {
                    let query = Taxicab(observer_arc.clone());
                    t.k_nearest(&query, k).len()
                }
                SpatialTree::KdChebyshev(ref t) => {
                    let query = Chebyshev(observer_arc.clone());
                    t.k_nearest(&query, k).len()
                }
                SpatialTree::KdMinkowski(ref t) => {
                    let p = self.minkowski_p.unwrap_or(3.0);
                    let query = Lp::new(observer_arc.clone(), p);
                    t.k_nearest(&query, k).len()
                }
            }
        } else {
            // Fallback zu Brute-Force
            self.count_points_in_neighborhood(observer, _data, x)
        }
    }
}

impl SDO {
    /// Erstellt tree_active_observers lazy, wenn noch nicht vorhanden
    pub(crate) fn ensure_tree_active_observers(&self) {
        if self.tree_active_observers.borrow().is_none() {
            let num_active = ((self.observers.len() as f64) * (1.0 - self.rho)).ceil() as usize;
            let num_active = num_active.max(1).min(self.observers.len());
            let active_observers = self.observers.get_active(num_active);
            *self.tree_active_observers.borrow_mut() = self.build_tree_from_observers(
                &active_observers,
                self.distance_metric,
                self.minkowski_p,
                self.tree_type,
            );
        }
    }

    /// Verwendet Tree für k-nearest neighbors und gibt Distanzen zurück (für SpatialTreeObserver)
    fn predict_with_tree_observer(
        &self,
        point: &[f64],
        tree: &SpatialTreeObserver,
        k: usize,
    ) -> Vec<f64> {
        let point_observer = Observer {
            data: point.to_vec(),
            observations: 0.0,
            age: 0.0,
            index: 0,
        };
        match tree {
            SpatialTreeObserver::VpEuclidean(ref t) => {
                let query = ObserverEuclidean(point_observer);
                t.k_nearest(&query, k)
                    .into_iter()
                    .map(|neighbor| neighbor.distance.value())
                    .collect()
            }
            SpatialTreeObserver::VpManhattan(ref t) => {
                let query = ObserverManhattan(point_observer.clone());
                t.k_nearest(&query, k)
                    .into_iter()
                    .map(|neighbor| neighbor.distance)
                    .collect()
            }
            SpatialTreeObserver::VpChebyshev(ref t) => {
                let query = ObserverChebyshev(point_observer.clone());
                t.k_nearest(&query, k)
                    .into_iter()
                    .map(|neighbor| neighbor.distance)
                    .collect()
            }
            SpatialTreeObserver::VpMinkowski(ref t) => {
                let p = self.minkowski_p.unwrap_or(3.0);
                let query = ObserverMinkowski {
                    observer: point_observer.clone(),
                    p,
                };
                t.k_nearest(&query, k)
                    .into_iter()
                    .map(|neighbor| neighbor.distance)
                    .collect()
            }
            SpatialTreeObserver::KdEuclidean(ref t) => {
                let query = ObserverEuclidean(point_observer);
                t.k_nearest(&query, k)
                    .into_iter()
                    .map(|neighbor| neighbor.distance.value())
                    .collect()
            }
            SpatialTreeObserver::KdManhattan(ref t) => {
                let query = ObserverManhattan(point_observer.clone());
                t.k_nearest(&query, k)
                    .into_iter()
                    .map(|neighbor| neighbor.distance)
                    .collect()
            }
            SpatialTreeObserver::KdChebyshev(ref t) => {
                let query = ObserverChebyshev(point_observer.clone());
                t.k_nearest(&query, k)
                    .into_iter()
                    .map(|neighbor| neighbor.distance)
                    .collect()
            }
            SpatialTreeObserver::KdMinkowski(ref t) => {
                let p = self.minkowski_p.unwrap_or(3.0);
                let query = ObserverMinkowski {
                    observer: point_observer.clone(),
                    p,
                };
                t.k_nearest(&query, k)
                    .into_iter()
                    .map(|neighbor| neighbor.distance)
                    .collect()
            }
        }
    }

    /// Interne Methode, um active_observers zu erhalten (für SDOclust)
    pub(crate) fn get_active_observers_internal(&self) -> Vec<Vec<f64>> {
        let num_active = ((self.observers.len() as f64) * (1.0 - self.rho)).ceil() as usize;
        let num_active = num_active.max(1).min(self.observers.len());
        self.observers
            .get_active(num_active)
            .iter()
            .map(|obs| obs.data.clone())
            .collect()
    }

    /// Interne Methode, um active_observers mit Indizes zu erhalten (für SDOclust)
    /// Gibt Observer-Objekte zurück, wobei index die Position in der active_observers Liste ist
    pub(crate) fn get_active_observers_with_indices(&self) -> Vec<Observer> {
        let num_active = ((self.observers.len() as f64) * (1.0 - self.rho)).ceil() as usize;
        let num_active = num_active.max(1).min(self.observers.len());
        self.observers
            .get_active(num_active)
            .iter()
            .enumerate()
            .map(|(idx, obs)| Observer {
                data: obs.data.clone(),
                observations: obs.observations,
                age: obs.age,
                index: idx, // Index in active_observers Liste
            })
            .collect()
    }

    /// Erstellt und gibt tree_active_observers zurück (für SDOclust)
    /// Diese Funktion stellt sicher, dass der Tree erstellt ist und gibt eine Referenz zurück
    pub(crate) fn get_tree_active_observers(
        &self,
    ) -> Option<std::cell::Ref<'_, Option<SpatialTreeObserver>>> {
        self.ensure_tree_active_observers();
        Some(self.tree_active_observers.borrow())
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

    /// Interne Methode, um tree_type zu erhalten (für SDOclust)
    pub(crate) fn get_tree_type_internal(&self) -> TreeType {
        self.tree_type
    }

    pub(crate) fn compute_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        compute_distance(a, b, self.distance_metric, self.minkowski_p)
    }

    /// Interne Methode, um Observer-observations zu aktualisieren (für SDOstream)
    pub(crate) fn update_observer_observations(
        &mut self,
        index: usize,
        new_observations: f64,
    ) -> bool {
        self.observers.update_observations(index, new_observations)
    }

    /// Interne Methode, um Observer-observations und age zu aktualisieren (für SDOstream)
    #[allow(dead_code)]
    pub(crate) fn update_observer_with_age(
        &mut self,
        index: usize,
        new_observations: f64,
        new_age: f64,
    ) -> bool {
        self.observers
            .update_observer(index, new_observations, new_age)
    }

    /// Interne Methode, um Observer-age zu aktualisieren (für SDOstream)
    pub(crate) fn update_observer_age(&mut self, index: usize, new_age: f64) -> bool {
        // Hole aktuellen Observer über get_active_observers_with_indices
        let observers = self.get_active_observers_with_indices();
        if let Some(observer) = observers.iter().find(|obs| obs.index == index) {
            self.observers
                .update_observer(index, observer.observations, new_age)
        } else {
            false
        }
    }

    /// Interne Methode, um einen Observer zu ersetzen (für SDOstream)
    pub(crate) fn replace_observer(&mut self, old_index: usize, new_observer: Observer) -> bool {
        self.observers.replace(old_index, new_observer)
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
