use kiddo::float::kdtree::KdTree;
use kiddo::SquaredEuclidean;
use std::cell::RefCell;
use std::sync::Arc;

use crate::obs::Observer;
use crate::utils::DistanceMetric;

/// Räumliche Indexierung: kiddo KD-Tree nur für euklidische Distanz
/// Für andere Metriken wird kein Baum verwendet (Brute-Force)
/// kiddo ist für euklidische Distanzen optimiert und unterstützt dynamische Updates
/// Maximale Dimension: 100 (kiddo benötigt Dimension zur Compile-Zeit)
/// B = 64: Bucket-Größe für den KD-Tree (erhöht für Duplikate)
/// IDX = u32: Index-Typ für interne Strukturen
pub(crate) enum SpatialTreeObserver {
    /// kiddo KD-Tree für euklidische Distanz (bis zu 100 Dimensionen)
    Euclidean(KdTree<f64, usize, 100, 64, u32>),
    /// Kein Baum für nicht-euklidische Metriken (Brute-Force wird verwendet)
    NonEuclidean,
}

/// Tree-bezogene Funktionalität für ObserverSet
/// Diese Funktionen werden von ObserverSet verwendet, wenn Tree-basierte Suche aktiviert ist
pub(crate) struct ObserverSetTree {
    pub(crate) spatial_tree: RefCell<Option<SpatialTreeObserver>>,
    pub(crate) spatial_tree_active: RefCell<Option<SpatialTreeObserver>>,
}

impl ObserverSetTree {
    pub(crate) fn new() -> Self {
        Self {
            spatial_tree: RefCell::new(None),
            spatial_tree_active: RefCell::new(None),
        }
    }

    /// Invalidiert beide Bäume
    pub(crate) fn invalidate(&self) {
        if let Ok(mut tree_opt) = self.spatial_tree.try_borrow_mut() {
            *tree_opt = None;
        }
        if let Ok(mut tree_opt) = self.spatial_tree_active.try_borrow_mut() {
            *tree_opt = None;
        }
    }

    /// Stellt sicher, dass der Spatial Tree gebaut ist (lazy initialization)
    /// Nur für euklidische Distanz wird ein kiddo KD-Tree gebaut
    pub(crate) fn ensure_spatial_tree(
        &self,
        active: bool,
        observers: &[Arc<Observer>],
        distance_metric: DistanceMetric,
    ) {
        if distance_metric != DistanceMetric::Euclidean {
            return; // Kein Baum für nicht-euklidische Metriken
        }

        // Prüfe zuerst, ob der Baum bereits existiert (immutable borrow)
        if let Ok(tree_opt) = if active {
            self.spatial_tree_active.try_borrow()
        } else {
            self.spatial_tree.try_borrow()
        } {
            if tree_opt.is_some() {
                return; // Baum existiert bereits
            }
        } else {
            return; // RefCell ist bereits geborrowt
        }

        // Jetzt mutable borrow für den Build
        if let Ok(mut tree_opt) = if active {
            self.spatial_tree_active.try_borrow_mut()
        } else {
            self.spatial_tree.try_borrow_mut()
        } {
            if tree_opt.is_none() {
                *tree_opt = Self::build_tree_from_observers(observers, distance_metric);
            }
        }
    }

    /// Build kiddo KD-Tree from observers (nur für euklidische Distanz)
    fn build_tree_from_observers(
        observers: &[Arc<Observer>],
        metric: DistanceMetric,
    ) -> Option<SpatialTreeObserver> {
        if observers.is_empty() {
            return None;
        }

        match metric {
            DistanceMetric::Euclidean => {
                let mut kdtree: KdTree<f64, usize, 100, 64, u32> = KdTree::new();

                for arc in observers {
                    let data = &arc.data;
                    // Prüfe Dimension
                    if data.len() > 100 {
                        continue; // Überspringe Observer mit zu vielen Dimensionen
                    }

                    // Erstelle Punkt-Array für kiddo (pad auf 100 Dimensionen)
                    let mut padded_point = [0.0; 100];
                    for (i, &val) in data.iter().take(100).enumerate() {
                        padded_point[i] = val;
                    }

                    // kiddo API: add(point, data) - data ist der Index
                    kdtree.add(&padded_point, arc.index);
                }

                Some(SpatialTreeObserver::Euclidean(kdtree))
            }
            _ => {
                // Für andere Metriken: kein Baum, Brute-Force wird verwendet
                Some(SpatialTreeObserver::NonEuclidean)
            }
        }
    }

    /// Insert observer into kiddo KD-Tree incrementally (nur für euklidische Distanz)
    pub(crate) fn insert_into_tree(
        &self,
        index: usize,
        data: &[f64],
        distance_metric: DistanceMetric,
    ) {
        if distance_metric != DistanceMetric::Euclidean {
            return; // Kein Baum für nicht-euklidische Metriken
        }

        // Prüfe Dimension - kiddo unterstützt bis zu 100 Dimensionen
        if data.len() > 100 {
            return; // Zu viele Dimensionen für kiddo
        }

        let mut tree_opt = self.spatial_tree.borrow_mut();
        if let Some(SpatialTreeObserver::Euclidean(ref mut kdtree)) = *tree_opt {
            // kiddo API: add(point, data) - data ist der Index
            let mut padded_point = [0.0; 100];
            for (i, &val) in data.iter().take(100).enumerate() {
                padded_point[i] = val;
            }
            kdtree.add(&padded_point, index);
        }
    }

    /// Remove observer from kiddo KD-Tree incrementally (nur für euklidische Distanz)
    pub(crate) fn remove_from_tree(
        &self,
        index: usize,
        data: &[f64],
        distance_metric: DistanceMetric,
    ) {
        if distance_metric != DistanceMetric::Euclidean {
            return; // Kein Baum für nicht-euklidische Metriken
        }

        let mut tree_opt = self.spatial_tree.borrow_mut();
        if let Some(SpatialTreeObserver::Euclidean(ref mut kdtree)) = *tree_opt {
            if data.len() > 100 {
                return; // Zu viele Dimensionen
            }

            // Erstelle Punkt-Array für kiddo
            let mut padded_point = [0.0; 100];
            for (i, &val) in data.iter().take(100).enumerate() {
                padded_point[i] = val;
            }

            // kiddo API: remove(point, item) entfernt den Punkt mit dem gegebenen Index
            kdtree.remove(&padded_point, index);
        }
    }

    /// Perform k-nearest neighbor search using tree
    /// Returns distances to k nearest neighbors
    pub(crate) fn search_k_nearest_distances_tree(
        &self,
        query_point: &[f64],
        k: usize,
        active: bool,
        distance_metric: DistanceMetric,
        observers: &[Arc<Observer>],
    ) -> Option<Vec<f64>> {
        if distance_metric != DistanceMetric::Euclidean {
            return None; // Kein Tree für nicht-euklidische Metriken
        }

        // Stelle sicher, dass der Baum existiert
        self.ensure_spatial_tree(active, observers, distance_metric);

        let tree_opt = if active {
            self.spatial_tree_active.borrow()
        } else {
            self.spatial_tree.borrow()
        };

        if let Some(SpatialTreeObserver::Euclidean(ref kdtree)) = *tree_opt {
            // Prüfe Dimension - kiddo unterstützt bis zu 100 Dimensionen
            if query_point.len() > 100 {
                return None; // Fallback zu Brute-Force für sehr hohe Dimensionen
            }

            // kiddo API: nearest_n benötigt &[f64; 100] - pad query_point
            let mut padded_query = [0.0; 100];
            for (i, &val) in query_point.iter().take(100).enumerate() {
                padded_query[i] = val;
            }

            // kiddo API: nearest_n gibt Vec<NearestNeighbor> zurück
            let neighbors = kdtree.nearest_n::<SquaredEuclidean>(&padded_query, k);
            // kiddo gibt quadrierte Distanz zurück, wir müssen die Wurzel ziehen
            Some(neighbors.iter().map(|n| n.distance.sqrt()).collect())
        } else {
            None
        }
    }

    /// Perform k-nearest neighbor search using tree
    /// Returns indices of k nearest neighbors
    pub(crate) fn search_k_nearest_indices_tree(
        &self,
        query_point: &[f64],
        k: usize,
        active: bool,
        distance_metric: DistanceMetric,
        observers: &[Arc<Observer>],
    ) -> Option<Vec<usize>> {
        if distance_metric != DistanceMetric::Euclidean {
            return None; // Kein Tree für nicht-euklidische Metriken
        }

        // Stelle sicher, dass der Baum existiert
        self.ensure_spatial_tree(active, observers, distance_metric);

        let tree_opt = if active {
            self.spatial_tree_active.borrow()
        } else {
            self.spatial_tree.borrow()
        };

        if let Some(SpatialTreeObserver::Euclidean(ref kdtree)) = *tree_opt {
            // Prüfe Dimension - kiddo unterstützt bis zu 100 Dimensionen
            if query_point.len() > 100 {
                return None; // Fallback zu Brute-Force für sehr hohe Dimensionen
            }

            // kiddo API: nearest_n benötigt &[f64; 100] - pad query_point
            let mut padded_query = [0.0; 100];
            for (i, &val) in query_point.iter().take(100).enumerate() {
                padded_query[i] = val;
            }

            // kiddo API: nearest_n gibt Vec<NearestNeighbor> zurück
            let neighbors = kdtree.nearest_n::<SquaredEuclidean>(&padded_query, k);
            // Der Baum enthält bereits nur aktive Observer, wenn active=true
            Some(neighbors.iter().map(|n| n.item).collect())
        } else {
            None
        }
    }
}
