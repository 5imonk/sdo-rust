use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;

use mtree::node::ObjectNode;
use mtree::Point;

/// Neighbor information: entweder mtree-Struktur (von knn_search) oder Index+Distanz (z. B. fit_point).
/// Lesbarkeit: überall als „NeighborInfo“ nutzbar; Index/Distanz über Helper.
#[derive(Clone)]
pub enum NeighborInfo {
    /// Von mtree knn_search: (Arc<ObjectNode>, Distanz)
    MTree(Arc<ObjectNode<Point, usize>>, f64),
    /// Nur Index + Distanz (z. B. wenn kein ObjectNode verfügbar)
    IndexDist(usize, f64),
}

impl fmt::Debug for NeighborInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NeighborInfo::MTree(node, d) => write!(f, "NeighborInfo::MTree({{index: {}}}, {})", node.value.1, d),
            NeighborInfo::IndexDist(idx, d) => write!(f, "NeighborInfo::IndexDist({}, {})", idx, d),
        }
    }
}

/// Index eines NeighborInfo-Eintrags.
#[inline(always)]
pub fn neighbor_index(n: &NeighborInfo) -> usize {
    match n {
        NeighborInfo::MTree(node, _) => node.value.1,
        NeighborInfo::IndexDist(idx, _) => *idx,
    }
}

/// Distanz eines NeighborInfo-Eintrags.
#[inline(always)]
pub fn neighbor_distance(n: &NeighborInfo) -> f64 {
    match n {
        NeighborInfo::MTree(_, d) => *d,
        NeighborInfo::IndexDist(_, d) => *d,
    }
}

// Helper struct for comparing floats in collections
#[derive(Debug, Clone, Copy)]
pub(crate) struct OrderedFloat(pub(crate) f64);

impl PartialEq for OrderedFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}
