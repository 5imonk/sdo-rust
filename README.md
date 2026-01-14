# SDO - Sparse Data Observers

High-performance implementation of the Sparse Data Observers (SDO) algorithm and SDOclust clustering extension in Rust with Python bindings.

## Features

- **SDO**: Outlier detection algorithm using sparse representative models
- **SDOclust**: Clustering extension using Connected Components Clustering (CCC)
- **High Performance**: Rust implementation with Python bindings via PyO3
- **Scikit-learn Compatible**: Easy integration with existing ML pipelines
- **Non-convex Clusters**: SDOclust can identify non-convex cluster shapes (e.g., spirals)

## Installation

### Prerequisites

- Rust (latest stable)
- Python 3.8+
- pip

### Build from Source

```bash
# Clone the repository
git clone https://github.com/5imonk/sdo-rust.git
cd sdo-rust

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install maturin
pip install maturin

# Build and install
maturin develop
```

## Quick Start

### SDO - Outlier Detection

```python
import numpy as np
from sdo import SDO

# Create SDO instance with parameters
sdo = SDO(k=5, x=3, rho=0.2)

# Training data
data = np.array([
    [1.0, 2.0], [2.0, 3.0], [3.0, 4.0],
    [10.0, 11.0],  # Outlier
], dtype=np.float64)

# Train model
sdo.learn(data)

# Predict outlier score
point = np.array([[10.0, 11.0]], dtype=np.float64)
score = sdo.predict(point)
print(f"Outlier Score: {score}")
```

### SDOclust - Clustering

```python
import numpy as np
from sdo import SDOclust

# Create SDOclust instance with parameters
sdoclust = SDOclust(k=20, x=5, rho=0.2, chi=4, zeta=0.5, min_cluster_size=2)

# Training data with multiple clusters
np.random.seed(42)
cluster1 = np.random.randn(30, 2) * 0.5 + np.array([2.0, 2.0])
cluster2 = np.random.randn(30, 2) * 0.5 + np.array([8.0, 8.0])
data = np.vstack([cluster1, cluster2]).astype(np.float64)

# Train model
sdoclust.learn(data)

# Predict cluster label
point = np.array([[2.0, 2.0]], dtype=np.float64)
label = sdoclust.predict(point)
print(f"Cluster Label: {label}")
print(f"Number of Clusters: {sdoclust.n_clusters()}")
```

### Scikit-learn Compatible API

```python
from sdo_sklearn import SDOOutlierDetector
from sdoclust_sklearn import SDOclustClusterer

# Outlier Detection
detector = SDOOutlierDetector(k=10, x=5, rho=0.2)
scores = detector.fit_predict(X)

# Clustering
clusterer = SDOclustClusterer(k=20, x=5, chi=4, zeta=0.5)
labels = clusterer.fit_predict(X)
```

## Documentation

For detailed documentation, see [README_PYTHON.md](README_PYTHON.md).

## Examples

- `example.py` - Simple SDO example
- `test_sdo.py` - Comprehensive SDO tests
- `test_sdoclust.py` - Comprehensive SDOclust tests
- `sdo_sklearn.py` - Scikit-learn API example for SDO
- `sdoclust_sklearn.py` - Scikit-learn API example for SDOclust

## Algorithm Details

### SDO (Sparse Data Observers)

SDO is an outlier detection algorithm that:
1. Samples k observers from the data
2. Counts observations (neighbors) for each observer
3. Keeps the top (1-ρ) fraction as active observers
4. Computes outlier scores as median distance to x nearest active observers

### SDOclust

SDOclust extends SDO with clustering capabilities:
1. Builds a graph where observers are nodes
2. Uses local cutoff thresholds (h_ω = d(ω, ω←χ))
3. Applies a mixture model: h'_ω = ζ·h_ω + (1-ζ)·h
4. Connects observers if distance < both thresholds
5. Performs Connected Components Clustering
6. Removes small clusters (< min_cluster_size)

## Parameters

### SDO Parameters
- `k`: Number of sampled observers (default: 10)
- `x`: Number of nearest neighbors (default: 5)
- `rho`: Fraction of observers to remove, 0.0-1.0 (default: 0.2)

### SDOclust Parameters
- `k`: Number of sampled observers (default: 10)
- `x`: Number of nearest neighbors (default: 5)
- `rho`: Fraction of observers to remove (default: 0.2)
- `chi` (χ): Number of nearest observers for local thresholds (default: 4)
- `zeta` (ζ): Mixing parameter for global/local thresholds, 0.0-1.0 (default: 0.5)
- `min_cluster_size` (e): Minimum cluster size (default: 2)

## Requirements

- Rust 1.70+
- Python 3.8+
- numpy
- maturin (for building)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## References

- SDO Algorithm: [Original Paper]
- SDOclust: Iglesias et al. [IZHZ23]
- Connected Components Clustering: [HS00]

## Author

Simon

## Acknowledgments

- PyO3 for Python-Rust bindings
- numpy for efficient array operations
- kdtree crate for efficient nearest neighbor search

