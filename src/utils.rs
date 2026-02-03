use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::prelude::*;
use rand_distr::StandardNormal;

/// Distanzfunktion für SDO
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum DistanceMetric {
    Euclidean = 0,
    Manhattan = 1,
    Chebyshev = 2,
    Minkowski = 3,
}

/// Berechnet die euklidische Distanz (L²) zwischen zwei Punkten
pub fn compute_euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Berechnet die Manhattan-Distanz (L¹) zwischen zwei Punkten
pub fn compute_manhattan_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Berechnet die Chebyshev-Distanz (L∞) zwischen zwei Punkten
pub fn compute_chebyshev_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Berechnet die Minkowski-Distanz (Lᵖ) zwischen zwei Punkten
pub fn compute_minkowski_distance(a: &[f64], b: &[f64], p: f64) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs().powf(p))
        .sum::<f64>()
        .powf(1.0 / p)
}

/// Berechnet die Distanz zwischen zwei Punkten basierend auf der gewählten Metrik
pub fn compute_distance(
    a: &[f64],
    b: &[f64],
    metric: DistanceMetric,
    minkowski_p: Option<f64>,
) -> f64 {
    match metric {
        DistanceMetric::Euclidean => compute_euclidean_distance(a, b),
        DistanceMetric::Manhattan => compute_manhattan_distance(a, b),
        DistanceMetric::Chebyshev => compute_chebyshev_distance(a, b),
        DistanceMetric::Minkowski => {
            let p = minkowski_p.unwrap_or(3.0);
            compute_minkowski_distance(a, b, p)
        }
    }
}

/// Converts a numpy array with one row to a Vec<f64>
/// Returns: Vector of point coordinates
pub fn point_to_vec(point: PyReadonlyArray2<f64>) -> Vec<f64> {
    let point_slice = point.as_array();

    // Check if point has exactly one row
    if point_slice.nrows() != 1 {
        panic!("Point must be a 1D array or 2D array with exactly one row");
    }

    // Use row(0) for better performance
    point_slice.row(0).to_vec()
}

/// Converts a 2D numpy array to a Vec<Vec<f64>> (matrix)
/// Returns: Vector of rows, each row is a Vec<f64>
pub fn data_to_matrix(data: PyReadonlyArray2<f64>) -> (Vec<Vec<f64>>, usize) {
    let data_slice = data.as_array();
    let rows = data_slice.nrows();

    // More efficient allocation
    let mut matrix = Vec::with_capacity(rows);
    for i in 0..rows {
        let row: Vec<f64> = data_slice.row(i).to_vec();
        matrix.push(row);
    }

    (matrix, rows)
}

pub fn time_to_f64(
    time: Option<PyReadonlyArray1<f64>>,
    use_explicit_time: bool,
    fallback_time: usize,
) -> Result<f64, PyErr> {
    match (use_explicit_time, time) {
        // Case 1: Time required but not provided
        (true, None) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Zeit-Parameter erforderlich (Modell wurde mit Zeit initialisiert)",
        )),
        // Case 2: Time provided, validate and extract
        (_, Some(time_array)) => {
            let time_slice = time_array.as_array();
            if time_slice.len() != 1 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Zeit muss ein 1D-Array mit einem Wert sein",
                ));
            }
            Ok(time_slice[[0]])
        }
        // Case 3: Time not required and not provided → use fallback
        (false, None) => Ok(fallback_time as f64),
    }
}

/// Konvertiert optionales Zeit-Array zu Vec<f64> für Batch-Verarbeitung.
/// Wenn kein Zeit-Array gegeben ist, werden Zeiten automatisch generiert basierend auf use_explicit_time.
pub fn times_to_vec_batch(
    time: Option<PyReadonlyArray1<f64>>,
    rows: usize,
    use_explicit_time: bool,
    data_points_processed: usize,
) -> Result<Vec<f64>, PyErr> {
    let mut times_vec = Vec::with_capacity(rows);
    if let Some(time_array) = time {
        let time_slice = time_array.as_array();
        if time_slice.len() != rows {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "time muss gleiche Länge wie points haben: {} != {}",
                time_slice.len(),
                rows
            )));
        }
        for i in 0..rows {
            times_vec.push(time_slice[i]);
        }
    } else {
        // Auto-generiere Zeiten basierend auf use_explicit_time
        for i in 0..rows {
            let current_time = time_to_f64(None, use_explicit_time, data_points_processed + i)?;
            times_vec.push(current_time);
        }
    }
    Ok(times_vec)
}

/// Generates a random matrix with normally distributed values using rand_distr
pub fn sample_random_matrix_distr(dimension: usize, sample_size: usize) -> Vec<Vec<f64>> {
    let mut rng = thread_rng();
    let normal = StandardNormal;

    let mut random_matrix: Vec<Vec<f64>> = Vec::with_capacity(sample_size);

    for _ in 0..sample_size {
        let point: Vec<f64> = (0..dimension).map(|_| rng.sample(normal)).collect();
        random_matrix.push(point);
    }

    random_matrix
}

/// Generates a random matrix uniformly distributed in the unit hypercube [0, 1]^dimension.
/// Used for SDOstream dimension-only init (no domain knowledge).
pub fn sample_random_matrix_uniform_unit(dimension: usize, sample_size: usize) -> Vec<Vec<f64>> {
    let mut rng = thread_rng();
    let mut random_matrix: Vec<Vec<f64>> = Vec::with_capacity(sample_size);
    for _ in 0..sample_size {
        let point: Vec<f64> = (0..dimension).map(|_| rng.gen::<f64>()).collect();
        random_matrix.push(point);
    }
    random_matrix
}

pub fn compute_median(values: &Vec<f64>) -> f64 {
    let mut sorted_values = values.clone();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = sorted_values.len();
    if len % 2 == 0 {
        (sorted_values[len / 2 - 1] + sorted_values[len / 2]) / 2.0
    } else {
        sorted_values[len / 2]
    }
}

/// Konvertiert einen Score-Vektor für Python: ein Wert wenn rows == 1, sonst Liste.
/// Caller stellt sicher, dass scores.len() == rows.
pub fn scores_single_or_list_to_py(
    scores: &[f64],
    rows: usize,
    py: Python<'_>,
) -> PyResult<PyObject> {
    if rows == 1 {
        Ok(scores[0].into_py(py))
    } else {
        Ok(scores.to_vec().into_py(py))
    }
}

/// Konvertiert (label, score)-Ergebnisse für Python: ein Tupel wenn rows == 1, sonst Liste von Tupeln.
/// Caller stellt sicher, dass results.len() == rows.
pub fn label_score_results_to_py(
    results: &[(i32, f64)],
    rows: usize,
    py: Python<'_>,
) -> PyResult<PyObject> {
    use pyo3::types::{PyList, PyTuple};
    if rows == 1 {
        let (label, score) = results[0];
        let tuple = PyTuple::new_bound(py, [label.into_py(py), score.into_py(py)]);
        Ok(tuple.into_py(py))
    } else {
        let list: Vec<pyo3::Py<pyo3::PyAny>> = results
            .iter()
            .map(|(label, score)| {
                let tuple = PyTuple::new_bound(py, [label.into_py(py), score.into_py(py)]);
                tuple.into_py(py)
            })
            .collect();
        let py_list = PyList::new_bound(py, list);
        Ok(py_list.into_py(py))
    }
}
