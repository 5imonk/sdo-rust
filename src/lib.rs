mod sdo_impl;
mod sdoclust_impl;
mod utils;

use pyo3::prelude::*;

use sdo_impl::{SDOParams, SDO};
use sdoclust_impl::{SDOclust, SDOclustParams};

/// Python-Modul
#[pymodule]
fn sdo(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<SDO>()?;
    m.add_class::<SDOParams>()?;
    m.add_class::<SDOclust>()?;
    m.add_class::<SDOclustParams>()?;
    Ok(())
}
