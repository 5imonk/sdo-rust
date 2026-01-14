mod observer;
mod observer_clust;
mod observer_stream;
mod observer_streamclust;
mod sdo_impl;
mod sdoclust_impl;
mod sdostream_impl;
mod sdostreamclust_impl;
mod utils;

use pyo3::prelude::*;

use sdo_impl::{SDOParams, SDO};
use sdoclust_impl::{SDOclust, SDOclustParams};
use sdostream_impl::{SDOstream, SDOstreamParams};
use sdostreamclust_impl::{SDOstreamclust, SDOstreamclustParams};

/// Python-Modul
#[pymodule]
fn sdo(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<SDO>()?;
    m.add_class::<SDOParams>()?;
    m.add_class::<SDOclust>()?;
    m.add_class::<SDOclustParams>()?;
    m.add_class::<SDOstream>()?;
    m.add_class::<SDOstreamParams>()?;
    m.add_class::<SDOstreamclust>()?;
    m.add_class::<SDOstreamclustParams>()?;
    Ok(())
}
