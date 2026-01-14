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

use sdo_impl::SDO;
use sdoclust_impl::SDOclust;
use sdostream_impl::SDOstream;
use sdostreamclust_impl::SDOstreamclust;

/// Python-Modul
#[pymodule]
fn sdo(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<SDO>()?;
    m.add_class::<SDOclust>()?;
    m.add_class::<SDOstream>()?;
    m.add_class::<SDOstreamclust>()?;
    Ok(())
}
