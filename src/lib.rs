pub mod obs;
pub mod obset;
mod obset_clust;
mod obset_strcl;
mod obset_stream;
pub mod sdo_impl;
pub mod sdoclust_impl;
pub mod sdostrcl_impl;
pub mod sdostream_impl;

pub mod utils;

use pyo3::prelude::*;

use sdo_impl::SDO;
use sdoclust_impl::SDOclust;
use sdostrcl_impl::SDOstreamclust;
use sdostream_impl::SDOstream;

/// Python-Modul
#[pymodule]
fn sdo(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<SDO>()?;
    m.add_class::<SDOclust>()?;
    m.add_class::<SDOstream>()?;
    m.add_class::<SDOstreamclust>()?;
    Ok(())
}
