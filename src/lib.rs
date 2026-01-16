mod obs;
mod obset;
mod obset_clust;
mod obset_strcl;
mod obset_stream;
mod obset_tree;
mod sdo_impl;
mod sdoclust_impl;
mod sdostrcl_impl;
mod sdostream_impl;
mod utils;

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
