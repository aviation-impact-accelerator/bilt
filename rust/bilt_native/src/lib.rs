use pyo3::prelude::*;

#[pyfunction]
fn hello_from_rust() -> &'static str {
    "hello from Rust"
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_from_rust, m)?)?;
    Ok(())
}
