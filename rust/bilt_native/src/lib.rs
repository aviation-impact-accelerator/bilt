use pyo3::prelude::*;

mod structural;

#[pyfunction]
fn hello_from_rust() -> &'static str {
    "hello from Rust"
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_from_rust, m)?)?;
    m.add_class::<structural::NativeStructuralModel>()?;
    Ok(())
}
