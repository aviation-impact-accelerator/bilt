use pyo3::prelude::*;

#[pyfunction]
fn hello_world() -> PyResult<String> {
    Ok("hello-world".to_string())
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_world, m)?)?;
    Ok(())
}
