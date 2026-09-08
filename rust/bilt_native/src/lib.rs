use pyo3::prelude::*;

#[path = "symbolic/atom.rs"]
mod atom;
#[path = "symbolic/tree.rs"]
mod tree;

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<atom::AtomType>()?;
    m.add_class::<atom::Atom>()?;
    m.add_class::<tree::Tree>()?;
    Ok(())
}
