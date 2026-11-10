use pyo3::prelude::*;

#[pyclass(from_py_object)]
#[derive(PartialEq, Debug, Clone)]
pub struct Atom {
    #[pyo3(get)]
    atom_type: AtomType,

    #[pyo3(get)]
    value: String,
}

#[pymethods]
impl Atom {
    #[new]
    fn new(atom_type: AtomType, value: String) -> Self {
        Atom { atom_type, value }
    }
}

#[pyclass(from_py_object)]
#[derive(PartialEq, Debug, Clone)]
pub struct AtomType {
    #[pyo3(get)]
    name: String,
}

#[pymethods]
impl AtomType {
    #[new]
    fn new(name: String) -> Self {
        AtomType { name }
    }
}
