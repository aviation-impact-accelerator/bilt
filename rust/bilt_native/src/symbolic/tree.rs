use pyo3::prelude::*;

use crate::atom::Atom;

#[pyclass(from_py_object)]
#[derive(Debug, PartialEq, Clone)]
pub struct Tree {
    #[pyo3(get)]
    head: Atom,
    #[pyo3(get)]
    args: Vec<Tree>,
}

impl Tree {
    pub fn into_children(self) -> Vec<Tree> {
        self.args
    }
}

#[pymethods]
impl Tree {
    #[new]
    pub fn new(head: Atom, args: Vec<Tree>) -> Self {
        Tree { head, args }
    }

    pub fn value(&self) -> Atom {
        self.head.clone()
    }

    pub fn children(&self) -> Vec<Tree> {
        self.args.clone()
    }

    pub fn is_leaf(&self) -> bool {
        self.args.is_empty()
    }

    pub fn is_binary(&self) -> bool {
        self.args.len() == 2
    }

    pub fn left(&self) -> Option<Tree> {
        if !self.is_binary() {
            None
        } else {
            Some(self.args[0].clone())
        }
    }

    pub fn right(&self) -> Option<Tree> {
        if !self.is_binary() {
            None
        } else {
            Some(self.args[1].clone())
        }
    }
}
