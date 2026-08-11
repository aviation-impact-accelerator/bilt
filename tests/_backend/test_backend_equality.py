"""Tests for Backend value semantics: equality and hashing."""

import ast

from bilt._backend._jax._backend import JAXBackend
from bilt._backend._python._backend import PythonBackend


def test_python_backend_equality_and_hash() -> None:
    backend1 = PythonBackend()
    backend2 = PythonBackend()
    assert backend1 == backend2
    assert hash(backend1) == hash(backend2)


def test_python_backend_not_equal_to_jax_backend() -> None:
    python_backend = PythonBackend()
    jax_backend = JAXBackend()
    assert python_backend != jax_backend


def test_jax_backend_equality_and_hash() -> None:
    backend1 = JAXBackend()
    backend2 = JAXBackend()
    assert backend1 == backend2
    assert hash(backend1) == hash(backend2)


def test_jax_backend_with_different_ast_transformer_not_equal() -> None:
    class CustomTransformer1(ast.NodeTransformer):
        """A custom AST transformer for testing."""

    class CustomTransformer2(ast.NodeTransformer):
        """Another custom AST transformer for testing."""

    backend1 = JAXBackend(ast_transformer=CustomTransformer1())
    backend2 = JAXBackend(ast_transformer=CustomTransformer2())
    assert backend1 != backend2


def test_python_backend_not_equal_to_object() -> None:
    backend = PythonBackend()
    assert (backend == object()) is False
    assert (backend != object()) is True


def test_equal_backends_collapse_to_single_dict_key() -> None:
    backend1 = PythonBackend()
    backend2 = PythonBackend()

    value_1 = 1
    value_2 = 2
    d = {backend1: value_1, backend2: value_2}
    assert len(d) == 1
    assert d[backend1] == value_2
