"""Tests for Backend value semantics: equality and hashing."""

import ast
from typing import TYPE_CHECKING

import pytest

from bilt._backend._jax._backend import JAXBackend
from bilt._backend._python._backend import PythonBackend

if TYPE_CHECKING:
    from bilt._backend._base_backend import Backend


@pytest.mark.parametrize(
    ("backend_type"),
    (
        PythonBackend,
        JAXBackend,
    ),
)
def test_equivalent_backends_are_equal_and_hash_alike(backend_type: type[Backend]) -> None:
    assert backend_type() == backend_type()
    assert hash(backend_type()) == hash(backend_type())


def test_python_backend_not_equal_to_jax_backend() -> None:
    assert PythonBackend() != JAXBackend()


def test_jax_backend_with_different_ast_transformer_not_equal() -> None:
    class CustomTransformer1(ast.NodeTransformer):
        """A custom AST transformer for testing."""

    class CustomTransformer2(ast.NodeTransformer):
        """Another custom AST transformer for testing."""

    assert JAXBackend(ast_transformer=CustomTransformer1()) != JAXBackend(ast_transformer=CustomTransformer2())


def test_python_backend_not_equal_to_object() -> None:
    assert (PythonBackend() == object()) is False
