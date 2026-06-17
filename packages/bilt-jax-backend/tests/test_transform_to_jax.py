"""Tests for transform_to_jax module."""

import ast
import typing

import jax
import jax.numpy as jnp
import pytest

from bilt.jax_backend.ast_to_jax import (
    bilt_asts_to_jax,
    transpile,
)
from bilt.jax_backend.supported_node_check import UnsupportedNodeError


def parse_to_ast(src: str) -> ast.FunctionDef:
    return typing.cast("ast.FunctionDef", ast.parse(src).body[0])


def test_transpile_simple_arithmetic() -> None:
    fn = transpile(parse_to_ast("def f(x, y):\n    return x + y"))
    assert float(fn(jnp.array(2.0), jnp.array(3.0))) == pytest.approx(5.0)


def test_transpile_jnp_call() -> None:
    fn = transpile(parse_to_ast("def f(x):\n    return jnp.sqrt(x)"))
    assert float(fn(jnp.array(9.0))) == pytest.approx(3.0)


def test_transpile_raises_on_unsupported_node() -> None:
    with pytest.raises(UnsupportedNodeError):
        transpile(parse_to_ast("def f(x):\n    if x > 0:\n        return x\n    return -x"))


def test_build_pipeline_two_functions() -> None:
    fdefs = [
        parse_to_ast("def f(x):\n    return x * 2"),
        parse_to_ast("def g(f, y):\n    return f + y"),
    ]
    pipeline = bilt_asts_to_jax(fdefs)
    # f(3) = 6, g(6, 1) = 7
    assert float(pipeline(x=jnp.array(3.0), y=jnp.array(1.0))) == pytest.approx(7.0)


def test_build_pipeline_external_inputs_attribute() -> None:
    fdefs = [
        parse_to_ast("def f(x, y):\n    return x + y"),
        parse_to_ast("def g(f, z):\n    return f * z"),
    ]
    pipeline = bilt_asts_to_jax(fdefs)
    assert pipeline.external_inputs == {"x", "y", "z"}
    assert pipeline(x=jnp.array(2.0), y=jnp.array(3.0), z=jnp.array(4.0)) == pytest.approx(20.0)


def test_pipeline_with_diamond_dependency() -> None:
    fdefs = [
        parse_to_ast("def f(x):\n    return x + 1"),
        parse_to_ast("def g(f, x):\n    return f * x"),
        parse_to_ast("def h(f, x):\n    return f - x"),
        parse_to_ast("def i(g, h):\n    return g + h"),
    ]
    pipeline = bilt_asts_to_jax(fdefs)
    assert pipeline.external_inputs == {"x"}
    assert float(pipeline(x=jnp.array(3.0))) == pytest.approx(13.0)


def test_build_pipeline_shared_external_input() -> None:
    fdefs = [
        parse_to_ast("def f(x):\n    return x + 1"),
        parse_to_ast("def g(f, x):\n    return f * x"),
    ]
    pipeline = bilt_asts_to_jax(fdefs)
    assert pipeline.external_inputs == {"x"}
    assert float(pipeline(x=jnp.array(3.0))) == pytest.approx(12.0)


def test_build_pipeline_raises_on_unsatisfied_arg() -> None:
    fdefs = [
        parse_to_ast("def f(x):\n    return x"),
        parse_to_ast("def g(h):\n    return h"),
        parse_to_ast("def h(g):\n    return g"),
    ]
    with pytest.raises(
        ValueError, match="Argument 'h' of 'g' is is not an expected input or a function that takes place before 'g'"
    ):
        bilt_asts_to_jax(fdefs)


def test_pipeline_rejects_unexpected_input() -> None:
    pipeline = bilt_asts_to_jax([parse_to_ast("def f(x):\n    return x")])
    with pytest.raises(ValueError, match=r"Unexpected inputs: {'z'}"):
        pipeline(x=jnp.array(1.0), z=jnp.array(2.0))


def test_transpile_list_literal_becomes_jnp_array() -> None:
    fn = transpile(parse_to_ast("def f():\n    return [1.0, 2.0, 3.0]"))
    result = fn()
    assert isinstance(result, jnp.ndarray)
    assert list(result) == pytest.approx([1.0, 2.0, 3.0])


def test_transpile_nested_list_both_converted() -> None:
    fn = transpile(parse_to_ast("def f():\n    return [[1.0, 2.0], [3.0, 4.0]]"))
    result = fn()
    assert isinstance(result, jnp.ndarray)
    assert result.shape == (2, 2)


def test_build_pipeline_with_list_operation() -> None:
    fdefs = [
        parse_to_ast("def f():\n    return [1.0,2.0,3.0,4.0,5.0]"),
        parse_to_ast("def g(f):\n    return f + 1"),
    ]
    pipeline = bilt_asts_to_jax(fdefs)
    assert float(pipeline() == pytest.approx([2.0, 3.0, 4.0, 5.0, 6.0]))


def test_grad_pipeline() -> None:
    """A proof of concept that we can differentate the output here, even though we have to do this hacky thing to turn kwargs into positional args for jax.grad."""
    fdefs = [
        parse_to_ast("def f(x):\n    return x ** 2"),
        parse_to_ast("def g(f, y):\n    return f + y"),
    ]
    pipeline = bilt_asts_to_jax(fdefs)
    grad_pipeline = jax.grad(lambda x, y: pipeline(x=x, y=y), argnums=0)
    assert float(grad_pipeline(3.0, 1.0)) == pytest.approx(6.0)
