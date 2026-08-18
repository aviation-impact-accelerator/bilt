"""Tests for transform_to_jax module."""

import ast
import typing

import jax.numpy as jnp
import pytest

from bilt._backend import Backend, BackendCase
from bilt._backend._jax import JAX_BACKEND, UnsupportedNodeError

if typing.TYPE_CHECKING:
    from bilt._backend._jax._backend import JAXBackend


@typing.runtime_checkable
class _RuntimeBackend(Backend, typing.Protocol):
    """An isinstance-able view of the protocol, so it need not be runtime checkable in the source."""


def parse_to_ast(src: str) -> ast.FunctionDef:
    return typing.cast("ast.FunctionDef", ast.parse(src).body[0])


def test_jax_backend_conforms_to_the_backend_protocol() -> None:
    assert isinstance(JAX_BACKEND, _RuntimeBackend)


@pytest.fixture
def jax_backend() -> JAXBackend:
    return JAX_BACKEND


SIMPLE_CASE = BackendCase(
    name="test_case",
    parameters=("a", "b", "d"),
    intermediates=[
        JAX_BACKEND.transpile(parse_to_ast("def c(a: int, b: int) -> int:\n    return a + b")),
        JAX_BACKEND.transpile(parse_to_ast("def e(c: int, d: int) -> int:\n    return c * d")),
    ],
    return_name="e",
)


DIAMOND_CASE = BackendCase(
    name="test_case",
    parameters=("x",),
    intermediates=[
        JAX_BACKEND.transpile(parse_to_ast("def f(x: int) -> int:\n    return x + 1")),
        JAX_BACKEND.transpile(parse_to_ast("def g(f: int, x: int) -> int:\n        return f * x")),
        JAX_BACKEND.transpile(parse_to_ast("def h(f: int, x: int) -> int:\n    return f - x")),
        JAX_BACKEND.transpile(parse_to_ast("def i(g: int, h: int) -> int:\n        return g + h")),
    ],
    return_name="i",
)

LIST_CASE = BackendCase(
    name="test_case",
    parameters=(),
    intermediates=[
        JAX_BACKEND.transpile(parse_to_ast("def l1() -> list[float]:\n    return [1.0, 2.0, 3.0, 4.0, 5.0]")),
        JAX_BACKEND.transpile(parse_to_ast("def l2(l1: float) -> float:\n    return l1 + 1.0")),
    ],
    return_name="l2",
)


def test_transpile_simple_arithmetic(jax_backend: JAXBackend) -> None:
    fn = jax_backend.transpile(parse_to_ast("def f(x, y):\n    return x + y"))
    assert float(fn.callable(jnp.array(2.0), jnp.array(3.0))) == pytest.approx(5.0)


def test_transpile_jnp_call(jax_backend: JAXBackend) -> None:
    fn = jax_backend.transpile(parse_to_ast("def f(x):\n    return jnp.sqrt(x)"))
    assert float(fn.callable(jnp.array(9.0))) == pytest.approx(3.0)


def test_transpile_raises_on_unsupported_node(jax_backend: JAXBackend) -> None:
    with pytest.raises(UnsupportedNodeError):
        jax_backend.transpile(parse_to_ast("def f(x):\n    if x > 0:\n        return x\n    return -x"))


def test_transpile_list_literal_becomes_jnp_array(jax_backend: JAXBackend) -> None:
    fn = jax_backend.transpile(parse_to_ast("def f():\n    return [1.0, 2.0, 3.0]"))
    result = fn.callable()
    assert isinstance(result, jnp.ndarray)
    assert list(result) == pytest.approx([1.0, 2.0, 3.0])


def test_transpile_nested_list_both_converted(jax_backend: JAXBackend) -> None:
    fn = jax_backend.transpile(parse_to_ast("def f():\n    return [[1.0, 2.0], [3.0, 4.0]]"))
    result = fn.callable()
    assert isinstance(result, jnp.ndarray)
    assert result.shape == (2, 2)


@pytest.mark.parametrize(
    ("case", "inputs", "expected"),
    (
        (
            SIMPLE_CASE,
            (3.0, 1.0, 2.0),
            8.0,
        ),
        (
            DIAMOND_CASE,
            (3.0,),
            13.0,
        ),
    ),
)
def test_build_pipeline_for_cases(
    case: BackendCase, inputs: tuple[float, ...], expected: float, jax_backend: JAXBackend
) -> None:
    pipeline = jax_backend.chain_case(case)
    assert float(pipeline.callable(*inputs)) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("case", "inputs"),
    (
        (
            SIMPLE_CASE,
            (3.0, 1.0, 2.0, 2.0),
        ),
        (
            DIAMOND_CASE,
            (),
        ),
    ),
)
def test_pipeline_rejects_unexpected_input(
    case: BackendCase, inputs: tuple[float, ...], jax_backend: JAXBackend
) -> None:
    pipeline = jax_backend.chain_case(case)
    with pytest.raises(TypeError, match=r"argument"):
        pipeline.callable(*inputs)


def test_build_pipeline_with_list_operation(jax_backend: JAXBackend) -> None:
    pipeline = jax_backend.chain_case(LIST_CASE)
    assert float(pipeline.callable() == pytest.approx([2.0, 3.0, 4.0, 5.0, 6.0]))


@pytest.mark.parametrize(
    ("case", "inputs"),
    (
        (
            SIMPLE_CASE,
            {"a": 3.0, "b": 1.0, "d": 2.0},
        ),
        (
            SIMPLE_CASE,
            (3.0, 1.0, 2.0),
        ),
    ),
)
def test_can_accept_args_or_kwargs(
    case: BackendCase, inputs: dict[str, float] | tuple[float, ...], jax_backend: JAXBackend
) -> None:
    pipeline = jax_backend.chain_case(case)
    if isinstance(inputs, dict):
        assert pipeline.callable(**inputs) == pytest.approx(8.0)
    else:
        assert pipeline.callable(*inputs) == pytest.approx(8.0)
