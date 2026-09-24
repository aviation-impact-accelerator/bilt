"""Tests for transform_to_jax module."""

import ast
import dataclasses
import typing

import pytest

from bilt._backend import Backend, BackendCase
from bilt._backend._python import PYTHON_BACKEND

if typing.TYPE_CHECKING:
    from bilt._backend._python._backend import PythonBackend


@typing.runtime_checkable
class _RuntimeBackend(Backend, typing.Protocol):
    """An isinstance-able view of the protocol, so it need not be runtime checkable in the source."""


def parse_to_ast(src: str) -> ast.FunctionDef:
    return typing.cast("ast.FunctionDef", ast.parse(src).body[0])


def test_python_backend_conforms_to_the_backend_protocol() -> None:
    assert isinstance(PYTHON_BACKEND, _RuntimeBackend)


@pytest.fixture
def python_backend() -> PythonBackend:
    return PYTHON_BACKEND


SIMPLE_CASE = BackendCase(
    name="test_case",
    parameters=("a", "b", "d"),
    intermediates=[
        PYTHON_BACKEND.transpile(parse_to_ast("def c(a: int, b: int) -> int:\n    return a + b")),
        PYTHON_BACKEND.transpile(parse_to_ast("def e(c: int, d: int) -> int:\n    return c * d")),
    ],
    return_name="e",
)


DIAMOND_CASE = BackendCase(
    name="test_case",
    parameters=("x",),
    intermediates=[
        PYTHON_BACKEND.transpile(parse_to_ast("def f(x: int) -> int:\n    return x + 1")),
        PYTHON_BACKEND.transpile(parse_to_ast("def g(f: int, x: int) -> int:\n        return f * x")),
        PYTHON_BACKEND.transpile(parse_to_ast("def h(f: int, x: int) -> int:\n    return f - x")),
        PYTHON_BACKEND.transpile(parse_to_ast("def i(g: int, h: int) -> int:\n        return g + h")),
    ],
    return_name="i",
)


def test_transpile_simple_arithmetic(python_backend: PythonBackend) -> None:
    fn = python_backend.transpile(parse_to_ast("def f(x, y):\n    return x + y"))
    assert float(fn.callable(2.0, 3.0)) == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("case", "inputs", "expected"),
    (
        (
            SIMPLE_CASE,
            {"a": 3.0, "b": 1.0, "d": 2.0},
            {"c": 4.0, "e": 8.0},
        ),
        (
            DIAMOND_CASE,
            {"x": 3.0},
            {"f": 4.0, "g": 12.0, "h": 1.0, "i": 13.0},
        ),
    ),
)
def test_build_pipeline_for_cases(
    case: BackendCase, inputs: dict[str, float], expected: dict[str, float], python_backend: PythonBackend
) -> None:
    pipeline = python_backend.chain_case(case)
    assert pipeline.evaluate(inputs) == pytest.approx(expected)


def test_evaluate_selects_only_the_parameters_the_case_declared(python_backend: PythonBackend) -> None:
    """Dispatch hands over every value the instance holds, so evaluate must pick out its own parameters."""
    pipeline = python_backend.chain_case(SIMPLE_CASE)

    outputs = pipeline.evaluate({"a": 3.0, "b": 1.0, "d": 2.0, "unrelated": 99.0})

    assert outputs == pytest.approx({"c": 4.0, "e": 8.0})


@pytest.mark.parametrize(
    ("case", "inputs"),
    (
        (
            SIMPLE_CASE,
            {"a": 3.0, "b": 1.0, "d": 2.0, "extra": 5.0},
        ),
        (
            DIAMOND_CASE,
            {},
        ),
    ),
)
def test_pipeline_rejects_unexpected_input(
    case: BackendCase, inputs: dict[str, float], python_backend: PythonBackend
) -> None:
    pipeline = python_backend.chain_case(case)
    with pytest.raises(TypeError, match=r"argument"):
        pipeline.callable(**inputs)


def test_a_chained_case_returns_its_values_positionally(python_backend: PythonBackend) -> None:
    """The one place the positional return shape is pinned; everything else goes through `evaluate`."""
    pipeline = python_backend.chain_case(SIMPLE_CASE)

    assert pipeline.outputs == ("c", "e")
    assert pipeline.callable(a=3.0, b=1.0, d=2.0) == pytest.approx((4.0, 8.0))


def test_a_case_that_does_not_expose_intermediates_returns_one_value(python_backend: PythonBackend) -> None:
    pipeline = python_backend.chain_case(dataclasses.replace(SIMPLE_CASE, expose_intermediates=False))

    assert pipeline.outputs is None
    assert pipeline.callable(a=3.0, b=1.0, d=2.0) == pytest.approx(8.0)


def test_a_chained_case_can_be_an_intermediate_of_another_case(python_backend: PythonBackend) -> None:
    inner = python_backend.chain_case(dataclasses.replace(SIMPLE_CASE, expose_intermediates=False))
    assert inner.name == "e"

    outer = BackendCase(
        name="outer_case",
        parameters=(*SIMPLE_CASE.parameters, "k"),
        intermediates=[
            inner,
            python_backend.transpile(parse_to_ast("def f(e: int, k: int) -> int:\n    return e + k")),
        ],
        return_name="f",
    )
    pipeline = python_backend.chain_case(outer)

    assert pipeline.evaluate({"a": 3.0, "b": 1.0, "d": 2.0, "k": 5.0}) == pytest.approx({"e": 8.0, "f": 13.0})
