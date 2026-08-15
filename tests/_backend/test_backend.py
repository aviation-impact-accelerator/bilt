"""Tests for shared backend behaviour."""

import ast
import typing

import pytest

from bilt._backend import Backend, BackendCase, BackendLinearSystem
from bilt._backend._jax import JAXBackend
from bilt._backend._python import PythonBackend


def parse_to_ast(src: str) -> ast.FunctionDef:
    return typing.cast("ast.FunctionDef", ast.parse(src).body[0])


@pytest.fixture(params=[PythonBackend, JAXBackend])
def backend(request: pytest.FixtureRequest) -> Backend:
    backend_type = typing.cast("type[Backend]", request.param)
    return backend_type()


def make_simple_case(backend: Backend) -> BackendCase:
    return BackendCase(
        name="test_case",
        parameters=("a", "b", "d"),
        intermediates=[
            backend.transpile(parse_to_ast("def c(a: int, b: int) -> int:\n    return a + b")),
            backend.transpile(parse_to_ast("def e(c: int, d: int) -> int:\n    return c * d")),
        ],
        return_name="e",
    )


def make_diamond_case(backend: Backend) -> BackendCase:
    return BackendCase(
        name="test_case",
        parameters=("x",),
        intermediates=[
            backend.transpile(parse_to_ast("def f(x: int) -> int:\n    return x + 1")),
            backend.transpile(parse_to_ast("def g(f: int, x: int) -> int:\n    return f * x")),
            backend.transpile(parse_to_ast("def h(f: int, x: int) -> int:\n    return f - x")),
            backend.transpile(parse_to_ast("def i(g: int, h: int) -> int:\n    return g + h")),
        ],
        return_name="i",
    )


def test_transpile_simple_arithmetic(backend: Backend) -> None:
    fn = backend.transpile(parse_to_ast("def f(x, y):\n    return x + y"))

    assert float(fn.callable(2.0, 3.0)) == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("case_factory", "inputs", "expected"),
    (
        (
            make_simple_case,
            {"a": 3.0, "b": 1.0, "d": 2.0},
            8.0,
        ),
        (
            make_diamond_case,
            {"x": 3.0},
            13.0,
        ),
    ),
)
def test_build_pipeline_for_cases(
    backend: Backend,
    case_factory: typing.Callable[[Backend], BackendCase],
    inputs: dict[str, float],
    expected: float,
) -> None:
    pipeline = backend.chain_case(case_factory(backend))

    assert float(pipeline.callable(**inputs)) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("case_factory", "inputs"),
    (
        (
            make_simple_case,
            {"a": 3.0, "b": 1.0, "d": 2.0, "extra": 5.0},
        ),
        (
            make_diamond_case,
            {},
        ),
    ),
)
def test_pipeline_rejects_unexpected_input(
    backend: Backend,
    case_factory: typing.Callable[[Backend], BackendCase],
    inputs: dict[str, float],
) -> None:
    pipeline = backend.chain_case(case_factory(backend))

    with pytest.raises(TypeError, match=r"argument"):
        pipeline.callable(**inputs)


def test_can_accept_args(backend: Backend) -> None:
    pipeline = backend.chain_case(make_simple_case(backend))

    assert pipeline.callable(3.0, 1.0, 2.0) == pytest.approx(8.0)


def test_can_accept_kwargs(backend: Backend) -> None:
    pipeline = backend.chain_case(make_simple_case(backend))

    assert pipeline.callable(a=3.0, b=1.0, d=2.0) == pytest.approx(8.0)


def test_solve_linear_system(backend: Backend) -> None:
    system = BackendLinearSystem(
        parameters=(),
        unknowns=("x", "y"),
        coefficients=(
            (
                backend.transpile(parse_to_ast("def a() -> float:\n    return 2.0")),
                backend.transpile(parse_to_ast("def b() -> float:\n    return 1.0")),
            ),
            (
                backend.transpile(parse_to_ast("def c() -> float:\n    return 1.0")),
                backend.transpile(parse_to_ast("def d() -> float:\n    return -1.0")),
            ),
        ),
        rhs=(
            backend.transpile(parse_to_ast("def p() -> float:\n    return 5.0")),
            backend.transpile(parse_to_ast("def q() -> float:\n    return 1.0")),
        ),
    )

    result = backend.solve_linear(system, {})

    assert result["x"] == pytest.approx(2.0)
    assert result["y"] == pytest.approx(1.0)
