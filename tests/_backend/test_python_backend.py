"""Tests for the Python backend."""

import ast
import typing

import numpy as np
import pytest

from bilt._backend import Backend, BackendCase, BackendLinearSystem
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
            8.0,
        ),
        (
            DIAMOND_CASE,
            {"x": 3.0},
            13.0,
        ),
    ),
)
def test_build_pipeline_for_cases(
    case: BackendCase,
    inputs: dict[str, float],
    expected: float,
    python_backend: PythonBackend,
) -> None:
    pipeline = python_backend.chain_case(case)
    assert float(pipeline.callable(**inputs)) == pytest.approx(expected)


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
    case: BackendCase,
    inputs: dict[str, float],
    python_backend: PythonBackend,
) -> None:
    pipeline = python_backend.chain_case(case)
    with pytest.raises(TypeError, match=r"argument"):
        pipeline.callable(**inputs)


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
    case: BackendCase,
    inputs: dict[str, float] | tuple[float, ...],
    python_backend: PythonBackend,
) -> None:
    pipeline = python_backend.chain_case(case)
    if isinstance(inputs, dict):
        assert pipeline.callable(**inputs) == pytest.approx(8.0)
    else:
        assert pipeline.callable(*inputs) == pytest.approx(8.0)


def test_solve_linear_system(python_backend: PythonBackend) -> None:
    system = BackendLinearSystem(
        parameters=(),
        unknowns=("x", "y"),
        coefficients=(
            (
                python_backend.transpile(parse_to_ast("def a() -> float:\n    return 2.0")),
                python_backend.transpile(parse_to_ast("def b() -> float:\n    return 1.0")),
            ),
            (
                python_backend.transpile(parse_to_ast("def c() -> float:\n    return 1.0")),
                python_backend.transpile(parse_to_ast("def d() -> float:\n    return -1.0")),
            ),
        ),
        rhs=(
            python_backend.transpile(parse_to_ast("def p() -> float:\n    return 5.0")),
            python_backend.transpile(parse_to_ast("def q() -> float:\n    return 1.0")),
        ),
    )

    result = python_backend.solve_linear(system, {})

    x = typing.cast("typing.SupportsFloat", result["x"])
    y = typing.cast("typing.SupportsFloat", result["y"])

    assert float(x) == pytest.approx(2.0)
    assert float(y) == pytest.approx(1.0)


def test_solve_linear_system_broadcasts_scalar_coefficients(
    python_backend: PythonBackend,
) -> None:
    system = BackendLinearSystem(
        parameters=("p",),
        unknowns=("x", "y"),
        coefficients=(
            (
                python_backend.transpile(parse_to_ast("def a() -> float:\n    return 2.0")),
                python_backend.transpile(parse_to_ast("def b() -> float:\n    return 1.0")),
            ),
            (
                python_backend.transpile(parse_to_ast("def c() -> float:\n    return 1.0")),
                python_backend.transpile(parse_to_ast("def d() -> float:\n    return -1.0")),
            ),
        ),
        rhs=(
            python_backend.transpile(parse_to_ast("def rhs_p(p):\n    return p")),
            python_backend.transpile(parse_to_ast("def rhs_q() -> float:\n    return 1.0")),
        ),
    )

    result = python_backend.solve_linear(
        system,
        {"p": np.array([5.0, 8.0])},
    )

    x = typing.cast("np.ndarray", result["x"])
    y = typing.cast("np.ndarray", result["y"])

    assert x == pytest.approx([2.0, 3.0])
    assert y == pytest.approx([1.0, 2.0])


def test_chain_case_can_include_linear_system(
    python_backend: PythonBackend,
) -> None:
    linear_system = BackendLinearSystem(
        parameters=("p", "q"),
        unknowns=("x", "y"),
        coefficients=(
            (
                python_backend.transpile(parse_to_ast("def a() -> float:\n    return 2.0")),
                python_backend.transpile(parse_to_ast("def b() -> float:\n    return 1.0")),
            ),
            (
                python_backend.transpile(parse_to_ast("def c() -> float:\n    return 1.0")),
                python_backend.transpile(parse_to_ast("def d() -> float:\n    return -1.0")),
            ),
        ),
        rhs=(
            python_backend.transpile(parse_to_ast("def rhs_p(p):\n    return p")),
            python_backend.transpile(parse_to_ast("def rhs_q(q):\n    return q")),
        ),
    )

    total = python_backend.transpile(parse_to_ast("def total(x: float, y: float) -> float:\n    return x + y"))

    case = BackendCase(
        name="linear_case",
        parameters=("p", "q"),
        intermediates=[linear_system, total],
        return_name="total",
    )

    pipeline = python_backend.chain_case(case)

    assert pipeline.callable(p=5.0, q=1.0) == pytest.approx(3.0)
