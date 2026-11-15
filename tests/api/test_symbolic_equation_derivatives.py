# ruff: noqa: SIM108
"""Expression-level derivatives can be used as regular equation constraints."""

import typing

import pytest

from bilt import (
    JAX_BACKEND,
    PYTHON_BACKEND,
    Implementation,
    Interface,
    equation,
    field,
    grad,
    hessian,
    jacobian,
    system,
)
from bilt.core._to_ast import tree_to_functiondef
from bilt.symbolic._expressions import Expr, symbol_expr

if typing.TYPE_CHECKING:
    from bilt._backend._base_backend import Backend


@system
class Polynomial(metaclass=Interface):
    x: float = field()
    y: float = field()
    slope: float = field()
    other_slope: float = field()
    mixed: float = field()
    diagonal: float = field()
    entry: float = field()


@equation
def scalar_gradient(x: float, y: float, slope: float) -> bool:
    expression = x**3 + x * y + y**2
    return slope == grad(expression, wrt=x)


@equation
def tuple_gradient(x: float, y: float, other_slope: float) -> bool:
    expression = x**3 + x * y + y**2
    return other_slope == grad(expression, wrt=(x, y))[1]


@equation
def mixed_second_derivative(x: float, y: float, mixed: float) -> bool:
    expression = x**3 + x * y + y**2
    return mixed == grad(grad(expression, wrt=x), wrt=y)


@equation
def diagonal_second_derivative(x: float, y: float, diagonal: float) -> bool:
    expression = x**3 + x * y + y**2
    return diagonal == grad(grad(expression, wrt=x), wrt=x)


@equation
def cross_sensitivity(x: float, y: float, entry: float) -> bool:
    return entry == grad(x * y, wrt=x)


POLYNOMIAL = Implementation(
    Polynomial,
    [scalar_gradient, tuple_gradient, mixed_second_derivative, diagonal_second_derivative, cross_sensitivity],
)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_symbolic_derivatives_inside_equations(backend: Backend) -> None:
    with backend, POLYNOMIAL:
        model = Polynomial(x=2.0, y=3.0)
        assert model.slope == pytest.approx(15.0)
        assert model.other_slope == pytest.approx(8.0)
        assert model.mixed == pytest.approx(1.0)
        assert model.diagonal == pytest.approx(12.0)
        assert model.entry == pytest.approx(3.0)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_full_symbolic_matrices(backend: Backend) -> None:
    """Matrix helpers are for whole derivative matrices, not single entries."""
    x = symbol_expr("x")
    y = symbol_expr("y")
    outputs = (x**2 + y, x * y)
    matrix = jacobian(outputs, wrt=(x, y))
    curvature = hessian(x**3 + x * y + y**2, wrt=(x, y))

    expected_size = len((x, y))
    assert len(matrix) == len(curvature) == expected_size
    assert all(len(row) == expected_size for row in (*matrix, *curvature))
    for name, values, expected in (
        ("jacobian", matrix, ((4.0, 1.0), (3.0, 2.0))),
        ("hessian", curvature, ((12.0, 1.0), (1.0, 2.0))),
    ):
        for i, row in enumerate(values):
            for j, entry in enumerate(row):
                assert isinstance(entry, Expr)
                compiled = backend.transpile(
                    tree_to_functiondef(entry.tree, name=f"{name}_{i}_{j}", parameter_order=("x", "y"))
                )
                assert compiled.evaluate({"x": 2.0, "y": 3.0}) == pytest.approx(expected[i][j])


@system
class ThirdOrder(metaclass=Interface):
    x: float = field()
    result: float = field()


@equation
def third_derivative(x: float, result: float) -> bool:
    expression = x**3
    return result == grad(grad(grad(expression, wrt=x), wrt=x), wrt=x)


THIRD_ORDER = Implementation(ThirdOrder, [third_derivative])


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_nested_grad_supports_higher_order_partials(backend: Backend) -> None:
    with backend, THIRD_ORDER:
        assert ThirdOrder(x=5.0).result == pytest.approx(6.0)


@system
class Piecewise(metaclass=Interface):
    x: float = field()
    slope: float = field()


@equation
def piecewise_gradient(x: float, slope: float) -> bool:
    if x > 0:
        selected = x**2
    else:
        selected = -x
    return slope == grad(selected, wrt=x)


PIECEWISE = Implementation(Piecewise, [piecewise_gradient])


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
@pytest.mark.parametrize(("x", "expected"), ((3.0, 6.0), (-3.0, -1.0)))
def test_symbolic_gradient_of_conditional_branch(backend: Backend, x: float, expected: float) -> None:
    with backend, PIECEWISE:
        model = Piecewise(x=x)
        assert model.slope == pytest.approx(expected)


def test_rejects_non_symbolic_wrt() -> None:
    x = symbol_expr("x")
    with pytest.raises(TypeError, match="symbolic equation parameter"):
        grad(x**2, wrt=x + 1)
    with pytest.raises(TypeError, match="symbolic equation parameter"):
        grad(3.0, wrt=2.0)


def test_rejects_empty_or_duplicate_wrt() -> None:
    x = symbol_expr("x")
    with pytest.raises(ValueError, match="at least one"):
        grad(x**2, wrt=())
    with pytest.raises(ValueError, match="duplicate"):
        hessian(x**2, wrt=(x, x))


def test_rejects_empty_jacobian_outputs() -> None:
    x = symbol_expr("x")
    with pytest.raises(ValueError, match="non-empty"):
        jacobian((), wrt=(x,))


def test_local_expression_does_not_infer_other_equations() -> None:
    """Independent symbolic inputs remain independent; no implicit system derivative is invented."""
    x = symbol_expr("x")
    y = symbol_expr("y")
    result = grad(y, wrt=x)
    assert result.tree.head.value == 0.0
