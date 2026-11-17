# ruff: noqa: SIM108, PLR2004, PLR5501
"""Differentiate supported conditional equations through the public model API."""

import typing

import jax
import jax.numpy as jnp
import pytest

from bilt import JAX_BACKEND, PYTHON_BACKEND, Implementation, Interface, equation, field, system

if typing.TYPE_CHECKING:
    from bilt._backend._base_backend import Backend


@system
class Piecewise(metaclass=Interface):
    x: float = field()
    value: float = field()
    bias: float = field()
    total: float = field()


@equation
def piecewise_value(x: float, value: float) -> bool:
    if x > 0:
        selected = x**2
    else:
        selected = -x
    return value == selected


@equation
def scaled_total(value: float, bias: float, total: float) -> bool:
    return total == 3 * value + bias


PIECEWISE = Implementation(Piecewise, [piecewise_value, scaled_total])


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
@pytest.mark.parametrize(
    ("x", "expected_value", "expected_derivative"),
    ((3.0, 9.0, 6.0), (-3.0, 3.0, -1.0)),
)
def test_if_else_differentiates_active_branch(
    backend: Backend, x: float, expected_value: float, expected_derivative: float
) -> None:
    with backend, PIECEWISE:
        model = Piecewise(x=x)
        assert model.value == pytest.approx(expected_value)
        assert model.d.value(wrt=Piecewise.x) == pytest.approx(expected_derivative)  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
@pytest.mark.parametrize(("x", "expected_derivative"), ((2.0, 12.0), (-2.0, -3.0)))
def test_conditional_tangent_flows_through_next_equation(
    backend: Backend, x: float, expected_derivative: float
) -> None:
    with backend, PIECEWISE:
        model = Piecewise(x=x, bias=10.0)
        assert model.d.total(wrt=Piecewise.x) == pytest.approx(expected_derivative)  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]
        assert model.d.total(wrt=Piecewise.bias) == pytest.approx(1.0)  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]


@system
class Selected(metaclass=Interface):
    x: float = field()
    threshold: float = field()
    scale: float = field()
    offset: float = field()
    result: float = field()


@equation
def selected_value(x: float, threshold: float, scale: float, offset: float, result: float) -> bool:
    if x > threshold:
        selected = scale * x**2
    else:
        selected = offset - x
    return result == selected


SELECTED = Implementation(Selected, [selected_value])


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
@pytest.mark.parametrize(
    ("x", "dx", "dscale", "doffset"),
    ((3.0, 12.0, 9.0, 0.0), (-3.0, -1.0, 0.0, 1.0)),
)
def test_conditional_predicate_and_inactive_branch(
    backend: Backend, x: float, dx: float, dscale: float, doffset: float
) -> None:
    with backend, SELECTED:
        model = Selected(x=x, threshold=0.0, scale=2.0, offset=7.0)
        assert model.d.result(wrt=Selected.x) == pytest.approx(dx)  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]
        assert model.d.result(wrt=Selected.scale) == pytest.approx(dscale)  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]
        assert model.d.result(wrt=Selected.offset) == pytest.approx(doffset)  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]
        # A switch position has no local first-order effect away from the threshold.
        assert model.d.result(wrt=Selected.threshold) == pytest.approx(0.0)  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]


@system
class MultiBranch(metaclass=Interface):
    x: float = field()
    result: float = field()


@equation
def elif_value(x: float, result: float) -> bool:
    if x > 10:
        selected = x**2
    elif x > 0:
        selected = 2 * x
    else:
        selected = -3 * x
    return result == selected


@equation
def nested_value(x: float, result: float) -> bool:
    if x > 0:
        if x > 10:
            selected = x**2
        else:
            selected = 2 * x
    else:
        if x < -10:
            selected = -3 * x
        else:
            selected = x**2
    return result == selected


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
@pytest.mark.parametrize(("x", "expected"), ((12.0, 24.0), (4.0, 2.0), (-2.0, -3.0)))
def test_elif_branch_derivatives(backend: Backend, x: float, expected: float) -> None:
    with backend, Implementation(MultiBranch, [elif_value]):
        model = MultiBranch(x=x)
        assert model.d.result(wrt=MultiBranch.x) == pytest.approx(expected)  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
@pytest.mark.parametrize(("x", "expected"), ((12.0, 24.0), (4.0, 2.0), (-4.0, -8.0), (-12.0, -3.0)))
def test_nested_if_else_derivatives(backend: Backend, x: float, expected: float) -> None:
    with backend, Implementation(MultiBranch, [nested_value]):
        model = MultiBranch(x=x)
        assert model.d.result(wrt=MultiBranch.x) == pytest.approx(expected)  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_switching_boundary_returns_branch_derivative_not_mathematical_derivative(backend: Backend) -> None:
    """At x=0 the function is not differentiable, despite selecting the else branch."""
    with backend, PIECEWISE:
        model = Piecewise(x=0.0)
        assert model.value == pytest.approx(0.0)
        assert model.d.value(wrt=Piecewise.x) == pytest.approx(-1.0)  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]


@pytest.mark.parametrize("x", (3.0, -3.0))
def test_conditional_derivatives_agree_with_jax_autodiff_away_from_switch(x: float) -> None:
    def direct(value: jax.Array) -> jax.Array:
        return jax.lax.cond(value > 0, lambda z: z**2, lambda z: -z, value)

    expected = jax.grad(direct)(jnp.asarray(x))
    with PYTHON_BACKEND, PIECEWISE:
        model = Piecewise(x=x)
        assert model.d.value(wrt=Piecewise.x) == pytest.approx(float(expected))  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]
