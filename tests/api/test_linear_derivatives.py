"""Derivative queries through parameter-dependent coupled linear systems."""

import typing

import jax
import pytest

from bilt import JAX_BACKEND, PYTHON_BACKEND, Domain, Implementation, Interface, equation, field, system
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest

if typing.TYPE_CHECKING:
    from bilt._backend._base_backend import Backend


@system
class Coupled(metaclass=Interface):
    x: float = field()
    y: float = field()
    z: float = field()
    p: float = field()
    q: float = field()
    stiffness: float = field()


@equation
def balance(x: float, y: float, stiffness: float, p: float) -> bool:
    return stiffness * x + y == p


@equation
def difference(x: float, y: float, q: float) -> bool:
    return x - y == q


@equation
def product(z: float, x: float, y: float) -> bool:
    return z == x * y


COUPLED = Implementation(Coupled, [balance, difference, product])


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=["python", "jax"])
def test_coupled_implicit_derivatives(backend: Backend) -> None:
    with backend, COUPLED:
        sample = Coupled(p=10.0, q=2.0, stiffness=3.0)
        assert sample.x == pytest.approx(3.0)
        assert sample.y == pytest.approx(1.0)

        assert sample.d.x(wrt=Coupled.p) == pytest.approx(0.25)  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]
        assert sample.d.y(wrt=Coupled.p) == pytest.approx(0.25)  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]

        assert sample.d.x(wrt=Coupled.q) == pytest.approx(0.25)  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]
        assert sample.d.y(wrt=Coupled.q) == pytest.approx(-0.75)  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]

        # A depends on stiffness, so this exercises -dA*y, not just db.
        assert sample.d.x(wrt=Coupled.stiffness) == pytest.approx(-0.75)  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]
        assert sample.d.y(wrt=Coupled.stiffness) == pytest.approx(-0.75)  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=["python", "jax"])
def test_chain_rule_after_coupled_solve(backend: Backend) -> None:
    with backend, COUPLED:
        sample = Coupled(p=10.0, q=2.0, stiffness=3.0)
        assert sample.z == pytest.approx(3.0)
        assert sample.d.z(wrt=Coupled.p) == pytest.approx(1.0)  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]
        assert sample.d.z(wrt=Coupled.stiffness) == pytest.approx(-3.0)  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]


@system
class Driven(metaclass=Interface):
    p: float = field()
    q: float = field()
    t: float = field()
    stiffness: float = field()
    x: float = field()
    y: float = field()


@equation
def stiffness_from_t(t: float, stiffness: float) -> bool:
    return stiffness == 2 * t


DRIVEN = Implementation(Driven, [stiffness_from_t, balance, difference])


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=["python", "jax"])
def test_coefficient_derivative_uses_previous_equation_tangent(backend: Backend) -> None:
    with backend, DRIVEN:
        sample = Driven(p=10.0, q=2.0, t=1.5)
        assert sample.x == pytest.approx(3.0)
        assert sample.d.x(wrt=Driven.t) == pytest.approx(-1.5)  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]


def test_implicit_derivative_agrees_with_jax_autodiff() -> None:
    with EvaluationContext():
        compiled = COUPLED.get_evaluation(
            EvaluationRequest(
                output="x",
                input_domains={"p": Domain.scalar(), "q": Domain.scalar(), "stiffness": Domain.scalar()},
            ),
            JAX_BACKEND,
        )

    expected = jax.jacfwd(lambda k: compiled.callable(p=10.0, q=2.0, stiffness=k))(3.0)
    with COUPLED:
        sample = Coupled(p=10.0, q=2.0, stiffness=3.0)
        assert sample.d.x(wrt=Coupled.stiffness) == pytest.approx(float(expected), abs=1e-5)  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]
