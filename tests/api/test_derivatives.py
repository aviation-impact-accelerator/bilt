"""Public instance derivative queries for scalar acausal equations."""

import pytest

from bilt import Implementation, Interface, field, system
from bilt.core._equation import equation
from bilt.errors import BiltUnsupportedDerivativeError


@system
class Spring(metaclass=Interface):
    force: float = field()
    stiffness: float = field()
    displacement: float = field()


@equation
def spring_law(force: float, stiffness: float, displacement: float) -> bool:
    return force == stiffness * displacement


SPRING = Implementation(Spring, [spring_law])


@pytest.mark.parametrize("backend_name", ("python", "jax"))
def test_spring_acausal_derivatives(backend_name: str) -> None:
    if backend_name == "jax":
        from bilt import JAX_BACKEND

        backend = JAX_BACKEND
    else:
        from bilt import PYTHON_BACKEND

        backend = PYTHON_BACKEND

    with backend, SPRING:
        spring = Spring(force=100.0, stiffness=20.0)
        assert spring.displacement == pytest.approx(5.0)
        assert spring.d.displacement(wrt=Spring.force) == pytest.approx(0.05)
        assert spring.d.displacement(wrt=Spring.stiffness) == pytest.approx(-0.25)
        assert spring.diff.displacement(wrt="force") == pytest.approx(0.05)

        reversed_spring = Spring(stiffness=20.0, displacement=5.0)
        assert reversed_spring.d.force(wrt=Spring.displacement) == pytest.approx(20.0)
        assert reversed_spring.d.force(wrt=Spring.stiffness) == pytest.approx(5.0)


@system
class Chain(metaclass=Interface):
    x: float = field()
    a: float = field()
    b: float = field()


@equation
def first(x: float, a: float) -> bool:
    return a == 2 * x


@equation
def second(a: float, b: float) -> bool:
    return b == a * a


CHAIN = Implementation(Chain, [first, second])


def test_chain_rule_across_equations() -> None:
    with CHAIN:
        chain = Chain(x=3.0)
        assert chain.d.b(wrt=Chain.x) == pytest.approx(24.0)
        assert chain.b == pytest.approx(36.0)


def test_requires_supplied_input() -> None:
    with SPRING:
        spring = Spring(force=100.0, stiffness=20.0)
        with pytest.raises(BiltUnsupportedDerivativeError, match="was not supplied"):
            spring.d.displacement(wrt=Spring.displacement)


def test_supplied_output_has_identity_derivative() -> None:
    spring = Spring(force=100.0, stiffness=20.0, displacement=5.0)
    with SPRING:
        assert spring.d.force(wrt=Spring.force) == pytest.approx(1.0)
        assert spring.d.force(wrt=Spring.stiffness) == pytest.approx(0.0)


def test_invalid_field_name() -> None:
    with SPRING:
        spring = Spring(force=100.0, stiffness=20.0)
        with pytest.raises(ValueError, match="Unknown differentiation field"):
            spring.d.displacement(wrt="unknown")


def test_existing_d_field_is_not_overwritten() -> None:
    @system
    class HasD(metaclass=Interface):
        d: float = field()
        value: float = field()

    instance = HasD(d=2.0, value=4.0)
    assert instance.d == 2.0
    assert hasattr(instance, "diff")


def test_numpy_backend_requires_no_jax() -> None:
    with SPRING:
        spring = Spring(force=100.0, stiffness=20.0)
        assert spring.d.displacement(wrt=Spring.force) == pytest.approx(0.05)
