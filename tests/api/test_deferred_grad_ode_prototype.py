# ruff: noqa: PLR2004
"""Experimental unified grad notation for algebraic and ODE relationships."""

import math

import numpy as np
import pytest

from bilt.core._structural_analysis._causalization_block_solver._errors import BiltOverconstrainedError

from bilt import (
    JAX_BACKEND,
    PYTHON_BACKEND,
    Implementation,
    Interface,
    Tensor,
    at,
    dim,
    equation,
    field,
    grad,
    system,
    tensor,
)
from bilt.symbolic._differential import DEFERRED_GRAD
from bilt.symbolic._expressions import symbol_expr, tensor_symbol_expr

TIME = dim("time")


@system
class Decay(metaclass=Interface):
    t: Tensor = field()
    y: Tensor = field()
    rate: float = field()
    t0: float = field()
    y0: float = field()


@equation
def decay_rate(y: Tensor, t: Tensor, rate: float) -> bool:
    return grad(y, wrt=t) == -rate * y  # pyrefly: ignore[bad-return]  # ty: ignore[invalid-return-type]


@equation
def decay_initial(y: Tensor, t: Tensor, t0: float, y0: float) -> bool:
    return at(y, wrt=t, where=t0) == y0


DECAY = Implementation(Decay, [decay_rate, decay_initial])


def test_deferred_gradient_is_preserved_until_structural_context() -> None:
    x = tensor_symbol_expr("x")
    t = tensor_symbol_expr("t")
    formal = grad(x, wrt=t)
    assert formal.tree.head is DEFERRED_GRAD
    assert formal.tree.args == (x.tree, t.tree)
    scalar_x = symbol_expr("x")
    scalar_t = symbol_expr("t")
    assert grad(scalar_x, wrt=scalar_t).tree.head.value == 0.0
    assert grad(scalar_x**2, wrt=scalar_x).tree.head is not DEFERRED_GRAD


def test_decay_integration_from_shared_grad_operator() -> None:
    pytest.importorskip("scipy.integrate")
    times = np.linspace(0.0, 2.0, 41)
    with PYTHON_BACKEND, DECAY:
        case = Decay(t=tensor(times, over=TIME), rate=1.5, y0=3.0, t0=0.0)
        observed = case.y
        assert observed.domain == case.t.domain
        np.testing.assert_allclose(observed.values, 3 * np.exp(-1.5 * times), rtol=2e-7, atol=2e-7)


@system
class Oscillator(metaclass=Interface):
    t: Tensor = field()
    x: Tensor = field()
    v: Tensor = field()
    stiffness: float = field()
    mass: float = field()
    t0: float = field()
    x0: float = field()
    v0: float = field()


@equation
def position_rate(x: Tensor, v: Tensor, t: Tensor) -> bool:
    return grad(x, wrt=t) == v  # pyrefly: ignore[bad-return]  # ty: ignore[invalid-return-type]


@equation
def velocity_rate(x: Tensor, v: Tensor, t: Tensor, stiffness: float, mass: float) -> bool:
    potential = 0.5 * stiffness * x * x
    force = -grad(potential, wrt=x)
    return grad(v, wrt=t) == force / mass  # pyrefly: ignore[bad-return]  # ty: ignore[invalid-return-type]


@equation
def initial_x(x: Tensor, t: Tensor, t0: float, x0: float) -> bool:
    return at(x, wrt=t, where=t0) == x0


@equation
def initial_v(v: Tensor, t: Tensor, t0: float, v0: float) -> bool:
    return at(v, wrt=t, where=t0) == v0


OSCILLATOR = Implementation(Oscillator, [position_rate, velocity_rate, initial_x, initial_v])


def test_oscillator_uses_same_grad_for_potential_and_formal_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    scipy_integrate = pytest.importorskip("scipy.integrate")
    original = scipy_integrate.solve_ivp
    calls = []

    def counted(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(scipy_integrate, "solve_ivp", counted)
    times = np.linspace(0, math.pi, 101)
    with PYTHON_BACKEND, OSCILLATOR:
        model = Oscillator(
            t=tensor(times, over=TIME),
            stiffness=4.0,
            mass=1.0,
            t0=0.0,
            x0=1.0,
            v0=0.0,
        )
        x = model.x
        v = model.v
        assert len(calls) == 1
        np.testing.assert_allclose(x.values, np.cos(2 * times), atol=3e-8)
        np.testing.assert_allclose(v.values, -2 * np.sin(2 * times), atol=5e-8)


@system
class Lorenz(metaclass=Interface):
    t: Tensor = field()
    x: Tensor = field()
    y: Tensor = field()
    z: Tensor = field()
    sigma: float = field()
    rho: float = field()
    beta: float = field()
    t0: float = field()
    x0: float = field()
    y0: float = field()
    z0: float = field()


@equation
def lorenz_x(x: Tensor, y: Tensor, t: Tensor, sigma: float) -> bool:
    return grad(x, wrt=t) == sigma * (y - x)  # pyrefly: ignore[bad-return]  # ty: ignore[invalid-return-type]


@equation
def lorenz_y(x: Tensor, y: Tensor, z: Tensor, t: Tensor, rho: float) -> bool:
    return grad(y, wrt=t) == x * (rho - z) - y  # pyrefly: ignore[bad-return]  # ty: ignore[invalid-return-type]


@equation
def lorenz_z(x: Tensor, y: Tensor, z: Tensor, t: Tensor, beta: float) -> bool:
    return grad(z, wrt=t) == x * y - beta * z  # pyrefly: ignore[bad-return]  # ty: ignore[invalid-return-type]


@equation
def lorenz_initial_x(x: Tensor, t: Tensor, t0: float, x0: float) -> bool:
    return at(x, wrt=t, where=t0) == x0


@equation
def lorenz_initial_y(y: Tensor, t: Tensor, t0: float, y0: float) -> bool:
    return at(y, wrt=t, where=t0) == y0


@equation
def lorenz_initial_z(z: Tensor, t: Tensor, t0: float, z0: float) -> bool:
    return at(z, wrt=t, where=t0) == z0


LORENZ = Implementation(
    Lorenz,
    [lorenz_x, lorenz_y, lorenz_z, lorenz_initial_x, lorenz_initial_y, lorenz_initial_z],
)


def test_lorenz_shared_three_state_trajectory() -> None:
    pytest.importorskip("scipy.integrate")
    times = np.linspace(0.0, 0.5, 51)
    with PYTHON_BACKEND, LORENZ:
        model = Lorenz(
            t=tensor(times, over=TIME),
            sigma=10.0,
            rho=28.0,
            beta=8 / 3,
            t0=0.0,
            x0=1.0,
            y0=1.0,
            z0=1.0,
        )
        assert len(model.x.values) == 51
        assert model.y.values[0] == pytest.approx(1.0)
        assert model.z.values[0] == pytest.approx(1.0)
        assert model.x.values[1] > model.x.values[0]


def test_missing_initial_condition_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one rate and initial condition"):
        Implementation(Decay, [decay_rate])


def test_scalar_grad_remains_algebraic() -> None:
    x = symbol_expr("x")
    t = symbol_expr("t")
    assert grad(x, wrt=t).tree.head.value == 0.0
    assert grad(x**3, wrt=x).tree.head is not DEFERRED_GRAD


def test_numerical_ode_runtime_is_explicit_about_backend() -> None:
    with JAX_BACKEND, DECAY:
        model = Decay(t=tensor([0.0, 1.0], over=TIME), rate=2.0, t0=0.0, y0=1.0)
        with pytest.raises(NotImplementedError, match="Python backend"):
            _ = model.y


def test_nonincreasing_output_coordinates_rejected() -> None:
    with PYTHON_BACKEND, DECAY:
        model = Decay(t=tensor([0.0, 1.0, 1.0], over=TIME), rate=2.0, t0=0.0, y0=1.0)
        with pytest.raises(ValueError, match="strictly increasing"):
            _ = model.y


@equation
def unsupported_expression_rate(y: Tensor, t: Tensor, rate: float) -> bool:
    return grad(y * y, wrt=t) == -rate * y  # pyrefly: ignore[bad-return]  # ty: ignore[invalid-return-type]


def test_unknown_expression_rate_never_silently_becomes_zero() -> None:
    with pytest.raises(ValueError, match="named state and coordinate"):
        Implementation(Decay, [unsupported_expression_rate, decay_initial])


@equation
def unsupported_state_constraint(y: Tensor, rate: float) -> bool:
    return y == rate


def test_additional_state_constraint_is_not_ignored() -> None:
    constrained = Implementation(Decay, [decay_rate, decay_initial, unsupported_state_constraint])
    with PYTHON_BACKEND, constrained:
        model = Decay(t=tensor([0.0, 1.0], over=TIME), rate=2.0, t0=0.0, y0=1.0)
        with pytest.raises(BiltOverconstrainedError, match="Overconstrained"):
            _ = model.y


@system
class SharedRHS(metaclass=Interface):
    t: Tensor = field()
    x: Tensor = field()
    v: Tensor = field()
    restoring: float = field()
    stiffness: float = field()
    t0: float = field()
    x0: float = field()
    v0: float = field()


@equation
def shared_rate_x(x: Tensor, t: Tensor, restoring: float) -> bool:
    return grad(x, wrt=t) == restoring  # pyrefly: ignore[bad-return]  # ty: ignore[invalid-return-type]


@equation
def shared_rate_v(v: Tensor, t: Tensor, restoring: float) -> bool:
    return grad(v, wrt=t) == 2 * restoring  # pyrefly: ignore[bad-return]  # ty: ignore[invalid-return-type]


@equation
def shared_restoring(x: Tensor, restoring: float, stiffness: float) -> bool:
    return restoring == -stiffness * x


SHARED_RHS = Implementation(
    SharedRHS,
    [shared_rate_x, shared_rate_v, shared_restoring, initial_x, initial_v],
)


def test_shared_algebraic_rhs_intermediate() -> None:
    pytest.importorskip("scipy.integrate")
    times = np.linspace(0.0, 1.0, 41)
    with PYTHON_BACKEND, SHARED_RHS:
        model = SharedRHS(
            t=tensor(times, over=TIME),
            stiffness=0.7,
            t0=0.0,
            x0=3.0,
            v0=5.0,
        )
        expected_x = 3.0 * np.exp(-0.7 * times)
        np.testing.assert_allclose(model.x.values, expected_x, rtol=2e-7, atol=2e-7)
        np.testing.assert_allclose(model.v.values, 5.0 + 2 * (expected_x - 3.0), rtol=2e-7, atol=2e-7)


@equation
def conflicting_restoring(x: Tensor, restoring: float, stiffness: float) -> bool:
    return restoring == -2 * stiffness * x


def test_full_scope_rejects_competing_algebraic_rows() -> None:
    constrained = Implementation(
        SharedRHS,
        [shared_rate_x, shared_rate_v, shared_restoring, conflicting_restoring, initial_x, initial_v],
    )
    with PYTHON_BACKEND, constrained:
        model = SharedRHS(
            t=tensor([0.0, 1.0], over=TIME),
            stiffness=0.7,
            t0=0.0,
            x0=3.0,
            v0=5.0,
        )
        with pytest.raises(BiltOverconstrainedError, match="Overconstrained"):
            _ = model.x
