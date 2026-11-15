"""First- and second-order parameter sensitivity queries."""

import typing

import jax
import jax.numpy as jnp
import pytest

from bilt import JAX_BACKEND, PYTHON_BACKEND, Implementation, Interface, equation, field, system

if typing.TYPE_CHECKING:
    from bilt._backend._base_backend import Backend


@system
class Polynomial(metaclass=Interface):
    x: float = field()
    y: float = field()
    energy: float = field()
    force: float = field()


@equation
def energy_law(energy: float, x: float, y: float) -> bool:
    return energy == x**3 + x * y + y**2


@equation
def force_law(force: float, x: float, y: float) -> bool:
    return force == 2 * x + 3 * y


POLYNOMIAL = Implementation(Polynomial, [energy_law, force_law])


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_gradient_of_scalar_output(backend: Backend) -> None:
    with backend, POLYNOMIAL:
        model = Polynomial(x=2.0, y=3.0)
        assert model.energy == pytest.approx(23.0)
        assert model.grad.energy(wrt=(Polynomial.x, Polynomial.y)) == pytest.approx((15.0, 8.0))  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]
        assert model.grad.energy(wrt=("y", "x")) == pytest.approx((8.0, 15.0))  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_hessian_mixed_partials(backend: Backend) -> None:
    with backend, POLYNOMIAL:
        model = Polynomial(x=2.0, y=3.0)
        actual = model.hessian.energy(wrt=(Polynomial.x, Polynomial.y))  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]
        assert actual[0] == pytest.approx((12.0, 1.0))
        assert actual[1] == pytest.approx((1.0, 2.0))


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_jacobian_scalar_and_multiple_named_outputs(backend: Backend) -> None:
    with backend, POLYNOMIAL:
        model = Polynomial(x=2.0, y=3.0)
        assert model.jacobian.energy(wrt=(Polynomial.x, Polynomial.y))[0] == pytest.approx((15.0, 8.0))  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]
        matrix = model.jacobian((Polynomial.energy, Polynomial.force), wrt=(Polynomial.x, Polynomial.y))  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]
        assert matrix[0] == pytest.approx((15.0, 8.0))
        assert matrix[1] == pytest.approx((2.0, 3.0))


@system
class Chain(metaclass=Interface):
    x: float = field()
    y: float = field()
    intermediate: float = field()
    output: float = field()


@equation
def intermediate_relation(intermediate: float, x: float, y: float) -> bool:
    return intermediate == x * y


@equation
def output_relation(output: float, intermediate: float, x: float) -> bool:
    return output == intermediate**2 + x


CHAIN = Implementation(Chain, [intermediate_relation, output_relation])


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_hessian_across_chained_equations(backend: Backend) -> None:
    with backend, CHAIN:
        model = Chain(x=2.0, y=3.0)
        assert model.grad.output(wrt=(Chain.x, Chain.y)) == pytest.approx((37.0, 24.0))  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]
        actual = model.hessian.output(wrt=(Chain.x, Chain.y))  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]
        assert actual[0] == pytest.approx((18.0, 24.0))
        assert actual[1] == pytest.approx((24.0, 8.0))


def test_against_jax_grad_and_hessian() -> None:
    def energy(xy: jax.Array) -> jax.Array:
        x, y = xy
        return x**3 + x * y + y**2

    values = jnp.array((2.0, 3.0))
    expected_grad = jax.grad(energy)(values)
    expected_hess = jax.hessian(energy)(values)
    with PYTHON_BACKEND, POLYNOMIAL:
        model = Polynomial(x=2.0, y=3.0)
        assert model.grad.energy(wrt=(Polynomial.x, Polynomial.y)) == pytest.approx(tuple(expected_grad.tolist()))  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]
        actual = model.hessian.energy(wrt=(Polynomial.x, Polynomial.y))  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]
        assert actual[0] == pytest.approx(tuple(expected_hess.tolist()[0]))
        assert actual[1] == pytest.approx(tuple(expected_hess.tolist()[1]))


def test_supplied_output_has_zero_hessian_and_jacobian_identity() -> None:
    with POLYNOMIAL:
        model = Polynomial(x=2.0, y=3.0, energy=23.0)
        assert model.hessian.energy(wrt=(Polynomial.x, Polynomial.y)) == ((0.0, 0.0), (0.0, 0.0))  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]
        assert model.jacobian((Polynomial.energy, Polynomial.x), wrt=(Polynomial.x, Polynomial.y)) == (  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]
            (0.0, 0.0),
            (1.0, 0.0),
        )


@pytest.mark.parametrize("namespace", ("grad", "jacobian", "hessian"))
def test_empty_and_repeated_inputs_rejected(namespace: str) -> None:
    with POLYNOMIAL:
        model = Polynomial(x=2.0, y=3.0)
        query = getattr(getattr(model, namespace), "energy")
        with pytest.raises(ValueError, match="non-empty tuple"):
            query(wrt=())
        with pytest.raises(ValueError, match="duplicate fields"):
            query(wrt=("x", "x"))


def test_jacobian_rejects_duplicate_outputs() -> None:
    with POLYNOMIAL:
        model = Polynomial(x=2.0, y=3.0)
        with pytest.raises(ValueError, match="duplicate fields"):
            model.jacobian((Polynomial.energy, Polynomial.energy), wrt=("x", "y"))  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]


def test_field_namespace_does_not_overwrite_declared_fields() -> None:
    @system
    class OwnGrad(metaclass=Interface):
        grad: float = field()
        hessian: float = field()
        jacobian: float = field()
        output: float = field()

    model = OwnGrad(grad=1.0, hessian=2.0, jacobian=3.0)
    assert model.grad == 1.0
    assert model.hessian == 2.0
    assert model.jacobian == 3.0


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_existing_single_derivative_remains_supported(backend: Backend) -> None:
    with backend, POLYNOMIAL:
        model = Polynomial(x=2.0, y=3.0)
        assert model.d.energy(wrt=Polynomial.x) == pytest.approx(15.0)  # pyrefly: ignore[missing-attribute]  # ty: ignore[missing-attribute]
