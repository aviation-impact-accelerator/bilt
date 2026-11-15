"""Checks for the optional asdex/IPOPT Bilt optimisation example."""

import importlib
import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from examples.optimisation.asdex_ipopt_cruise import (
    DISTANCES,
    REQUIRED_SPEEDS,
    compile_cruise_legs,
    optimise_cruise_speeds,
)


def test_lowered_bilt_cruise_model_is_differentiable() -> None:
    """Check JAX AD acts on compiled Bilt numerics, not Bilt evaluation."""
    compiled = compile_cruise_legs(np.asarray([1.0, 3.0]))
    speeds = jnp.asarray([1.0, 2.0])
    energies, times = compiled(speeds)

    np.testing.assert_allclose(np.asarray(energies), [2.0, 12.75], atol=1e-5)
    np.testing.assert_allclose(np.asarray(times), [1.0, 1.5], atol=1e-5)

    def objective(x: jax.Array) -> jax.Array:
        return jnp.sum(compiled(x)[0])

    gradient = jax.grad(objective)(speeds)
    np.testing.assert_allclose(np.asarray(gradient), [0.0, 11.25], atol=1e-5)


@pytest.mark.skipif(
    importlib.util.find_spec("asdex") is None,
    reason="Optional asdex dependency is not installed",
)
def test_sparse_derivatives_match_dense_jax() -> None:
    """Compare sparse derivatives of lowered Bilt numerics with dense JAX AD."""
    asdex = importlib.import_module("asdex")
    jax.config.update("jax_enable_x64", True)  # noqa: FBT003

    distances = jnp.asarray(DISTANCES, dtype=jnp.float64)
    required = jnp.asarray(REQUIRED_SPEEDS, dtype=jnp.float64)
    time_limits = distances / required
    compiled = compile_cruise_legs(np.asarray(DISTANCES))
    n = len(DISTANCES)

    def objective(speeds: jax.Array) -> jax.Array:
        energies, _ = compiled(speeds)
        return jnp.sum(energies)

    def constraints(speeds: jax.Array) -> jax.Array:
        _, times = compiled(speeds)
        return times - time_limits

    def lagrangian(speeds: jax.Array, multipliers: jax.Array, factor: jax.Array) -> jax.Array:
        return factor * objective(speeds) + jnp.dot(multipliers, constraints(speeds))

    initial = jnp.asarray([1.15, 1.26, 0.93, 1.51, 1.20, 1.08, 1.62, 1.35])
    sparse_jacobian = jax.jit(asdex.jacobian(constraints, initial))
    sparse_hessian = jax.jit(asdex.hessian(lagrangian, initial, jnp.zeros(n), jnp.asarray(1.0)))
    dense_jacobian = jax.jit(jax.jacrev(constraints))
    dense_hessian = jax.jit(jax.hessian(lagrangian, argnums=0))

    expected_positions = {(i, i) for i in range(n)}
    jacobian_positions: np.ndarray | None = None
    hessian_positions: np.ndarray | None = None

    # Test different speeds, multipliers and objective factors. The fixed
    # sparsity detected at the initial point must remain valid for all three.
    cases = (
        (initial, jnp.linspace(-0.5, 0.7, n), 1.0),
        (initial * 1.09, jnp.linspace(0.2, 1.0, n), 0.4),
        (initial * 0.91, jnp.linspace(-0.8, 0.3, n), 0.0),
    )
    for speeds, multipliers, objective_factor in cases:
        factor = jnp.asarray(objective_factor)
        jac = sparse_jacobian(speeds)
        hess = sparse_hessian(speeds, multipliers, factor)

        np.testing.assert_allclose(
            np.asarray(jac.todense()),
            np.asarray(dense_jacobian(speeds)),
            rtol=1e-8,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            np.asarray(hess.todense()),
            np.asarray(dense_hessian(speeds, multipliers, factor)),
            rtol=1e-8,
            atol=1e-9,
        )

        # IPOPT expects its sparse coordinate ordering not to change between
        # the structure and numerical-value callbacks.
        jac_indices = np.asarray(jac.indices)
        hess_indices = np.asarray(hess.indices)
        assert {tuple(position) for position in jac_indices} == expected_positions
        assert {tuple(position) for position in hess_indices} == expected_positions
        if jacobian_positions is not None:
            np.testing.assert_array_equal(jac_indices, jacobian_positions)
            np.testing.assert_array_equal(hess_indices, hessian_positions)
        jacobian_positions = jac_indices
        hessian_positions = hess_indices


@pytest.mark.skipif(
    importlib.util.find_spec("asdex") is None or importlib.util.find_spec("cyipopt") is None,
    reason="Optional asdex/cyipopt optimisation dependencies are not installed",
)
def test_ipopt_matches_analytic_solution() -> None:
    """Check sparse nonlinear optimisation against a known optimum."""
    result = optimise_cruise_speeds()
    expected = np.maximum(1.0, np.asarray(REQUIRED_SPEEDS))

    np.testing.assert_allclose(result.speeds, expected, atol=1e-5, rtol=0)
    assert result.jacobian_nonzeros == len(DISTANCES)
    assert result.hessian_lower_nonzeros == len(DISTANCES)
    assert np.all(np.asarray(DISTANCES) / result.speeds <= np.asarray(DISTANCES) / np.asarray(REQUIRED_SPEEDS) + 1e-7)
