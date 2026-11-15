"""Checks for the optional asdex/IPOPT Bilt optimisation example."""

import importlib.util

import numpy as np
import pytest

def test_cruise_model_evaluates_in_jax() -> None:
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from examples.optimisation.asdex_ipopt_cruise import evaluate_legs

    speeds = jnp.asarray([1.0, 2.0])
    distances = jnp.asarray([1.0, 3.0])
    energies, times = evaluate_legs(speeds, distances)

    np.testing.assert_allclose(np.asarray(energies), [2.0, 12.75], atol=1e-5)
    np.testing.assert_allclose(np.asarray(times), [1.0, 1.5], atol=1e-5)

    def objective(x: jax.Array) -> jax.Array:
        return jnp.sum(evaluate_legs(x, distances)[0])

    gradient = jax.grad(objective)(speeds)
    np.testing.assert_allclose(np.asarray(gradient), [0.0, 11.25], atol=1e-5)


@pytest.mark.skipif(
    importlib.util.find_spec("asdex") is None or importlib.util.find_spec("cyipopt") is None,
    reason="Optional asdex/cyipopt optimisation dependencies are not installed",
)
def test_ipopt_matches_analytic_solution() -> None:
    pytest.importorskip("jax")
    from examples.optimisation.asdex_ipopt_cruise import (
        DISTANCES,
        REQUIRED_SPEEDS,
        optimise_cruise_speeds,
    )

    result = optimise_cruise_speeds()
    expected = np.maximum(1.0, np.asarray(REQUIRED_SPEEDS))

    np.testing.assert_allclose(result.speeds, expected, atol=1e-5, rtol=0)
    assert result.jacobian_nonzeros == len(DISTANCES)
    assert result.hessian_lower_nonzeros == len(DISTANCES)
    assert np.all(np.asarray(DISTANCES) / result.speeds <= np.asarray(DISTANCES) / np.asarray(REQUIRED_SPEEDS) + 1e-7)
