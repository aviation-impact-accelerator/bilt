import typing

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bilt._backend._jax._backend import _solve_linear

if typing.TYPE_CHECKING:
    from collections.abc import Callable


def test_jitted_linear_validation_sends_only_scalar_flags_to_host(monkeypatch: pytest.MonkeyPatch) -> None:
    original = jax.debug.callback
    operands = []

    def observe(callback: Callable, *args: jax.Array, ordered: bool = False) -> None:
        operands.append([(arg.shape, arg.dtype) for arg in args])
        original(callback, *args, ordered=ordered)

    monkeypatch.setattr(jax.debug, "callback", observe)
    result = jax.jit(_solve_linear)(jnp.eye(8), jnp.arange(8.0))
    np.testing.assert_allclose(result, np.arange(8.0))
    jax.effects_barrier()
    assert operands == [[((), np.dtype(bool)), ((), np.dtype(bool))]]


@pytest.mark.parametrize(
    ("matrix", "rhs", "message"),
    (
        ([[float("nan"), 0.0], [0.0, 1.0]], [1.0, 2.0], "coefficients and right-hand side must be finite"),
        ([[1.0, 0.0], [0.0, 1.0]], [1.0, float("inf")], "coefficients and right-hand side must be finite"),
        ([[1.0, 1.0], [1.0, 1.0]], [1.0, 2.0], "singular or the solution is non-finite"),
    ),
)
def test_jitted_linear_status_preserves_runtime_errors(matrix: list, rhs: list, message: str) -> None:
    with pytest.raises(jax.errors.JaxRuntimeError, match=message):
        jax.block_until_ready(jax.jit(_solve_linear)(jnp.asarray(matrix), jnp.asarray(rhs)))
