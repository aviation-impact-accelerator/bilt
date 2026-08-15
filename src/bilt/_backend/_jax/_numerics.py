import typing

import jax.numpy as jnp

if typing.TYPE_CHECKING:
    from bilt._backend._numerics import Array


class JaxNumerics:
    def stack(
        self,
        values: typing.Sequence[Array],
        *,
        axis: int,
    ) -> Array:
        return jnp.stack(values, axis=axis)

    def solve(
        self,
        matrix: Array,
        rhs: Array,
    ) -> Array:
        return jnp.linalg.solve(matrix, rhs)
