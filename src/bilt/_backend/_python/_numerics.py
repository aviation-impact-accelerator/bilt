import typing

import numpy as np
import scipy.linalg

if typing.TYPE_CHECKING:
    from bilt._backend._numerics import Array


class ScipyNumerics:
    def stack(
        self,
        values: typing.Sequence[Array],
        *,
        axis: int,
    ) -> Array:
        return np.stack(values, axis=axis)

    def solve(
        self,
        matrix: Array,
        rhs: Array,
    ) -> Array:
        return scipy.linalg.solve(matrix, rhs)
