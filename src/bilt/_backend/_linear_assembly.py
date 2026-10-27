"""Broadcast and stack entries for independent linear systems."""

import typing

if typing.TYPE_CHECKING:
    import collections.abc
    import types


def assemble_pointwise(matrix: object, rhs: object, rank: int, *, xp: types.ModuleType) -> tuple[object, object]:
    rows = typing.cast("collections.abc.Sequence[collections.abc.Sequence[object]]", matrix)
    values = typing.cast("collections.abc.Sequence[object]", rhs)
    entries = [xp.asarray(value) for row in rows for value in row] + [xp.asarray(value) for value in values]
    entries = list(xp.broadcast_arrays(*entries))
    entries = [value.reshape((1,) * max(0, rank - value.ndim) + value.shape) for value in entries]
    width = len(rows)
    coefficients = xp.stack([xp.stack(entries[i * width : (i + 1) * width], axis=-1) for i in range(width)], axis=-2)
    return coefficients, xp.stack(entries[width * width :], axis=-1)
