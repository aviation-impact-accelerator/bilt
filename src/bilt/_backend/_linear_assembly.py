"""Array assembly for pointwise batches and joint tensor linear blocks."""

import math
import typing

from bilt._backend._errors import BiltLinearSolveError

if typing.TYPE_CHECKING:
    import collections.abc
    import types

# Each axis tuple names the canonical axes present in an unknown or equation.
type LinearLayout = tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], int]


def assemble_pointwise(matrix: object, rhs: object, rank: int, *, xp: types.ModuleType) -> tuple[object, object]:
    rows = typing.cast("collections.abc.Sequence[collections.abc.Sequence[object]]", matrix)
    values = typing.cast("collections.abc.Sequence[object]", rhs)
    entries = [xp.asarray(value) for row in rows for value in row] + [xp.asarray(value) for value in values]
    entries = list(xp.broadcast_arrays(*entries))
    entries = [value.reshape((1,) * max(0, rank - value.ndim) + value.shape) for value in entries]
    width = len(rows)
    coefficients = xp.stack([xp.stack(entries[i * width : (i + 1) * width], axis=-1) for i in range(width)], axis=-2)
    return coefficients, xp.stack(entries[width * width :], axis=-1)


def _shapes(
    values: tuple[object, ...], layout: LinearLayout, xp: types.ModuleType
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    unknown_axes, row_axes, rank = layout
    shape = tuple(xp.broadcast_shapes(*[xp.shape(value) for value in values])) if values else ()
    shape = (1,) * (rank - len(shape)) + shape

    def shape_for(axes: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(shape[i] if i in axes else 1 for i in range(rank))

    return tuple(map(shape_for, unknown_axes)), tuple(map(shape_for, row_axes))


def linear_basis(values: tuple[object, ...], layout: LinearLayout, *, xp: types.ModuleType) -> tuple[object, ...]:
    """Expose all block unknowns as columns of one exact identity basis."""
    shapes, _ = _shapes(values, layout, xp)
    widths = tuple(map(math.prod, shapes))
    size = sum(widths)
    identity = xp.eye(size)
    result = []
    offset = 0
    for shape, width in zip(shapes, widths, strict=True):
        result.append(identity[offset : offset + width].reshape((*shape, size)))
        offset += width
    return tuple(result)


def expand_linear_coefficient(value: object, *, xp: types.ModuleType) -> object:
    """Keep the identity-basis columns separate from named tensor axes."""
    return xp.asarray(value)[..., None]


def assemble_joint(
    matrix: object, rhs: object, values: tuple[object, ...], layout: LinearLayout, *, xp: types.ModuleType
) -> tuple[object, object]:
    unknown_shapes, row_shapes = _shapes(values, layout, xp)
    width = sum(map(math.prod, unknown_shapes))
    height = sum(map(math.prod, row_shapes))
    if width != height:
        problem = f"tensor block has {height} scalar equations for {width} scalar unknowns"
        raise BiltLinearSolveError(problem)
    rows = typing.cast("collections.abc.Sequence[object]", matrix)
    constants = typing.cast("collections.abc.Sequence[object]", rhs)
    coefficients = xp.concatenate(
        [
            xp.broadcast_to(xp.asarray(row), (*shape, width)).reshape((-1, width))
            for row, shape in zip(rows, row_shapes, strict=True)
        ],
        axis=0,
    )
    vector = xp.concatenate(
        [
            xp.broadcast_to(xp.asarray(value), shape).reshape(-1)
            for value, shape in zip(constants, row_shapes, strict=True)
        ],
        axis=0,
    )
    return coefficients, vector


def project_joint(
    solution: object, values: tuple[object, ...], layout: LinearLayout, *, xp: types.ModuleType
) -> tuple[object, ...]:
    shapes, _ = _shapes(values, layout, xp)
    vector = xp.asarray(solution)
    result = []
    offset = 0
    for shape in shapes:
        width = math.prod(shape)
        result.append(vector[offset : offset + width].reshape(shape))
        offset += width
    return tuple(result)
