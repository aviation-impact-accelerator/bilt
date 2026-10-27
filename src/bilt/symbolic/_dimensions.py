__all__ = [
    "Coordinate",
    "CoordinateAxis",
    "Dim",
    "coordinate_axis",
    "dim",
]

import dataclasses
import datetime as dt
import typing

if typing.TYPE_CHECKING:
    import collections.abc

type Coordinate = str | int | float | dt.date | dt.datetime


@dataclasses.dataclass(frozen=True, slots=True)
class Dim:
    """A named symbolic tensor dimension."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclasses.dataclass(frozen=True, slots=True)
class CoordinateAxis:
    """Ordered coordinates associated with a symbolic dimension."""

    dimension: Dim
    coordinates: tuple[Coordinate, ...]

    def __len__(self) -> int:
        return len(self.coordinates)


def dim(name: str) -> Dim:
    """Create a named symbolic dimension."""
    return Dim(name)


def coordinate_axis(
    dimension: Dim,
    coordinates: collections.abc.Iterable[Coordinate],
) -> CoordinateAxis:
    """Associate ordered coordinate values with a dimension."""
    return CoordinateAxis(
        dimension=dimension,
        coordinates=tuple(coordinates),
    )
