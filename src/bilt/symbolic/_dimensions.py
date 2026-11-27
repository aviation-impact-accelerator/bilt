__all__ = [
    "Coordinate",
    "CoordinateAxis",
    "Dim",
    "coordinate_axis",
    "dim",
]

import dataclasses
import datetime as dt
import enum
import typing

from bilt.symbolic._errors import BiltInvalidDomainError

if typing.TYPE_CHECKING:
    import collections.abc

type Coordinate = str | int | float | dt.date | dt.datetime | enum.Enum


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

    def __post_init__(self) -> None:
        members = [item for item in self.coordinates if isinstance(item, enum.Enum)]
        if not members:
            return
        expected = type(members[0])
        if len(members) != len(self.coordinates) or any(type(item) is not expected for item in members):
            msg = "A categorical axis must contain members of one exact enum class."
            raise BiltInvalidDomainError(msg)
        if len(set(members)) != len(members):
            msg = "A categorical axis cannot repeat an enum member."
            raise BiltInvalidDomainError(msg)

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
