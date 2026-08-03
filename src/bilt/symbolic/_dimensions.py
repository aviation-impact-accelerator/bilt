__all__ = ["CoordinateAxis", "Dim", "coordinate_axis", "dim"]

import dataclasses
import typing


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class Dim:
    """A symbolic axis identified by its declaration."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclasses.dataclass(frozen=True, slots=True)
class CoordinateAxis:
    """Concrete coordinates associated with a symbolic dimension."""

    dimension: Dim
    coordinates: tuple[object, ...]

    def __len__(self) -> int:
        return len(self.coordinates)


def dim(name: str) -> Dim:
    """Create a symbolic dimension."""
    return Dim(name)


def coordinate_axis(
    dimension: Dim,
    coordinates: typing.Iterable[object],
) -> CoordinateAxis:
    """Associate concrete coordinates with a dimension."""
    return CoordinateAxis(
        dimension=dimension,
        coordinates=tuple(coordinates),
    )
