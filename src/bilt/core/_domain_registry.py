__all__ = [
    "DomainRegistry",
    "DomainRegistryError",
]

import dataclasses
import typing

if typing.TYPE_CHECKING:
    from bilt.symbolic._dimensions import Coordinate, CoordinateAxis, Dim
    from bilt.symbolic._domains import Domain


class DomainRegistryError(ValueError):
    """Raised when dimensions have conflicting evaluation bindings."""


class _DomainValue(typing.Protocol):
    """A concrete value associated with a domain."""

    @property
    def domain(self) -> Domain: ...

    @property
    def shape(self) -> tuple[int, ...]: ...


@dataclasses.dataclass(frozen=True, slots=True)
class _DimensionBinding:
    size: int
    axis: CoordinateAxis | None = None


class DomainRegistry:
    """Dimension sizes and coordinates for one evaluation."""

    def __init__(self) -> None:
        self._bindings: dict[Dim, _DimensionBinding] = {}

    def bind_value(self, value: _DomainValue) -> None:
        """Bind the dimensions discovered from a concrete value."""
        if len(value.domain.dims) != len(value.shape):
            message = f"Value rank {len(value.shape)} does not match domain rank {len(value.domain.dims)}."
            raise DomainRegistryError(message)

        for dimension, size in zip(
            value.domain.dims,
            value.shape,
            strict=True,
        ):
            self._bind_size(dimension, size)

    def bind_axis(self, axis: CoordinateAxis) -> None:
        """Bind concrete coordinates to a dimension."""
        self._bind_size(axis.dimension, len(axis))

        existing = self._bindings[axis.dimension]

        if existing.axis is not None and existing.axis.coordinates != axis.coordinates:
            message = f"Dimension {axis.dimension!s} has conflicting coordinate definitions."
            raise DomainRegistryError(message)

        self._bindings[axis.dimension] = dataclasses.replace(
            existing,
            axis=axis,
        )

    def size(self, dimension: Dim) -> int:
        """Return the size bound to a dimension."""
        return self._binding(dimension).size

    def axis(self, dimension: Dim) -> CoordinateAxis:
        """Return the coordinate axis bound to a dimension."""
        binding = self._binding(dimension)

        if binding.axis is None:
            message = f"Dimension {dimension!s} has no coordinate axis."
            raise DomainRegistryError(message)

        return binding.axis

    def position(
        self,
        dimension: Dim,
        coordinate: Coordinate,
    ) -> int:
        """Return the position of a coordinate on a dimension."""
        axis = self.axis(dimension)

        try:
            return axis.coordinates.index(coordinate)
        except ValueError:
            message = f"Coordinate {coordinate!r} is not present on dimension {dimension!s}."
            raise DomainRegistryError(message) from None

    def _bind_size(
        self,
        dimension: Dim,
        size: int,
    ) -> None:
        existing = self._bindings.get(dimension)

        if existing is None:
            self._bindings[dimension] = _DimensionBinding(size=size)
            return

        if existing.size != size:
            message = f"Dimension {dimension!s} has inconsistent sizes: {existing.size} and {size}."
            raise DomainRegistryError(message)

    def _binding(self, dimension: Dim) -> _DimensionBinding:
        try:
            return self._bindings[dimension]
        except KeyError:
            message = f"Dimension {dimension!s} is not bound."
            raise DomainRegistryError(message) from None
