__all__ = [
    "DomainRegistry",
    "DomainRegistryError",
]

import typing

import numpy as np

from bilt.symbolic._dimensions import CoordinateAxis

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.symbolic._dimensions import Dim
    from bilt.symbolic._domains import Domain


class DomainRegistryError(ValueError):
    """Raised when dimensions have conflicting evaluation bindings."""


class DomainRegistryDimensionsLockedError(RuntimeError):
    """Raised when trying to bind a new dimension to a locked domain registry."""


@typing.runtime_checkable
class _DomainValue(typing.Protocol):
    """A concrete value associated with a domain."""

    @property
    def domain(self) -> Domain: ...

    @property
    def shape(self) -> tuple[int, ...]: ...


class DomainRegistry:
    """Concrete dimension bindings for one evaluation.

    Each named dimension has one extent and may optionally have a
    coordinate axis.
    """

    def __init__(self) -> None:
        self._bindings: dict[Dim, int | CoordinateAxis] = {}
        self._locked_against_new_dimensions = False

    def bind(self, value: object) -> None:
        """Bind any domain information exposed by a value."""
        if isinstance(value, CoordinateAxis):
            self.bind_axis(value)
        elif isinstance(value, _DomainValue):
            self.bind_value(value)

    def bind_all(
        self,
        values: collections.abc.Iterable[object],
    ) -> None:
        """Bind domain information from multiple values."""
        for value in values:
            self.bind(value)

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
        existing = self._bindings.get(axis.dimension)

        if existing is None:
            if self._locked_against_new_dimensions:
                raise DomainRegistryDimensionsLockedError

            self._bindings[axis.dimension] = axis
            return

        if isinstance(existing, int):
            if existing != len(axis):
                message = f"Dimension {axis.dimension!s} has inconsistent sizes: {existing} and {len(axis)}."
                raise DomainRegistryError(message)

            if self._locked_against_new_dimensions:
                raise DomainRegistryDimensionsLockedError

            self._bindings[axis.dimension] = axis
            return

        if existing.coordinates != axis.coordinates:
            message = f"Dimension {axis.dimension!s} has conflicting coordinate definitions."
            raise DomainRegistryError(message)

    def size(self, dimension: Dim) -> int:
        """Return the size bound to a dimension."""
        binding = self._binding(dimension)

        if isinstance(binding, int):
            return binding

        return len(binding)

    def axis(self, dimension: Dim) -> CoordinateAxis:
        """Return the coordinate axis bound to a dimension."""
        binding = self._binding(dimension)

        if isinstance(binding, int):
            message = f"Dimension {dimension!s} has no coordinate axis."
            raise DomainRegistryError(message)

        return binding

    def position(self, dim: Dim, coordinate: object) -> int:
        axis = self.axis(dim)

        for position, candidate in enumerate(axis.coordinates):
            if candidate == coordinate:
                return position

            if isinstance(candidate, float) and isinstance(coordinate, float):
                tolerance = 8 * np.finfo(float).eps * max(1.0, abs(candidate), abs(coordinate))
                if abs(candidate - coordinate) <= tolerance:
                    return position

        message = f"Coordinate {coordinate!r} is not present on dimension {dim!r}."
        raise DomainRegistryError(message)

    def lock_against_new_dimensions(self) -> None:
        self._locked_against_new_dimensions = True

    def _bind_size(
        self,
        dimension: Dim,
        size: int,
    ) -> None:
        existing = self._bindings.get(dimension)

        if existing is None:
            if self._locked_against_new_dimensions:
                raise DomainRegistryDimensionsLockedError

            self._bindings[dimension] = size
            return

        existing_size = existing if isinstance(existing, int) else len(existing)

        if existing_size != size:
            message = f"Dimension {dimension!s} has inconsistent sizes: {existing_size} and {size}."
            raise DomainRegistryError(message)

    def _binding(self, dimension: Dim) -> int | CoordinateAxis:
        try:
            return self._bindings[dimension]
        except KeyError:
            message = f"Dimension {dimension!s} is not bound."
            raise DomainRegistryError(message) from None
