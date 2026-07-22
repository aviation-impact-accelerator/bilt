__all__ = [
    "DomainBindingError",
    "DomainRegistry",
    "DomainShapeError",
]

import dataclasses
import typing

from bilt.core._tensor._error import TensorDomainError

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.symbolic._dim import Dim
    from bilt.symbolic._domains import Domain


class DomainBindingError(TensorDomainError):
    """Raised when a dimension binding is missing or inconsistent."""


class DomainShapeError(TensorDomainError):
    """Raised when a concrete shape does not match its symbolic domain."""


@dataclasses.dataclass(frozen=True, slots=True)
class _DomainBinding:
    size: int
    coordinates: tuple[object, ...] | None = None


class DomainRegistry:
    """Runtime dimension bindings for one evaluation."""

    __slots__ = ("_bindings",)

    def __init__(self) -> None:
        self._bindings: dict[Dim, _DomainBinding] = {}

    def observe_shape(
        self,
        domain: Domain,
        shape: tuple[int, ...],
    ) -> None:
        """Record or validate dimension sizes from a concrete tensor."""
        if len(domain.dims) != len(shape):
            message = f"Domain rank {len(domain.dims)} does not match shape {shape!r}."
            raise DomainShapeError(message)

        for dimension, size in zip(domain.dims, shape, strict=True):
            self._observe_size(dimension, size)

    def bind(
        self,
        dimension: Dim,
        coordinates: collections.abc.Iterable[object],
    ) -> None:
        """Bind concrete coordinates to a dimension."""
        coordinate_values = tuple(coordinates)
        candidate = _DomainBinding(
            size=len(coordinate_values),
            coordinates=coordinate_values,
        )
        existing = self._bindings.get(dimension)

        if existing is None:
            self._bindings[dimension] = candidate
            return

        if existing.size != candidate.size:
            message = (
                f"Dimension {dimension!r} is already bound with size {existing.size}; received size {candidate.size}."
            )
            raise DomainBindingError(message)

        if existing.coordinates is None:
            self._bindings[dimension] = candidate
            return

        if existing.coordinates != candidate.coordinates:
            message = f"Dimension {dimension!r} is already bound to different coordinates."
            raise DomainBindingError(message)

    def validate_shape(
        self,
        domain: Domain,
        shape: tuple[int, ...],
    ) -> None:
        """Validate a tensor against dimensions already in the registry."""
        if len(domain.dims) != len(shape):
            message = f"Domain rank {len(domain.dims)} does not match shape {shape!r}."
            raise DomainShapeError(message)

        for axis, (dimension, actual) in enumerate(zip(domain.dims, shape, strict=True)):
            expected = self.size(dimension)

            if actual != expected:
                message = f"Axis {axis} for dimension {dimension!r} has size {actual}; expected {expected}."
                raise DomainShapeError(message)

    def coordinates(self, dimension: Dim) -> tuple[object, ...]:
        binding = self._binding(dimension)

        if binding.coordinates is None:
            message = f"Dimension {dimension!r} has a known size but no registered coordinates."
            raise DomainBindingError(message)

        return binding.coordinates

    def size(self, dimension: Dim) -> int:
        return self._binding(dimension).size

    def _observe_size(self, dimension: Dim, size: int) -> None:
        existing = self._bindings.get(dimension)

        if existing is None:
            self._bindings[dimension] = _DomainBinding(size=size)
            return

        if existing.size != size:
            message = f"Dimension {dimension!r} has size {size}; expected {existing.size}."
            raise DomainShapeError(message)

    def _binding(self, dimension: Dim) -> _DomainBinding:
        try:
            return self._bindings[dimension]
        except KeyError:
            message = f"Dimension {dimension!r} has not been bound."
            raise DomainBindingError(message) from None
