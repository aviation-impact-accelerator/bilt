from __future__ import annotations

__all__ = ["DomainBindingError", "DomainRegistry", "DomainShapeError"]

import collections.abc
import dataclasses


class DomainBindingError(Exception):
    """Raised when a dimension binding is missing or inconsistent."""


class DomainShapeError(Exception):
    """Raised when a concrete shape does not match its symbolic domain."""


@dataclasses.dataclass(frozen=True, slots=True)
class _DomainBinding:
    coordinates: tuple[object, ...]

    @property
    def size(self) -> int:
        return len(self.coordinates)


class DomainRegistry:
    """Runtime dimension bindings for one evaluation."""

    __slots__ = ("_bindings",)

    def __init__(self) -> None:
        self._bindings: dict[collections.abc.Hashable, _DomainBinding] = {}

    def bind(
        self,
        dimension: collections.abc.Hashable,
        coordinates: collections.abc.Iterable[object],
    ) -> None:
        binding = _DomainBinding(tuple(coordinates))
        existing = self._bindings.get(dimension)

        if existing is None:
            self._bindings[dimension] = binding
            return

        if existing != binding:
            message = (
                f"Dimension {dimension!r} is already bound to different "
                "coordinates."
            )
            raise DomainBindingError(message)

    def coordinates(
        self,
        dimension: collections.abc.Hashable,
    ) -> tuple[object, ...]:
        return self._binding(dimension).coordinates

    def size(self, dimension: collections.abc.Hashable) -> int:
        return self._binding(dimension).size

    def validate_shape(
        self,
        domain: object,
        shape: tuple[int, ...],
    ) -> None:
        dimensions = tuple(getattr(domain, "dims", domain))

        if len(dimensions) != len(shape):
            message = (
                f"Domain rank {len(dimensions)} does not match shape "
                f"{shape!r}."
            )
            raise DomainShapeError(message)

        for axis, (dimension, actual) in enumerate(
            zip(dimensions, shape, strict=True)
        ):
            expected = self.size(dimension)

            if actual != expected:
                message = (
                    f"Axis {axis} for dimension {dimension!r} has size "
                    f"{actual}; expected {expected}."
                )
                raise DomainShapeError(message)

    def _binding(
        self,
        dimension: collections.abc.Hashable,
    ) -> _DomainBinding:
        try:
            return self._bindings[dimension]
        except KeyError:
            message = f"Dimension {dimension!r} has not been bound."
            raise DomainBindingError(message) from None
