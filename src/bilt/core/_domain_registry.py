__all__ = [
    "DomainRegistry",
]

import math
import typing

import numpy as np

from bilt.core._errors import BiltDomainBindingError, BiltDomainRegistryLockedError
from bilt.symbolic._atoms import FloatType, IntType, QuantityType, SymbolType
from bilt.symbolic._dimensions import CoordinateAxis
from bilt.units import Quantity
from bilt.units._category import UnitCategory
from bilt.units._conversion import conversion_between
from bilt.units._units import BASE_UNIT_SYSTEM

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.symbolic._dimensions import Coordinate, Dim
    from bilt.symbolic._domains import Domain
    from bilt.symbolic._tree import Tree
    from bilt.units import Unit


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
        self._derived_dimensions: set[Dim] = set()

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
            raise BiltDomainBindingError(message)

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
                raise BiltDomainRegistryLockedError

            self._bindings[axis.dimension] = axis
            return

        if isinstance(existing, int):
            if existing != len(axis):
                message = f"Dimension {axis.dimension!s} has inconsistent sizes: {existing} and {len(axis)}."
                raise BiltDomainBindingError(message)

            if self._locked_against_new_dimensions:
                raise BiltDomainRegistryLockedError

            self._bindings[axis.dimension] = axis
            return

        if existing.coordinates != axis.coordinates:
            message = f"Dimension {axis.dimension!s} has conflicting coordinate definitions."
            raise BiltDomainBindingError(message)

    def clone(self) -> DomainRegistry:
        """Copy registry state for an alternative structural route."""
        cloned = DomainRegistry()
        cloned._bindings = self._bindings.copy()
        cloned._locked_against_new_dimensions = self._locked_against_new_dimensions
        cloned._derived_dimensions = self._derived_dimensions.copy()
        return cloned

    def dimension_order(self) -> Domain:
        """Read currently bound dimensions without freezing the registry."""
        from bilt.symbolic._domains import Domain  # noqa: PLC0415

        return Domain(tuple(self._bindings))

    @property
    def has_derived_dimensions(self) -> bool:
        """Whether this evaluation has derived new axis bindings."""
        return bool(self._derived_dimensions)

    def bind_concat(self, sources: tuple[Dim, ...], target: Dim) -> None:
        """Derive the extent and coordinate labels of a concatenated dimension."""
        if target in sources:
            message = f"Concatenation output {target} must differ from its source dimensions."
            raise BiltDomainBindingError(message)
        bindings = tuple(self._binding(source) for source in sources)
        coordinates = tuple(binding for binding in bindings if isinstance(binding, CoordinateAxis))
        if coordinates and len(coordinates) != len(bindings):
            message = "Cannot concatenate a mix of labelled and unlabelled source axes."
            raise BiltDomainBindingError(message)
        if coordinates:
            all_coordinates = tuple(label for axis in coordinates for label in axis.coordinates)
            self.bind_axis(CoordinateAxis(target, all_coordinates))
        else:
            self._bind_size(target, sum(binding for binding in bindings if isinstance(binding, int)))
        self._derived_dimensions.add(target)

    def bind_slice(self, source: Dim, target: Dim, selector: slice) -> None:
        """Derive extent and optional coordinates from a Python positional slice."""
        if source == target:
            message = "The destination slice dimension must differ from the source."
            raise BiltDomainBindingError(message)
        binding = self._binding(source)
        size = binding if isinstance(binding, int) else len(binding)
        # Python's slice.indices matches NumPy's handling of omitted,
        # negative, and out-of-range positional bounds.
        start, stop, step = selector.indices(size)
        selected_length = len(range(start, stop, step))
        if isinstance(binding, int):
            self._bind_size(target, selected_length)
        else:
            selected = binding.coordinates[selector]
            self.bind_axis(CoordinateAxis(target, selected))
        self._derived_dimensions.add(target)

    def coordinate_slice(self, dimension: Dim, selector: slice) -> slice:
        """Resolve inclusive coordinate labels into a positional Python slice.

        Labels are looked up in axis order, including for reverse slices.
        Endpoint labels must exist rather than being clipped.
        """
        self.axis(dimension)
        stride = 1 if selector.step is None else selector.step
        if type(stride) is not int or stride == 0:
            message = "Coordinate slice step must be a non-zero integer."
            raise BiltDomainBindingError(message)
        start = None if selector.start is None else self.position(dimension, selector.start)
        stop_position = None if selector.stop is None else self.position(dimension, selector.stop)
        if stop_position is None:
            stop = None
        elif stride > 0:
            stop = stop_position + 1
        else:
            # In a reverse slice -1 indexes from the end, not before zero.
            stop = None if stop_position == 0 else stop_position - 1
        return slice(start, stop, selector.step)

    def bind_sel_slice(self, source: Dim, target: Dim, selector: slice) -> None:
        """Register the extent and coordinate labels of an inclusive slice."""
        self.bind_slice(source, target, self.coordinate_slice(source, selector))

    def bind_arange(  # noqa: PLR0913
        self,
        dimension: Dim,
        start: Tree,
        stop: Tree,
        *,
        values: collections.abc.Mapping[str, object],
        step: int,
        inclusive: bool,
        unit: Unit | None,
    ) -> tuple[int, int, int, float, float]:
        """Bind a labelled axis from direct scalar boundaries before compilation.

        The returned numeric parameters are fixed for this evaluation. The
        backend must use them instead of recreating a dynamic-sized result.
        """
        start_value, stop_value, resolved_step, scale, offset = self.resolve_arange(
            dimension, start, stop, values=values, step=step, inclusive=inclusive, unit=unit
        )
        self.bind_axis(CoordinateAxis(dimension, tuple(range(start_value, stop_value, resolved_step))))
        self._derived_dimensions.add(dimension)
        return start_value, stop_value, resolved_step, scale, offset

    def resolve_arange(  # noqa: PLR0913
        self,
        dimension: Dim,
        start: Tree,
        stop: Tree,
        *,
        values: collections.abc.Mapping[str, object],
        step: int,
        inclusive: bool,
        unit: Unit | None,
    ) -> tuple[int, int, int, float, float]:
        """Compute static range parameters without rebinding the dimension."""
        first = self._range_boundary(start, values=values, unit=unit)
        last = self._range_boundary(stop, values=values, unit=unit)
        exclusive = last + (1 if step > 0 else -1) if inclusive else last
        limit = 100_000
        if len(range(first, exclusive, step)) > limit:
            message = f"Generated dimension {dimension} exceeds the {limit} element safety limit."
            raise BiltDomainBindingError(message)

        if unit is None:
            return first, exclusive, step, 1.0, 0.0
        coherent = BASE_UNIT_SYSTEM.coherent_unit(unit.dimension)
        if unit.category is UnitCategory.ABSOLUTE:
            coherent = abs(coherent)
        conversion = conversion_between(unit, coherent)
        return first, exclusive, step, conversion.scale, conversion.offset

    @staticmethod
    def _range_boundary(
        tree: Tree,
        *,
        values: collections.abc.Mapping[str, object],
        unit: Unit | None,
    ) -> int:
        """Accept only literal constants or directly supplied scalar inputs."""
        if not tree.is_leaf:
            message = (
                "Tensor.arange boundaries must be literals or direct scalar inputs; derived inputs are not supported."
            )
            raise BiltDomainBindingError(message)
        if tree.head.atom_type is SymbolType:
            name = typing.cast("str", tree.head.value)
            if name not in values:
                message = (
                    f"Tensor.arange boundary {name!r} requires a directly supplied scalar input. "
                    "Intermediate scalar results cannot determine dimension extents yet."
                )
                raise BiltDomainBindingError(message)
            value = values[name]
        elif tree.head.atom_type in (IntType, FloatType):
            value = tree.head.value
        elif tree.head.atom_type is QuantityType:
            message = "Tensor.arange does not currently support quantity-valued literal boundary expressions."
            raise BiltDomainBindingError(message)
        else:
            message = "Tensor.arange boundary must be a literal or directly supplied scalar input."
            raise BiltDomainBindingError(message)

        if isinstance(value, Quantity):
            if unit is None:
                message = "Tensor.arange needs a unit when its boundaries carry physical units."
                raise BiltDomainBindingError(message)
            value = value.convert_to(unit).value
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            message = "Tensor.arange requires finite integer-valued scalar boundaries."
            raise BiltDomainBindingError(message)
        if not float(value).is_integer():
            message = f"Tensor.arange boundary {value!r} is not an integer."
            raise BiltDomainBindingError(message)
        return int(value)

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
            raise BiltDomainBindingError(message)

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
        raise BiltDomainBindingError(message)

    def coordinate_signature(
        self,
    ) -> tuple[tuple[Dim, tuple[Coordinate, ...]], ...]:
        """Return the coordinate bindings that affect compiled evaluation."""
        return tuple(
            (dimension, binding.coordinates)
            for dimension, binding in self._bindings.items()
            if isinstance(binding, CoordinateAxis)
        )

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
                raise BiltDomainRegistryLockedError

            self._bindings[dimension] = size
            return

        existing_size = existing if isinstance(existing, int) else len(existing)

        if existing_size != size:
            message = f"Dimension {dimension!s} has inconsistent sizes: {existing_size} and {size}."
            raise BiltDomainBindingError(message)

    def _binding(self, dimension: Dim) -> int | CoordinateAxis:
        try:
            return self._bindings[dimension]
        except KeyError:
            message = f"Dimension {dimension!s} is not bound."
            raise BiltDomainBindingError(message) from None
