"""The public `Quantity` protocol: what a value with a unit can do, whatever its shape."""

import typing

if typing.TYPE_CHECKING:
    from bilt.core._tensors._protocols import Number
    from bilt.units._units import Unit


@typing.runtime_checkable
class Quantity(typing.Protocol):
    """A value with a unit, of any shape: a scalar quantity or a tensor of quantities.

    Annotate a port `typing.Annotated[Quantity, DIM]` to accept either. Create one by
    multiplying a number by a unit, as in `2.0 * kilogram`, or with
    `tensor(..., unit=...)`.
    """

    @property
    def unit(self) -> Unit:
        """The unit the value is expressed in."""
        ...

    @property
    def value(self) -> Number:
        """The unitless magnitude: a number for a scalar, a tensor for a tensor."""
        ...

    def convert_to(self, unit: Unit, /) -> typing.Self:
        """Return the same quantity expressed in `unit`."""
        ...

    def cast_to(self, unit: Unit, /) -> typing.Self:
        """Relabel the same numbers with an equivalent `unit`, without converting them."""
        ...

    def strip_metadata(self) -> typing.Self:
        """Return the same quantity with every metadata tag removed from its unit."""
        ...

    def __add__(self, other: typing.Self, /) -> typing.Self: ...

    def __sub__(self, other: typing.Self, /) -> typing.Self: ...

    def __mul__(self, other: typing.Self | float, /) -> typing.Self: ...

    def __rmul__(self, other: float, /) -> typing.Self: ...

    def __truediv__(self, other: typing.Self | float, /) -> typing.Self: ...

    def __rtruediv__(self, other: float, /) -> typing.Self: ...

    def __pow__(self, other: float | Quantity, /) -> typing.Self: ...

    def __pos__(self) -> typing.Self: ...

    def __neg__(self) -> typing.Self: ...

    def __abs__(self) -> typing.Self: ...

    def __lt__(self, other: typing.Self, /) -> bool: ...

    def __le__(self, other: typing.Self, /) -> bool: ...

    def __gt__(self, other: typing.Self, /) -> bool: ...

    def __ge__(self, other: typing.Self, /) -> bool: ...
