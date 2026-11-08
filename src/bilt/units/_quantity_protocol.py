"""Shared, representation-independent typing for quantities.

This intentionally does not import Tensor (or Unit) at runtime. A quantity
can carry a scalar or tensor magnitude, but the two concrete representations
do not need to inherit from a common runtime implementation.
"""

import typing

if typing.TYPE_CHECKING:
    from bilt.units._units import Unit


class QuantityLike[M = object](typing.Protocol):
    """Read-only quantity contract, covariant in the magnitude type.

    The default is `object` for now because it accepts both scalar and
    tensor magnitudes without importing Tensor into the units layer.
    A production tensor quantity and richer arithmetic protocol can follow
    independently.
    """

    @property
    def value(self) -> M:
        """Return the unlabelled scalar or tensor magnitude."""
        ...

    @property
    def unit(self) -> Unit:
        """Return the non-optional physical unit."""
        ...


class ScalarQuantityLike(QuantityLike[float], typing.Protocol):
    """Scalar quantity operations supported by native Quantity.

    In an `@equation` this describes *scalar-local* arithmetic: Bilt
    can still lift the resulting symbolic relation over tensor domains.
    This is separate from the more general QuantityLike contract because
    tensor values don't implement concrete tensor arithmetic outside Bilt.
    """

    def __add__(self, other: typing.Self, /) -> typing.Self:
        """Add quantities with Bilt's dimensional checks."""
        ...

    def __sub__(self, other: typing.Self, /) -> typing.Self:
        """Subtract quantities."""
        ...

    def __mul__(self, other: float, /) -> typing.Self:
        """Scale by a plain number."""
        ...

    def __rmul__(self, other: float, /) -> typing.Self:
        """Multiply by a plain number from the left."""
        ...

    def __truediv__(self, other: float, /) -> typing.Self:
        """Divide by a plain number."""
        ...

    def __rtruediv__(self, other: float, /) -> typing.Self:
        """Divide a plain number by this quantity."""
        ...

    def __neg__(self) -> typing.Self:
        """Negate the quantity."""
        ...
