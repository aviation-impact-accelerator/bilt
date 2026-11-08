"""Shared, representation-independent typing for quantities.

This intentionally does not import Tensor (or Unit) at runtime. A quantity
can carry a scalar or tensor magnitude, but the two concrete representations
do not need to inherit from a common runtime implementation.
"""

import typing

if typing.TYPE_CHECKING:
    from bilt.core._tensors._tensors import Tensor
    from bilt.symbolic._dimensions import Coordinate, Dim
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


class TensorQuantityLike(QuantityLike["Tensor"], typing.Protocol):
    """Tensor-specific operations also exposed by concrete QuantityTensor.

    These operations are symbolic when executed inside a Bilt equation. Their
    declared result is Tensor, matching today's concrete Tensor API; Bilt's
    symbolic analysis determines which axes survive a reduction.
    """

    def sum(self, dim: Dim) -> Tensor:
        """Reduce over a named axis in an equation."""
        ...

    def mean(self, dim: Dim) -> Tensor:
        """Take a mean over a named axis in an equation."""
        ...

    def min(self, dim: Dim) -> Tensor:
        """Take a minimum over a named axis in an equation."""
        ...

    def max(self, dim: Dim) -> Tensor:
        """Take a maximum over a named axis in an equation."""
        ...

    def prod(self, dim: Dim) -> Tensor:
        """Multiply values along a named axis in an equation."""
        ...

    def isel(self, dim: Dim, index: int) -> Tensor:
        """Select an integer position along an axis in an equation."""
        ...

    def sel(self, dim: Dim, coordinate: Coordinate) -> Tensor:
        """Select a coordinate along an axis in an equation."""
        ...
