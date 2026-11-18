"""The public `Tensor`, `QuantityTensor` and `Number` annotations.

Together with `Quantity` they answer two independent questions about a port:

|          | Any shape  | Must be a tensor |
|----------|------------|------------------|
| Units    | `Quantity` | `QuantityTensor` |
| No units | `Number`   | `Tensor`         |
"""

__all__ = ["Number", "QuantityTensor", "Tensor"]

import typing

from bilt.units._protocols import Quantity

if typing.TYPE_CHECKING:
    import numpy as np

    from bilt.symbolic._dimensions import Coordinate, Dim
    from bilt.symbolic._domains import Domain


class _Tensorial(typing.Protocol):
    """What every tensor has, with or without a unit: a domain to reduce and select along."""

    @property
    def values(self) -> np.ndarray:
        """The values, with one array axis per dimension of the domain."""
        ...

    @property
    def domain(self) -> Domain:
        """The named dimensions the values vary over."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """The length of each dimension of the domain."""
        ...

    @property
    def ndim(self) -> int:
        """The number of dimensions of the domain."""
        ...

    def __float__(self) -> float:
        """The value of a tensor with no dimensions; any other tensor raises `TypeError`."""
        ...

    def sum(self, dim: Dim, /) -> typing.Self:
        """The sum over a named dimension."""
        ...

    def mean(self, dim: Dim, /) -> typing.Self:
        """The mean over a named dimension."""
        ...

    def min(self, dim: Dim, /) -> typing.Self:
        """The minimum over a named dimension."""
        ...

    def max(self, dim: Dim, /) -> typing.Self:
        """The maximum over a named dimension."""
        ...

    def prod(self, dim: Dim, /) -> typing.Self:
        """The product over a named dimension."""
        ...

    def isel(self, dim: Dim, index: int, /) -> typing.Self:
        """The values at a position along a named dimension."""
        ...

    def sel(self, dim: Dim, coordinate: Coordinate, /) -> typing.Self:
        """The values at a coordinate along a named dimension."""
        ...


@typing.runtime_checkable
class Tensor(_Tensorial, typing.Protocol):
    """Unitless values over named dimensions, which an equation can reduce or select along.

    Create one with `tensor(...)`. A tensor with a unit is a `QuantityTensor` instead.
    """

    def __add__(self, other: typing.Self | float, /) -> typing.Self: ...

    def __radd__(self, other: float, /) -> typing.Self: ...

    def __sub__(self, other: typing.Self | float, /) -> typing.Self: ...

    def __rsub__(self, other: float, /) -> typing.Self: ...

    @typing.overload
    def __mul__(self, other: Quantity, /) -> QuantityTensor: ...

    @typing.overload
    def __mul__(self, other: typing.Self | float, /) -> typing.Self: ...

    @typing.overload
    def __rmul__(self, other: Quantity, /) -> QuantityTensor: ...

    @typing.overload
    def __rmul__(self, other: float, /) -> typing.Self: ...

    @typing.overload
    def __truediv__(self, other: Quantity, /) -> QuantityTensor: ...

    @typing.overload
    def __truediv__(self, other: typing.Self | float, /) -> typing.Self: ...

    @typing.overload
    def __rtruediv__(self, other: Quantity, /) -> QuantityTensor: ...

    @typing.overload
    def __rtruediv__(self, other: float, /) -> typing.Self: ...

    def __pow__(self, other: float, /) -> typing.Self: ...

    def __pos__(self) -> typing.Self: ...

    def __neg__(self) -> typing.Self: ...

    def __abs__(self) -> typing.Self: ...

    def __lt__(self, other: typing.Self | float, /) -> bool: ...

    def __le__(self, other: typing.Self | float, /) -> bool: ...

    def __gt__(self, other: typing.Self | float, /) -> bool: ...

    def __ge__(self, other: typing.Self | float, /) -> bool: ...


@typing.runtime_checkable
class QuantityTensor(Quantity, _Tensorial, typing.Protocol):
    """A tensor of quantities: values with a unit over named dimensions.

    Create one with `tensor(..., unit=...)`, or by multiplying a tensor by a unit.
    """

    @property
    def value(self) -> Tensor:
        """The unitless magnitude, over the same domain."""
        ...

    # A scalar quantity or a unitless tensor combines with a quantity tensor to give a quantity tensor.

    def __add__(self, other: Quantity, /) -> typing.Self: ...

    def __radd__(self, other: Quantity, /) -> typing.Self: ...

    def __sub__(self, other: Quantity, /) -> typing.Self: ...

    def __rsub__(self, other: Quantity, /) -> typing.Self: ...

    def __mul__(self, other: Quantity | Tensor | float, /) -> typing.Self: ...

    def __rmul__(self, other: Quantity | Tensor | float, /) -> typing.Self: ...

    def __truediv__(self, other: Quantity | Tensor | float, /) -> typing.Self: ...

    def __rtruediv__(self, other: Quantity | Tensor | float, /) -> typing.Self: ...


Number = float | Tensor
"""A unitless value of any shape: a number or a unitless tensor."""
