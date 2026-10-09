__all__ = [
    "OverArg",
    "Tensor",
    "tensor",
]

import collections.abc
import dataclasses
import typing

import numpy as np

from bilt.core._tensors._errors import BiltTensorDomainError
from bilt.symbolic._dimensions import Coordinate, Dim
from bilt.symbolic._domains import Domain
from bilt.units._errors import BiltUnitError
from bilt.units._quantities import Quantity
from bilt.units._units import Unit

OverArg = Dim | collections.abc.Iterable[Dim]


@dataclasses.dataclass(frozen=True, slots=True)
class Tensor:
    """Concrete values over a Bilt domain."""

    values: np.ndarray
    domain: Domain
    unit: Unit | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        object.__setattr__(self, "values", values)

        _check_rank(values, self.domain)

    @classmethod
    def scalar(cls, value: object, unit: Unit | None = None) -> Tensor:
        values = np.asarray(value)

        if values.ndim != 0:
            message = f"Cannot create scalar Tensor from rank-{values.ndim} value."
            raise BiltTensorDomainError(message)

        return cls(
            values=values,
            domain=Domain.scalar(),
            unit=unit,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    @property
    def ndim(self) -> int:
        return self.values.ndim

    def __add__(self, other: object) -> Tensor:
        raise _arithmetic_not_implemented()

    def __radd__(self, other: object) -> Tensor:
        raise _arithmetic_not_implemented()

    def __sub__(self, other: object) -> Tensor:
        raise _arithmetic_not_implemented()

    def __rsub__(self, other: object) -> Tensor:
        raise _arithmetic_not_implemented()

    def __mul__(self, other: object) -> Tensor:
        if isinstance(other, Unit):
            return self._labelled(other if self.unit is None else self.unit * other)
        raise _arithmetic_not_implemented()

    def __rmul__(self, other: object) -> Tensor:
        if isinstance(other, Unit):
            if self.unit is None:
                return NotImplemented
            return self._labelled(other * self.unit)
        raise _arithmetic_not_implemented()

    def __truediv__(self, other: object) -> Tensor:
        if isinstance(other, Unit):
            if self.unit is None:
                return NotImplemented
            return self._labelled(self.unit / other)
        raise _arithmetic_not_implemented()

    def __rtruediv__(self, other: object) -> Tensor:
        raise _arithmetic_not_implemented()

    def __neg__(self) -> Tensor:
        raise _arithmetic_not_implemented()

    def mean(self, dim: Dim) -> Tensor:  # noqa: ARG002
        """Return the mean over a named dimension in a Bilt model."""
        raise _symbolic_operation_only()

    def sum(self, dim: Dim) -> Tensor:  # noqa: ARG002
        """Return the sum over a named dimension in a Bilt model."""
        raise _symbolic_operation_only()

    def min(self, dim: Dim) -> Tensor:  # noqa: ARG002
        """Return the minimum over a named dimension in a Bilt model."""
        raise _symbolic_operation_only()

    def max(self, dim: Dim) -> Tensor:  # noqa: ARG002
        """Return the maximum over a named dimension in a Bilt model."""
        raise _symbolic_operation_only()

    def prod(self, dim: Dim) -> Tensor:  # noqa: ARG002
        """Return the product over a named dimension in a Bilt model."""
        raise _symbolic_operation_only()

    def isel(self, dim: Dim, index: int) -> Tensor:  # noqa: ARG002
        """Select a position along a named dimension in a Bilt model."""
        raise _symbolic_operation_only()

    def sel(self, dim: Dim, coordinate: Coordinate) -> Tensor:  # noqa: ARG002
        """Select a coordinate along a named dimension in a Bilt model."""
        raise _symbolic_operation_only()

    def convert_to(self, unit: Unit) -> Tensor:
        """Return these values in `unit`, which must measure what this tensor's own unit measures.

        Raises:
            BiltUnitError: This tensor carries no unit, or its unit cannot be converted to `unit`.
        """
        if self.unit is None:
            message = (
                f"Cannot convert a tensor with no unit to {unit!s}. "
                f"Label it with `tensor(..., unit=...)` or by multiplying it by a unit."
            )
            raise BiltUnitError(message)
        converted = Quantity(typing.cast("float", self.values), self.unit).convert_to(unit)

        return dataclasses.replace(self, values=typing.cast("np.ndarray", converted.value), unit=unit)

    def _labelled(self, unit: Unit) -> Tensor:
        return dataclasses.replace(self, unit=unit)


def tensor(
    values: object,
    *,
    over: OverArg | None = None,
    domain: Domain | None = None,
    unit: Unit | None = None,
) -> Tensor:
    """Create a Bilt tensor from concrete user data.

    Scalars may omit a domain. Non-scalar values must provide either
    ``over`` or ``domain``. A ``unit`` labels every value in the tensor.
    """
    if over is not None and domain is not None:
        message = "Pass either over= or domain=, not both."
        raise BiltTensorDomainError(message)

    array = np.asarray(values)

    match over, domain:
        case None, Domain():
            pass
        case over, None if over is not None:
            domain = _domain_from_over(over)
        case None, None if array.ndim == 0:
            return Tensor.scalar(array, unit)
        case None, None:
            message = "Non-scalar values need an explicit domain"
            raise BiltTensorDomainError(message)
        case _:
            message = "Pass either over= or domain=, not both."
            raise BiltTensorDomainError(message)

    return Tensor(
        values=array,
        domain=domain,
        unit=unit,
    )


def _domain_from_over(over: OverArg) -> Domain:
    if isinstance(over, Dim):
        return Domain((over,))

    return Domain(tuple(over))


def _check_rank(values: np.ndarray, domain: Domain) -> None:
    domain_rank = len(domain.dims)

    if values.ndim != domain_rank:
        message = f"Value rank {values.ndim} does not match domain rank {domain_rank} for domain {domain!r}."
        raise BiltTensorDomainError(message)


def _arithmetic_not_implemented() -> NotImplementedError:
    return NotImplementedError(
        "Tensor arithmetic is not implemented. Operations will be dispatched through an active backend."
    )


def _symbolic_operation_only() -> TypeError:
    return TypeError("Tensor operations are symbolic and may only be used within Bilt equations and transforms.")
