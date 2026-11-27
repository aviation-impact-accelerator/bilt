__all__ = [
    "OverArg",
    "QuantityTensor",
    "Tensor",
    "tensor",
]

import collections.abc
import dataclasses
import enum
import typing

import numpy as np

from bilt.core._tensors._errors import BiltTensorDomainError
from bilt.symbolic._dimensions import Coordinate, CoordinateAxis, Dim
from bilt.symbolic._domains import Domain
from bilt.units._category import UnitCategory
from bilt.units._conversion import conversion_between
from bilt.units._errors import BiltUnitError
from bilt.units._units import Unit

OverArg = Dim | collections.abc.Iterable[Dim]


@dataclasses.dataclass(frozen=True, slots=True)
class Tensor:
    """Concrete values over a Bilt domain."""

    values: np.ndarray
    domain: Domain
    unit: Unit | None = None
    axes: tuple[CoordinateAxis, ...] = dataclasses.field(default=(), kw_only=True)

    def __post_init__(self) -> None:
        values = _enum_preserving_array(self.values)
        if self.unit is not None and _has_enum_members(values):
            msg = "Categorical tensors cannot carry physical units."
            raise BiltTensorDomainError(msg)
        object.__setattr__(self, "values", values)

        _check_rank(values, self.domain)
        if len({axis.dimension for axis in self.axes}) != len(self.axes):
            msg = "A tensor cannot have multiple coordinate axes for one dimension."
            raise BiltTensorDomainError(msg)
        for axis in self.axes:
            if (
                axis.dimension not in self.domain.dims
                or len(axis) != values.shape[self.domain.dims.index(axis.dimension)]
            ):
                msg = f"Coordinate axis {axis.dimension!s} does not match the tensor domain and shape."
                raise BiltTensorDomainError(msg)

    @classmethod
    def scalar(cls, value: object, unit: Unit | None = None) -> Tensor:
        values = _enum_preserving_array(value)

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

    @typing.overload
    def __mul__(self, other: Unit) -> QuantityTensor: ...

    @typing.overload
    def __mul__(self, other: object) -> Tensor: ...

    def __mul__(self, other: object) -> Tensor:
        if isinstance(other, Unit):
            return self._labelled(other if self.unit is None else self.unit * other)
        raise _arithmetic_not_implemented()

    @typing.overload
    def __rmul__(self, other: Unit) -> QuantityTensor: ...

    @typing.overload
    def __rmul__(self, other: object) -> Tensor: ...

    def __rmul__(self, other: object) -> Tensor:
        if isinstance(other, Unit):
            if self.unit is None:
                return NotImplemented
            return self._labelled(other * self.unit)
        raise _arithmetic_not_implemented()

    @typing.overload
    def __truediv__(self, other: Unit) -> QuantityTensor: ...

    @typing.overload
    def __truediv__(self, other: object) -> Tensor: ...

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

    def convert_to(self, unit: Unit) -> typing.Self:
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
        if self.unit.category is UnitCategory.ABSOLUTE and self.unit is unit:
            return self

        conversion = conversion_between(self.unit, unit)

        return dataclasses.replace(self, values=self.values * conversion.scale + conversion.offset, unit=unit)

    def _labelled(self, unit: Unit) -> QuantityTensor:
        return QuantityTensor(values=self.values, domain=self.domain, unit=unit, axes=self.axes)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class QuantityTensor(Tensor):
    """Unit-bearing tensor with a non-optional unit and a quantity magnitude.

    It retains the Tensor API, including symbolic operations, domains,
    conversion and unit composition. Its `value` is a unitless Tensor
    retaining the original domain.
    """

    unit: Unit

    def __post_init__(self) -> None:
        Tensor.__post_init__(self)
        if not isinstance(self.unit, Unit):
            message = "QuantityTensor requires a physical unit"
            raise BiltUnitError(message)

    @property
    def value(self) -> Tensor:
        """The unitless magnitude, retaining the original tensor domain."""
        return Tensor(values=self.values, domain=self.domain, axes=self.axes)


@typing.overload
def tensor(
    values: object, *, over: OverArg | None = None, domain: Domain | None = None, unit: Unit
) -> QuantityTensor: ...


@typing.overload
def tensor(
    values: object, *, over: OverArg | None = None, domain: Domain | None = None, unit: None = None
) -> Tensor: ...


@typing.overload
def tensor(
    values: object, *, over: OverArg | None = None, domain: Domain | None = None, unit: Unit | None
) -> Tensor: ...


def tensor(
    values: object,
    *,
    over: OverArg | None = None,
    domain: Domain | None = None,
    unit: Unit | None = None,
) -> Tensor:
    """Create a Bilt tensor from concrete user data.

    Scalars may omit a domain. Non-scalar values must provide either
    ``over`` or ``domain``. A CoordinateAxis supplies both values and the
    domain; its coordinates are retained as metadata. A ``unit`` labels
    every value in the tensor (coordinates remain in the supplied scale).
    """
    if over is not None and domain is not None:
        message = "Pass either over= or domain=, not both."
        raise BiltTensorDomainError(message)

    if isinstance(values, CoordinateAxis):
        if over is not None or domain is not None:
            message = "A coordinate axis already specifies the tensor domain."
            raise BiltTensorDomainError(message)
        axis = values
        array = _enum_preserving_array(axis.coordinates)
        domain = Domain((axis.dimension,))
        return (
            QuantityTensor(values=array, domain=domain, unit=unit, axes=(axis,))
            if unit is not None
            else Tensor(values=array, domain=domain, axes=(axis,))
        )

    array = _enum_preserving_array(values)

    match over, domain:
        case None, Domain():
            pass
        case over, None if over is not None:
            domain = _domain_from_over(over)
        case None, None if array.ndim == 0:
            return (
                QuantityTensor(values=array, domain=Domain.scalar(), unit=unit)
                if unit is not None
                else Tensor.scalar(array)
            )
        case None, None:
            message = "Non-scalar values need an explicit domain"
            raise BiltTensorDomainError(message)
        case _:
            message = "Pass either over= or domain=, not both."
            raise BiltTensorDomainError(message)

    return (
        QuantityTensor(values=array, domain=domain, unit=unit)
        if unit is not None
        else Tensor(values=array, domain=domain)
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


def _has_enum_members(values: np.ndarray) -> bool:
    return values.dtype == object and any(isinstance(value, enum.Enum) for value in values.flat)


def _enum_preserving_array(values: object) -> np.ndarray:
    """Do not coerce IntEnum and StrEnum members to numeric/string NumPy dtypes."""
    if isinstance(values, np.ndarray) and values.dtype != object:
        return values
    objects = np.asarray(values, dtype=object)
    if not _has_enum_members(objects):
        return np.asarray(values)
    members = list(objects.flat)
    enum_type = type(members[0])
    if not isinstance(members[0], enum.Enum) or any(type(member) is not enum_type for member in members):
        msg = "Categorical tensor values must all belong to the same enum class."
        raise BiltTensorDomainError(msg)
    return objects
