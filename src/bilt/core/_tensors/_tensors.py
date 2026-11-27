__all__ = [
    "OverArg",
    "tensor",
]

import collections.abc
import dataclasses
import typing

import numpy as np

from bilt.core._tensors._errors import BiltTensorDomainError
from bilt.symbolic._dimensions import Coordinate, Dim
from bilt.symbolic._domains import Domain
from bilt.units._category import UnitCategory
from bilt.units._conversion import conversion_between
from bilt.units._errors import BiltUnitError
from bilt.units._units import Unit
from bilt.units._validation import validate_cast

if typing.TYPE_CHECKING:
    from bilt.core._tensors._protocols import QuantityTensor, Tensor
    from bilt.units._protocols import Quantity

OverArg = Dim | collections.abc.Iterable[Dim]


@dataclasses.dataclass(frozen=True, slots=True, repr=False)
class _TensorBase:
    """Concrete values over a Bilt domain."""

    values: np.ndarray
    domain: Domain
    unit: Unit | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        object.__setattr__(self, "values", values)

        _check_rank(values, self.domain)

    @classmethod
    def scalar(cls, value: object, unit: Unit | None = None) -> _TensorBase:
        values = np.asarray(value)

        if values.ndim != 0:
            message = f"Cannot create scalar Tensor from rank-{values.ndim} value."
            raise BiltTensorDomainError(message)

        return cls(
            values=values,
            domain=Domain.scalar(),
            unit=unit,
        )

    def __repr__(self) -> str:
        # Named after the public annotation the value satisfies; the class itself is private.
        if self.unit is None:
            return f"Tensor(values={self.values!r}, domain={self.domain!r})"
        return f"QuantityTensor(values={self.values!r}, domain={self.domain!r}, unit={self.unit!r})"

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    @property
    def ndim(self) -> int:
        return self.values.ndim

    def __float__(self) -> float:
        """The value of a tensor with no dimensions; any other tensor raises `TypeError`, as in NumPy."""
        return float(self.values)

    # Arithmetic is symbolic: these methods exist to type equation bodies and raise outside a model.
    # A unitless tensor stays unitless unless a quantity or a unit joins it.

    @typing.overload
    def __add__(self, other: Quantity) -> _QuantityTensor: ...

    @typing.overload
    def __add__(self, other: object) -> _TensorBase: ...

    def __add__(self, other: object) -> _TensorBase:
        raise _arithmetic_not_implemented()

    @typing.overload
    def __radd__(self, other: Quantity) -> _QuantityTensor: ...

    @typing.overload
    def __radd__(self, other: object) -> _TensorBase: ...

    def __radd__(self, other: object) -> _TensorBase:
        raise _arithmetic_not_implemented()

    @typing.overload
    def __sub__(self, other: Quantity) -> _QuantityTensor: ...

    @typing.overload
    def __sub__(self, other: object) -> _TensorBase: ...

    def __sub__(self, other: object) -> _TensorBase:
        raise _arithmetic_not_implemented()

    @typing.overload
    def __rsub__(self, other: Quantity) -> _QuantityTensor: ...

    @typing.overload
    def __rsub__(self, other: object) -> _TensorBase: ...

    def __rsub__(self, other: object) -> _TensorBase:
        raise _arithmetic_not_implemented()

    @typing.overload
    def __mul__(self, other: Unit | Quantity) -> _QuantityTensor: ...

    @typing.overload
    def __mul__(self, other: object) -> _TensorBase: ...

    def __mul__(self, other: object) -> _TensorBase:
        if isinstance(other, Unit):
            return self._labelled(other if self.unit is None else self.unit * other)
        raise _arithmetic_not_implemented()

    @typing.overload
    def __rmul__(self, other: Unit | Quantity) -> _QuantityTensor: ...

    @typing.overload
    def __rmul__(self, other: object) -> _TensorBase: ...

    def __rmul__(self, other: object) -> _TensorBase:
        if isinstance(other, Unit):
            if self.unit is None:
                return NotImplemented
            return self._labelled(other * self.unit)
        raise _arithmetic_not_implemented()

    @typing.overload
    def __truediv__(self, other: Unit | Quantity) -> _QuantityTensor: ...

    @typing.overload
    def __truediv__(self, other: object) -> _TensorBase: ...

    def __truediv__(self, other: object) -> _TensorBase:
        if isinstance(other, Unit):
            if self.unit is None:
                return NotImplemented
            return self._labelled(self.unit / other)
        raise _arithmetic_not_implemented()

    @typing.overload
    def __rtruediv__(self, other: Quantity) -> _QuantityTensor: ...

    @typing.overload
    def __rtruediv__(self, other: object) -> _TensorBase: ...

    def __rtruediv__(self, other: object) -> _TensorBase:
        raise _arithmetic_not_implemented()

    def __pow__(self, other: float | Tensor | Quantity) -> typing.Self:
        raise _arithmetic_not_implemented()

    def __rpow__(self, other: float | Quantity) -> typing.Self:
        raise _arithmetic_not_implemented()

    def __pos__(self) -> typing.Self:
        raise _arithmetic_not_implemented()

    def __neg__(self) -> typing.Self:
        raise _arithmetic_not_implemented()

    def __abs__(self) -> typing.Self:
        raise _arithmetic_not_implemented()

    def __lt__(self, other: object) -> bool:
        raise _symbolic_operation_only()

    def __le__(self, other: object) -> bool:
        raise _symbolic_operation_only()

    def __gt__(self, other: object) -> bool:
        raise _symbolic_operation_only()

    def __ge__(self, other: object) -> bool:
        raise _symbolic_operation_only()

    def mean(self, dim: Dim) -> typing.Self:  # noqa: ARG002
        """Return the mean over a named dimension in a Bilt model."""
        raise _symbolic_operation_only()

    def sum(self, dim: Dim) -> typing.Self:  # noqa: ARG002
        """Return the sum over a named dimension in a Bilt model."""
        raise _symbolic_operation_only()

    def min(self, dim: Dim) -> typing.Self:  # noqa: ARG002
        """Return the minimum over a named dimension in a Bilt model."""
        raise _symbolic_operation_only()

    def max(self, dim: Dim) -> typing.Self:  # noqa: ARG002
        """Return the maximum over a named dimension in a Bilt model."""
        raise _symbolic_operation_only()

    def prod(self, dim: Dim) -> typing.Self:  # noqa: ARG002
        """Return the product over a named dimension in a Bilt model."""
        raise _symbolic_operation_only()

    def isel(self, dim: Dim, index: int) -> typing.Self:  # noqa: ARG002
        """Select a position along a named dimension in a Bilt model."""
        raise _symbolic_operation_only()

    def sel(self, dim: Dim, coordinate: Coordinate) -> typing.Self:  # noqa: ARG002
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

    def _labelled(self, unit: Unit) -> _QuantityTensor:
        return _QuantityTensor(values=self.values, domain=self.domain, unit=unit)


@dataclasses.dataclass(frozen=True, slots=True, repr=False)
class _Tensor(_TensorBase):
    """Concrete unitless tensor."""

    unit: None = None

    @property
    def _unitless_tensor_marker(self) -> typing.Literal[True]:
        return True


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class _QuantityTensor(_TensorBase):
    """Unit-bearing tensor with a non-optional unit and a quantity magnitude.

    It retains the _TensorBase API, including symbolic operations, domains,
    conversion and unit composition. Its `value` is a unitless _TensorBase
    retaining the original domain.
    """

    unit: Unit

    def __post_init__(self) -> None:
        _TensorBase.__post_init__(self)
        if not isinstance(self.unit, Unit):
            message = "QuantityTensor requires a physical unit"
            raise BiltUnitError(message)

    # Scaling by a number or a unitless tensor keeps the unit; the base class already types
    # addition and subtraction with quantities as returning a quantity tensor.

    @typing.overload
    def __mul__(self, other: Unit | Tensor | Quantity | float) -> _QuantityTensor: ...

    @typing.overload
    def __mul__(self, other: object) -> _TensorBase: ...

    def __mul__(self, other: object) -> _TensorBase:
        return _TensorBase.__mul__(self, other)

    @typing.overload
    def __rmul__(self, other: Unit | Tensor | Quantity | float) -> _QuantityTensor: ...

    @typing.overload
    def __rmul__(self, other: object) -> _TensorBase: ...

    def __rmul__(self, other: object) -> _TensorBase:
        return _TensorBase.__rmul__(self, other)

    @typing.overload
    def __truediv__(self, other: Unit | Tensor | Quantity | float) -> _QuantityTensor: ...

    @typing.overload
    def __truediv__(self, other: object) -> _TensorBase: ...

    def __truediv__(self, other: object) -> _TensorBase:
        return _TensorBase.__truediv__(self, other)

    @typing.overload
    def __rtruediv__(self, other: Tensor | Quantity | float) -> _QuantityTensor: ...

    @typing.overload
    def __rtruediv__(self, other: object) -> _TensorBase: ...

    def __rtruediv__(self, other: object) -> _TensorBase:
        return _TensorBase.__rtruediv__(self, other)

    def cast_to(self, unit: Unit) -> typing.Self:
        """Relabel these values with an equivalent `unit`, without converting them."""
        validate_cast(self, unit)
        return dataclasses.replace(self, unit=unit)

    def strip_metadata(self) -> typing.Self:
        """Return these values with every metadata tag removed from their unit."""
        return dataclasses.replace(self, unit=self.unit.strip_metadata())

    @property
    def value(self) -> _Tensor:
        """The unitless magnitude, retaining the original tensor domain."""
        return _Tensor(values=self.values, domain=self.domain)


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
) -> Tensor | QuantityTensor: ...


def tensor(
    values: object,
    *,
    over: OverArg | None = None,
    domain: Domain | None = None,
    unit: Unit | None = None,
) -> Tensor | QuantityTensor:
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
            return (
                _QuantityTensor(values=array, domain=Domain.scalar(), unit=unit)
                if unit is not None
                else _Tensor(values=array, domain=Domain.scalar())
            )
        case None, None:
            message = "Non-scalar values need an explicit domain"
            raise BiltTensorDomainError(message)
        case _:
            message = "Pass either over= or domain=, not both."
            raise BiltTensorDomainError(message)

    return (
        _QuantityTensor(values=array, domain=domain, unit=unit)
        if unit is not None
        else _Tensor(values=array, domain=domain)
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
