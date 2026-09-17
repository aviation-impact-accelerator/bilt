__all__ = [
    "OverArg",
    "Tensor",
    "tensor",
]

import collections.abc
import dataclasses

import numpy as np

from bilt.core._tensors._errors import BiltTensorDomainError
from bilt.symbolic._dimensions import Dim
from bilt.symbolic._domains import Domain

OverArg = Dim | collections.abc.Iterable[Dim]


@dataclasses.dataclass(frozen=True, slots=True)
class Tensor:
    """Concrete values over a Bilt domain."""

    values: np.ndarray
    domain: Domain

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        object.__setattr__(self, "values", values)

        _check_rank(values, self.domain)

    @classmethod
    def scalar(cls, value: object) -> Tensor:
        values = np.asarray(value)

        if values.ndim != 0:
            message = f"Cannot create scalar Tensor from rank-{values.ndim} value."
            raise BiltTensorDomainError(message)

        return cls(
            values=values,
            domain=Domain.scalar(),
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
        raise _arithmetic_not_implemented()

    def __rmul__(self, other: object) -> Tensor:
        raise _arithmetic_not_implemented()

    def __truediv__(self, other: object) -> Tensor:
        raise _arithmetic_not_implemented()

    def __rtruediv__(self, other: object) -> Tensor:
        raise _arithmetic_not_implemented()

    def __neg__(self) -> Tensor:
        raise _arithmetic_not_implemented()


def tensor(
    values: object,
    *,
    over: OverArg | None = None,
    domain: Domain | None = None,
) -> Tensor:
    """Create a Bilt tensor from concrete user data.

    Scalars may omit a domain. Non-scalar values must provide either
    ``over`` or ``domain``.
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
            return Tensor.scalar(array)
        case None, None:
            message = "Non-scalar values need an explicit domain"
            raise BiltTensorDomainError(message)
        case _:
            message = "Pass either over= or domain=, not both."
            raise BiltTensorDomainError(message)

    return Tensor(
        values=array,
        domain=domain,
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
