__all__ = [
    "as_domain",
    "domain",
]

import dataclasses
import itertools

from bilt.symbolic._dimensions import Dim

type DomainLike = Domain | Dim | tuple[Dim, ...] | list[Dim] | None


@dataclasses.dataclass(frozen=True, slots=True)
class Domain:
    """An ordered collection of symbolic dimensions.

    A scalar has an empty domain. Dimensions are compared by string, so
    Dim("i") == Dim("i")
    """

    dims: tuple[Dim, ...] = ()

    def __post_init__(self) -> None:
        if len(set(self.dims)) != len(self.dims):
            message = "A domain cannot contain the same dimension more than once."
            raise ValueError(message)

    @classmethod
    def scalar(cls) -> Domain:
        return cls()

    @property
    def is_scalar(self) -> bool:
        return not self.dims

    def __str__(self) -> str:
        if self.is_scalar:
            return "scalar"

        return "[" + ", ".join(str(dim) for dim in self.dims) + "]"


def domain(*dims: Dim) -> Domain:
    return Domain(dims)


def as_domain(value: DomainLike) -> Domain:
    if value is None:
        return Domain.scalar()

    if isinstance(value, Domain):
        return value

    if isinstance(value, Dim):
        return Domain((value,))

    return Domain(tuple(value))


@dataclasses.dataclass(frozen=True)
class CanonicalisedDomain:
    domain: Domain


def union(*domains: Domain) -> Domain:
    flattened_dims_in_original_order = itertools.chain.from_iterable(domain.dims for domain in domains)
    unique_dims_in_original_order = tuple(dict.fromkeys(flattened_dims_in_original_order))
    return Domain(unique_dims_in_original_order)