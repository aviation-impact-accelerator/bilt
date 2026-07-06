__all__ = [
    "Domain",
    "DomainLike",
    "as_domain",
    "broadcast_domain",
    "domain",
]

import dataclasses

from bilt.symbolic._dim import Dim

type DomainLike = Domain | Dim | tuple[Dim, ...] | list[Dim] | None


@dataclasses.dataclass(frozen=True, slots=True)
class Domain:
    """A simple ordered collection of dimensions.

    A scalar has an empty domain.
    """

    dims: tuple[Dim, ...] = ()

    @classmethod
    def scalar(cls) -> Domain:
        return cls(())

    @property
    def is_scalar(self) -> bool:
        return not self.dims

    def __str__(self) -> str:
        if not self.dims:
            return "scalar"

        return "[" + ", ".join(str(d) for d in self.dims) + "]"


def domain(*dims: Dim) -> Domain:
    return Domain(tuple(dims))


def as_domain(value: DomainLike) -> Domain:
    if value is None:
        return Domain.scalar()

    if isinstance(value, Domain):
        return value

    if isinstance(value, Dim):
        return Domain((value,))

    return Domain(tuple(value))


def broadcast_domain(left: DomainLike, right: DomainLike) -> Domain:
    """Merge two domains, preserving dimension order.

    Scalars broadcast to the other domain.
    """
    lhs = as_domain(left)
    rhs = as_domain(right)

    if lhs.is_scalar:
        return rhs

    if rhs.is_scalar:
        return lhs

    dims = list(lhs.dims)

    for dim in rhs.dims:
        if dim not in dims:
            dims.append(dim)

    return Domain(tuple(dims))