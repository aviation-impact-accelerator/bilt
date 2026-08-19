import dataclasses

from bilt.core._evaluation_context import current_evaluation
from bilt.symbolic._domains import Dim, Domain


@dataclasses.dataclass(frozen=True)
class CanonicalizedDomain:
    domain: Domain


def canonicalized(domain: Domain) -> CanonicalizedDomain:
    """Puts dimensions in canonical order.

    Order determined by the order of dimensions in the current
    evaluation's domain registry.
    """
    canonicalised_dims: list[Dim] = [dim for dim in current_evaluation().canonical_order.dims if dim in domain.dims]

    return CanonicalizedDomain(Domain(tuple(canonicalised_dims)))
