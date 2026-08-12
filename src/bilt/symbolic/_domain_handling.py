import itertools
import typing

from bilt.core._evaluation_context import current_evaluation
from bilt.symbolic._domains import CanonicalisedDomain, Domain

if typing.TYPE_CHECKING:
    from bilt.symbolic._dimensions import Dim


def union(*domains: Domain) -> Domain:
    flattened_dims_in_original_order = itertools.chain.from_iterable(domain.dims for domain in domains)
    unique_dims_in_original_order = tuple(dict.fromkeys(flattened_dims_in_original_order))
    return Domain(unique_dims_in_original_order)


def canonicalised(domain: Domain) -> CanonicalisedDomain:
    """Canonicalises order of passed domain.

    Order determined by the order of dimensions in the current
    evaluation's domain registry.
    """
    canonical_domain = current_evaluation().domains.get_canonical_domain()
    canonicalised_dims: list[Dim] = [dim for dim in canonical_domain.domain.dims if dim in domain.dims]

    return CanonicalisedDomain(Domain(tuple(canonicalised_dims)))
