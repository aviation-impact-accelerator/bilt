import dataclasses
from typing import TYPE_CHECKING

from bilt.symbolic._domains import Dim, Domain

if TYPE_CHECKING:
    from bilt.core._evaluation_context import EvaluationContext


@dataclasses.dataclass(frozen=True)
class CanonicalizedDomain:
    domain: Domain


def canonicalized(domain: Domain, canonical_order: Domain) -> CanonicalizedDomain:
    """Puts dimensions in canonical order.

    Order determined by the order of dimensions in the current
    evaluation's domain registry.
    """
    canonicalised_dims: list[Dim] = [dim for dim in canonical_order.dims if dim in domain.dims]

    return CanonicalizedDomain(Domain(tuple(canonicalised_dims)))


def _derive_canonical_order(current_evaluation: EvaluationContext) -> Domain:
    return Domain(tuple(current_evaluation.domains._bindings.keys()))  # noqa: SLF001
