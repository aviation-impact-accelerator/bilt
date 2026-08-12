from bilt._backend._case import BackendCase
from bilt.symbolic._domain_handling import canonicalised, union
from bilt.symbolic._domains import CanonicalisedDomain


def derive_union_domain(case: BackendCase) -> CanonicalisedDomain:
    """Assumes the union domain within the backend case's scope can be dermined
    entirely from its parameter + return domains (i.e. no dimension addition)."""
    return canonicalised(union(
        *[param.domain for param in case.parameters.values()],
        case.return_domain.domain
    ))