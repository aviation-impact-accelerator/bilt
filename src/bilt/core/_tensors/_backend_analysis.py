import typing

from bilt.symbolic._domain_handling import canonicalised, union

if typing.TYPE_CHECKING:
    from bilt._backend._case import BackendCase
    from bilt.symbolic._domains import CanonicalisedDomain


def derive_union_domain(case: BackendCase) -> CanonicalisedDomain:
    """Derives from parameter + return domains (assumes no dim addition)."""
    return canonicalised(union(*[param.domain for param in case.parameters.values()], case.return_domain.domain))
