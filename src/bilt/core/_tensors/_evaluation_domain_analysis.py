import dataclasses
import typing

from bilt.core._tensors._tensors import Tensor
from bilt.symbolic._domain_handling import canonicalised, union
from bilt.symbolic._domains import CanonicalisedDomain, Domain

if typing.TYPE_CHECKING:
    from bilt._backend._callable import BackendCallable


@dataclasses.dataclass(frozen=True)
class EvaluationDomainAnalysis:
    extra_dims: CanonicalisedDomain
    output_domain: CanonicalisedDomain


def identify_extra_dims(domain: Domain, relative_to: Domain) -> Domain:
    return Domain(tuple(set(domain.dims) - set(relative_to.dims)))


def analyse_evaluation_domains(inputs: dict[str, object], backend_callable: BackendCallable) -> EvaluationDomainAnalysis:
    input_domains: list[Domain] = [input_value.domain
                                   for input_value in inputs.values()
                                   if isinstance(input_value, Tensor)]

    input_union = canonicalised(union(*input_domains))
    callable_union = backend_callable.expected_union_domain
    extra_dims = canonicalised(identify_extra_dims(input_union.domain, callable_union.domain))

    return EvaluationDomainAnalysis(
        extra_dims,
        output_domain=input_union,  # Assuming no domain addition or reduction
    )
