import dataclasses
import typing

from bilt.core._tensors._reshaping import ReshapeSlices, derive_reshape_slices, is_scalar_reshape, is_unchanged_reshape
from bilt.symbolic._domain_handling import canonicalised, union

if typing.TYPE_CHECKING:
    from bilt._backend._callable import BackendCallable
    from bilt._backend._case import BackendCase
    from bilt.symbolic._domains import CanonicalisedDomain

type ExpectedName = str


@dataclasses.dataclass(frozen=True)
class Var:
    """The `ExpectedName` is 'a' or 'c'.

    But for a variable that's the output of a previous intermediate,
    `actual_name` would be 'c_'.
    """

    actual_name: str
    domain: CanonicalisedDomain


@dataclasses.dataclass
class IntermediateParameterAnalysis:
    var: Var
    reshape_slices: ReshapeSlices | None


@dataclasses.dataclass
class IntermediateAnalysis:
    parameters: dict[ExpectedName, IntermediateParameterAnalysis]
    actual_output_domain: CanonicalisedDomain


@dataclasses.dataclass
class IntermediatesAnalysis:
    intermediates: dict[ExpectedName, IntermediateAnalysis]


def analyse_intermediates(case: BackendCase) -> IntermediatesAnalysis:
    vars_in_scope: dict[ExpectedName, Var] = {}

    vars_in_scope |= {var_name: Var(var_name, var_domain) for var_name, var_domain in case.parameters.items()}

    vars_in_scope |= {
        intermediate.name: Var(f"{intermediate.name}_", intermediate.expected_return_domain)
        for intermediate in case.intermediates.values()
    }

    analysed_intermediates: dict[ExpectedName, IntermediateAnalysis] = {}

    for intermediate in case.intermediates.values():
        intermediate_analysis = _analyse_intermediate(intermediate, vars_in_scope)

        cur_val = vars_in_scope[intermediate.name]
        vars_in_scope[intermediate.name] = Var(cur_val.actual_name, intermediate_analysis.actual_output_domain)

        analysed_intermediates[intermediate.name] = intermediate_analysis

    return IntermediatesAnalysis(analysed_intermediates)


def _analyse_intermediate(
    intermediate: BackendCallable, vars_in_scope: typing.Mapping[ExpectedName, Var]
) -> IntermediateAnalysis:
    vars_to_use: dict[ExpectedName, Var] = {
        param_name: vars_in_scope[param_name] for param_name in intermediate.expected_parameters
    }

    actual_union_input_domain = union(*[var.domain.domain for var in vars_to_use.values()])
    actual_union_domain = canonicalised(union(actual_union_input_domain, intermediate.expected_union_domain.domain))
    actual_output_domain = actual_union_domain  # Assuming no reduction

    analysed_parameters: dict[ExpectedName, IntermediateParameterAnalysis] = {}

    for expected_param_name in intermediate.expected_parameters:
        actual_var = vars_to_use[expected_param_name]
        reshape_slices = derive_reshape_slices(actual_var.domain, actual_union_domain)

        if is_unchanged_reshape(reshape_slices) or is_scalar_reshape(reshape_slices):
            reshape_slices = None

        analysed_parameters[expected_param_name] = IntermediateParameterAnalysis(actual_var, reshape_slices)

    return IntermediateAnalysis(parameters=analysed_parameters, actual_output_domain=actual_output_domain)
