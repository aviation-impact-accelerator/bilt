import ast
import dataclasses
import typing

import numpy as np

from bilt._backend._callable import BackendCallable
from bilt.core._evaluation_context import current_evaluation
from bilt.core._tensors import _tensors
from bilt.core._tensors._tensors import Tensor, tensor
from bilt.symbolic._dimensions import Dim, dim
from bilt.symbolic._domains import CanonicalisedDomain, Domain, domain
from bilt.symbolic._domains import union

SLICE_COLON = slice(None)
SLICE_NEWAXIS = None

type ReshapeSlices = tuple[slice | None, ...]

def extract_domains_from_transform_signature(func_def: ast.FunctionDef) -> dict[str, Domain]:
    res: dict[str, Domain] = {}

    # TODO: What about posonly and kwonly args
    for arg in func_def.args.args + [ast.arg("return", annotation=func_def.returns)]:
        name = arg.arg
        annotation = arg.annotation

        match annotation:
            case ast.Subscript():
                value = typing.cast(ast.Name, annotation.value).id
                slice = annotation.slice
                dims: list[Dim] = []

                assert value == "Tensor"

                match slice:
                    case ast.Tuple():
                        for elem in typing.cast(list[ast.Name], slice.elts):
                            dims.append(dim(elem.id))
                    case ast.Constant():
                        assert slice.value is None
                        # dims will be left empty (scalar domain)
                    case ast.Name():
                        dims.append(dim(slice.id))
                    case _:
                        raise RuntimeError()

                res[name] = Domain(tuple(dims))
            case ast.Name():
                assert annotation.id == "float"
                res[name] = Domain.scalar()
            case _:
                raise NotImplementedError()

    return res

def derive_reshape_slices(domain: CanonicalisedDomain, target_domain: CanonicalisedDomain) -> ReshapeSlices:
    reshape_slices: list[slice | None] = []

    domain_index = 0

    def any_domain_dims_left():
        return domain_index < len(domain.domain.dims)

    for target_domain_index in range(len(target_domain.domain.dims)):
        target_domain_dim = target_domain.domain.dims[target_domain_index]

        if any_domain_dims_left() and domain.domain.dims[domain_index] == target_domain_dim:
            reshape_slices.append(SLICE_COLON)
            domain_index += 1
        else:
            reshape_slices.append(SLICE_NEWAXIS)

    if any_domain_dims_left():
        message = "Went all the way through `target_domain` and still have `domain` dims left - " \
        "`domain` and `target_domain` not in the same order? Shouldn't be possible now we're taking " \
        "in `CanonicalisedDomain`s."
        
        raise RuntimeError(message)

    return tuple(reshape_slices)

def is_scalar_reshape(reshape_slices: ReshapeSlices) -> bool:
    for slice in reshape_slices:
        if slice != SLICE_NEWAXIS:
            return False

    return True

def is_unchanged_reshape(reshape_slices: ReshapeSlices) -> bool:
    for slice in reshape_slices:
        if slice != SLICE_COLON:
            return False

    return True

def domain_to_tensor_type_string(domain: Domain) -> str:
    if not domain.dims:
        return "Tensor[None]"
    
    return f"Tensor[{','.join(dim.name for dim in domain.dims)}]"

def clear_signature_type_hints(func_def: ast.FunctionDef) -> None:
    # TODO: What about posonly and kwonly args - will we ever have any?
    for arg in func_def.args.args:
        arg.annotation = None

    func_def.returns = None

@dataclasses.dataclass(frozen=True)
class DomainAnalysis:
    parameters: dict[str, DomainedVar]
    return_domain: CanonicalisedDomain
    union_domain: CanonicalisedDomain

@dataclasses.dataclass(frozen=True)
class DomainedVar:
    name: str
    domain: CanonicalisedDomain


def analyse_domains_in_transform_signature(func_def: ast.FunctionDef) -> DomainAnalysis:
    domains = extract_domains_from_transform_signature(func_def)
    domains = {name: canonicalised(domain) for name, domain in domains.items()}
    union_domain = canonicalised(union(*[domain.domain for domain in domains.values()]))

    return_domain = domains.pop("return")
    parameters = {
        name: DomainedVar(name, domain)
        for name, domain in domains.items()
    }

    return DomainAnalysis(
        parameters,
        return_domain,
        union_domain,
    )

def convert_any_tensors_to_raw_arrays(inputs: dict[str, object]) -> dict[str, object]:
    """Convert `Tensor`s to raw arrays in `inputs`.
    
    If an input is not a `Tensor`, it is left alone."""
    res: dict[str, object] = {}

    for name, value in inputs.items():
        if isinstance(value, Tensor):
            value = value.values

        res[name] = value

    return res

def identify_extra_dims(domain: Domain, relative_to: Domain) -> Domain:
    return Domain(tuple(set(domain.dims) - set(relative_to.dims)))

def add_dimensions(domain: Domain, extra_dims: tuple[Dim, ...]) -> Domain:
    return Domain(domain.dims + extra_dims)

@dataclasses.dataclass(frozen=True)
class LoweringPreparationResult:
    prepared_inputs: dict[str, object]
    any_tensors: bool
    output_domain: CanonicalisedDomain

def prepare_any_tensor_inputs_for_lowering_to_callable(inputs: dict[str, object], callable: BackendCallable) -> LoweringPreparationResult:
    prepared_inputs: dict[str, object] = {}
    any_tensors = False

    analysis = analyse_evaluation_domains(inputs, callable)

    for name, input in inputs.items():
        if isinstance(input, Tensor):
            prepared_inputs[name] = prepare_tensor_for_lowering(input, analysis.extra_dims)
            any_tensors = True
        else:
            prepared_inputs[name] = input

    return LoweringPreparationResult(
        prepared_inputs,
        any_tensors,
        analysis.output_domain,
    )

@dataclasses.dataclass(frozen=True)
class EvaluationDomainsAnalysis:
    extra_dims: CanonicalisedDomain
    output_domain: CanonicalisedDomain

def analyse_evaluation_domains(inputs: dict[str, object], callable: BackendCallable) -> EvaluationDomainsAnalysis:
    input_domains: list[Domain] = []
    
    for name in inputs.keys():
        input = inputs[name]
        if isinstance(input, Tensor):
            input_domains.append(input.domain)

    input_union = canonicalised(union(*input_domains))
    callable_union = callable.expected_union_domain
    extra_dims = canonicalised(identify_extra_dims(input_union.domain, callable_union.domain))

    return EvaluationDomainsAnalysis(
        extra_dims, output_domain=input_union  # Assuming no domain addition or reduction
    )    

def prepare_tensor_for_lowering(tensor: Tensor, extra_dims: CanonicalisedDomain) -> np.ndarray:
    prepared_tensor = transpose_tensor_to_canonical_order(tensor, current_evaluation().domains.get_canonical_domain())
    prepared_tensor = align_extra_dims(prepared_tensor, extra_dims)
    return prepared_tensor.values

def lift_output(output_value: np.ndarray, result: LoweringPreparationResult):
    if result.any_tensors:
        return tensor(
            output_value,
            over=result.output_domain.domain.dims
        )
    else:
        return output_value.item()

def transpose_any_tensors_to_canonical_order(inputs: dict[str, object], canonical_domain: CanonicalisedDomain) -> dict[str, object]:
    res: dict[str, object] = {}

    for name, value in inputs.items():
        if isinstance(value, Tensor):
            value = transpose_tensor_to_canonical_order(value, canonical_domain)

        res[name] = value

    return res

def transpose_tensor_to_canonical_order(tensor: Tensor, canonical_domain: CanonicalisedDomain) -> Tensor:
    """If a dim in `new_dim_order` is absent in `tensor`, that dim is ignored."""
    transposition_axes: list[int] = []
    transposed_dims: list[Dim] = []

    for dim in canonical_domain.domain.dims:
        if dim in tensor.domain.dims:
            transposition_axes.append(tensor.domain.dims.index(dim))
            transposed_dims.append(dim)

    transposed_values = tensor.values.transpose(transposition_axes)

    return _tensors.tensor(transposed_values, over=transposed_dims)

def canonicalised(domain: Domain) -> CanonicalisedDomain:
    """Canonicalises order of passed domain.
    
    Order determined by the order of dimensions in the current
    evaluation's domain registry.
    """

    canonical_domain = current_evaluation().domains.get_canonical_domain()
    canonicalised_dims: list[Dim] = []

    for dim in canonical_domain.domain.dims:
        if dim in domain.dims:            
            canonicalised_dims.append(dim)

    return CanonicalisedDomain(Domain(tuple(canonicalised_dims)))

def any_tensors(values: list[object]) -> bool:
    for value in values:
        if isinstance(value, Tensor):
            return True

    return False

def align_extra_dims(tensor: Tensor, extra_dims: CanonicalisedDomain) -> Tensor:
    """Important: `tensor` must already be in canonical order."""
    if len(extra_dims.domain.dims) == 0:
        return tensor

    new_values = tensor.values.copy()

    new_domain = canonicalised(domain(*list(set(tensor.domain.dims) | set(extra_dims.domain.dims))))

    for i, dim in enumerate(new_domain.domain.dims):
        if not dim in tensor.domain.dims and dim in extra_dims.domain.dims:
            new_values = np.expand_dims(new_values, axis=i)

    return Tensor(new_values, new_domain.domain)