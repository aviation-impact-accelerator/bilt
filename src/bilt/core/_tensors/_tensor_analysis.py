__all__ = [
    #"extract_domains_from_signature",
    "extract_domains_from_func_def_signature",
    "union",
    "derive_reshape_slices",
    "ReshapeSlices",
    "make_reshaping_assignment",
    "is_scalar_reshape",
    "is_unchanged_reshape",
    "domain_to_tensor_type_string",
    "add_alignment_prelude",
    "clear_signature_type_hints",
    "analyse_domains_in_signature",
    "DomainedVar",
    "DomainAnalysis",
    "convert_any_tensors_to_raw_arrays",
    "identify_extra_dims",
    "add_dimensions",
    "make_slice_ast",
    "transpose_tensor_to_canonical_order",
    "transpose_any_tensors_to_canonical_order",
    "canonicalised",
    "any_tensors",
    "align_extra_dims",
]

import ast
import dataclasses
import itertools
import typing

import numpy as np

from bilt.core._evaluation_context import current_evaluation
from bilt.core._tensors import _tensors
from bilt.core._tensors._tensors import Tensor
from bilt.symbolic._dimensions import Dim, dim
from bilt.symbolic._domains import Domain, domain

SLICE_COLON = slice(None)
SLICE_NEWAXIS = None

type ReshapeSlices = tuple[slice | None, ...]

# def extract_domains_from_signature[R, **P](func: collections.abc.Callable[P, R]) -> dict[str, Domain]:
#     res: dict[str, Domain] = {}

#     for name, annotated in func.__annotations__.items():  # pyright: ignore[reportPrivateUsage]
#         _, domain = typing.get_args(annotated)
#         res[name] = domain
    
#     return res

def extract_domains_from_func_def_signature(func_def: ast.FunctionDef) -> dict[str, Domain]:
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

def union(*domains: Domain) -> Domain:
    flattened_dims_in_original_order = itertools.chain.from_iterable(domain.dims for domain in domains)
    unique_dims_in_original_order = tuple(dict.fromkeys(flattened_dims_in_original_order))
    return Domain(unique_dims_in_original_order)

def derive_reshape_slices(domain: Domain, target_domain: Domain) -> ReshapeSlices:
    reshape_slices: list[slice | None] = []

    domain_index = 0

    def any_domain_dims_left():
        return domain_index < len(domain.dims)

    for target_domain_index in range(len(target_domain.dims)):
        target_domain_dim = target_domain.dims[target_domain_index]

        if any_domain_dims_left() and domain.dims[domain_index] == target_domain_dim:
            reshape_slices.append(SLICE_COLON)
            domain_index += 1
        else:
            reshape_slices.append(SLICE_NEWAXIS)

    if any_domain_dims_left():
        message = "Went all the way through `target_domain` and still have `domain` dims left - `domain` and `target_domain` not in the same order?"
        raise RuntimeError(message)

    return tuple(reshape_slices)

def make_slice_ast(param_name: str, reshape_slices: ReshapeSlices) -> ast.Subscript:
    elts: list[ast.expr] = []
    
    for slice in reshape_slices:
        if slice == SLICE_COLON:
            elts.append(ast.Slice(lower=None, upper=None, step=None))
        elif slice == SLICE_NEWAXIS:
            elts.append(ast.Constant(value=None))
        else:
            assert False

    return ast.Subscript(
        value=ast.Name(id=param_name, ctx=ast.Load()),
        slice=ast.Tuple(elts=elts, ctx=ast.Load()),
        ctx=ast.Load(),
    )


def make_reshaping_assignment(param_name: str, reshape_slices: ReshapeSlices) -> ast.Assign:
    """Generates an AST Assign node for: target_name = target_name[reshape_slices]"""
    return ast.Assign(
        targets=[ast.Name(id=param_name, ctx=ast.Store())],
        value=make_slice_ast(param_name, reshape_slices),
    )

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

def add_alignment_prelude(func_def: ast.FunctionDef, analysis: DomainAnalysis) -> None:
    reshape_slices: dict[str, ReshapeSlices] = {
        name: derive_reshape_slices(param.domain, analysis.union_domain)
        for (name, param) in analysis.parameters.items()
    }

    reshaping_assignments: list[ast.Assign] = [
        make_reshaping_assignment(name, reshape_slices)
        for (name, reshape_slices) in reshape_slices.items()
        if not is_scalar_reshape(reshape_slices)
        and not is_unchanged_reshape(reshape_slices)
    ]

    func_def.body[0:0] = reshaping_assignments

def clear_signature_type_hints(func_def: ast.FunctionDef) -> None:
    # TODO: What about posonly and kwonly args - will we ever have any?
    for arg in func_def.args.args:
        arg.annotation = None

    func_def.returns = None

@dataclasses.dataclass(frozen=True)
class DomainAnalysis:
    parameters: dict[str, DomainedVar]
    return_domain: Domain
    union_domain: Domain

@dataclasses.dataclass(frozen=True)
class DomainedVar:
    name: str
    domain: Domain


def analyse_domains_in_signature(func_def: ast.FunctionDef) -> DomainAnalysis:
    domains = extract_domains_from_func_def_signature(func_def)
    domains = {name: canonicalised(domain) for name, domain in domains.items()}
    union_domain = union(*domains.values())

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

def identify_extra_dims(domain: Domain, relative_to: Domain) -> tuple[Dim, ...]:
    return tuple(set(domain.dims) - set(relative_to.dims))

def add_dimensions(domain: Domain, extra_dims: tuple[Dim, ...]) -> Domain:
    return Domain(domain.dims + extra_dims)

def transpose_any_tensors_to_canonical_order(inputs: dict[str, object], canonical_order: list[Dim]) -> dict[str, object]:
    res: dict[str, object] = {}

    for name, value in inputs.items():
        if isinstance(value, Tensor):
            value = transpose_tensor_to_canonical_order(value, canonical_order)

        res[name] = value

    return res

def transpose_tensor_to_canonical_order(tensor: Tensor, canonical_order: list[Dim]) -> Tensor:
    """If a dim in `new_dim_order` is absent in `tensor`, that dim is ignored."""
    transposition_axes: list[int] = []
    transposed_dims: list[Dim] = []

    for dim in canonical_order:
        if dim in tensor.domain.dims:
            transposition_axes.append(tensor.domain.dims.index(dim))
            transposed_dims.append(dim)

    transposed_values = tensor.values.transpose(transposition_axes)

    return _tensors.tensor(transposed_values, over=transposed_dims)

def canonicalised(domain: Domain) -> Domain:
    """Canonicalises order of passed domain.
    
    Order determined by the order of dimensions in the current
    evaluation's domain registry.
    """

    canonical_order = list(current_evaluation().domains._bindings.keys())
    canonicalised_dims: list[Dim] = []

    for dim in canonical_order:
        if dim in domain.dims:            
            canonicalised_dims.append(dim)

    return Domain(tuple(canonicalised_dims))

def any_tensors(values: list[object]) -> bool:
    for value in values:
        if isinstance(value, Tensor):
            return True

    return False

def align_extra_dims(tensor: Tensor, extra_dims: list[Dim]) -> Tensor:
    """Align pre-supposes that everything is in canonical order and so we only need
    to add singleton dimensions to get extra dimensions to align. We also assume that
    extra_dims is in canonical order too."""
    new_values = tensor.values.copy()

    new_domain = canonicalised(domain(*list(set(tensor.domain.dims) | set(extra_dims))))

    for i, dim in enumerate(new_domain.dims):
        if not dim in tensor.domain.dims and dim in extra_dims:
            new_values = np.expand_dims(new_values, axis=i)

    return Tensor(new_values, new_domain)