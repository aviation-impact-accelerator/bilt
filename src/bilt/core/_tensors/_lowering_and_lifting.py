import numpy as np

from bilt._backend._callable import BackendCallable
from bilt.core._evaluation_context import current_evaluation
from bilt.core._tensors import _tensors
from bilt.core._tensors._evaluation_domain_analysis import analyse_evaluation_domains
from bilt.core._tensors._tensors import Tensor, tensor
from bilt.symbolic._dimensions import Dim
from bilt.symbolic._domain_handling import canonicalised
from bilt.symbolic._domains import CanonicalisedDomain, domain


import dataclasses


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
            prepared_inputs[name] = _prepare_tensor_for_lowering(input, analysis.extra_dims)
            any_tensors = True
        else:
            prepared_inputs[name] = input

    return LoweringPreparationResult(
        prepared_inputs,
        any_tensors,
        analysis.output_domain,
    )


def lift_output(output_value: np.ndarray, result: LoweringPreparationResult):
    if result.any_tensors:
        return tensor(
            output_value,
            over=result.output_domain.domain.dims
        )
    else:
        return output_value.item()


def _transpose_tensor_to_canonical_order(tensor: Tensor, canonical_domain: CanonicalisedDomain) -> Tensor:
    """If a dim in `new_dim_order` is absent in `tensor`, that dim is ignored."""
    transposition_axes: list[int] = []
    transposed_dims: list[Dim] = []

    for dim in canonical_domain.domain.dims:
        if dim in tensor.domain.dims:
            transposition_axes.append(tensor.domain.dims.index(dim))
            transposed_dims.append(dim)

    transposed_values = tensor.values.transpose(transposition_axes)

    return _tensors.tensor(transposed_values, over=transposed_dims)


def _align_extra_dims(tensor: Tensor, extra_dims: CanonicalisedDomain) -> Tensor:
    """Important: `tensor` must already be in canonical order."""
    if len(extra_dims.domain.dims) == 0:
        return tensor

    new_values = tensor.values.copy()

    new_domain = canonicalised(domain(*list(set(tensor.domain.dims) | set(extra_dims.domain.dims))))

    for i, dim in enumerate(new_domain.domain.dims):
        if not dim in tensor.domain.dims and dim in extra_dims.domain.dims:
            new_values = np.expand_dims(new_values, axis=i)

    return Tensor(new_values, new_domain.domain)


def _prepare_tensor_for_lowering(tensor: Tensor, extra_dims: CanonicalisedDomain) -> np.ndarray:
    prepared_tensor = _transpose_tensor_to_canonical_order(tensor, current_evaluation().domains.get_canonical_domain())
    prepared_tensor = _align_extra_dims(prepared_tensor, extra_dims)
    return prepared_tensor.values