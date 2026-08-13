import dataclasses
import typing

from bilt.core._tensors._canonical_ordering import CanonicalDomain, canonicalised, transpose_and_align_to_canonical
from bilt.core._tensors._tensors import Tensor, tensor
from bilt.symbolic._domains import Domain, union

if typing.TYPE_CHECKING:
    import numpy as np


@dataclasses.dataclass(frozen=True)
class LoweringPreparationResult:
    prepared_inputs: dict[str, object]
    any_tensors: bool
    output_domain: CanonicalDomain


def prepare_any_tensor_inputs_for_lowering_to_callable(inputs: dict[str, object]) -> LoweringPreparationResult:
    prepared_inputs: dict[str, object] = {}
    any_tensors = False

    output_domain = _derive_output_domain(inputs)

    for input_name, input_value in inputs.items():
        if isinstance(input_value, Tensor):
            prepared_inputs[input_name] = _prepare_tensor_for_lowering(input_value)
            any_tensors = True
        else:
            prepared_inputs[input_name] = input_value

    return LoweringPreparationResult(
        prepared_inputs,
        any_tensors,
        output_domain,
    )


def lift_output(output_value: object, result: LoweringPreparationResult) -> Tensor | object:
    if result.any_tensors:
        return tensor(output_value, over=result.output_domain.domain.dims)
    return output_value


def _prepare_tensor_for_lowering(tensor: Tensor) -> np.ndarray:
    canonical_tensor = transpose_and_align_to_canonical(tensor)
    return canonical_tensor.values


def _derive_output_domain(inputs: dict[str, object]) -> CanonicalDomain:
    input_domains: list[Domain] = [
        input_value.domain for input_value in inputs.values() if isinstance(input_value, Tensor)
    ]

    return canonicalised(union(*input_domains))  # assuming no addition/reduction
