import dataclasses
import typing

import numpy as np

from bilt.core._tensors._canonical_ordering import CanonicalDomain, canonicalised, transpose_and_align_to_canonical
from bilt.core._tensors._tensors import Tensor, tensor
from bilt.symbolic._domains import Domain, union


@dataclasses.dataclass(frozen=True)
class LoweringPreparationResult:
    prepared_inputs: dict[str, object]
    any_tensors: bool
    output_domain: CanonicalDomain


def prepare_any_tensor_inputs_for_lowering_to_callable(
    inputs: dict[str, object]
) -> LoweringPreparationResult:
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


def lift_output(output_value: typing.Any, result: LoweringPreparationResult) -> Tensor | object:
    if result.any_tensors:
        return tensor(output_value, over=result.output_domain.domain.dims)
    else:
        return output_value


def _prepare_tensor_for_lowering(tensor: Tensor) -> np.ndarray:
    res = transpose_and_align_to_canonical(tensor)
    res = res.values
    return res


def _derive_output_domain(inputs: dict[str, object]) -> CanonicalDomain:
    input_domains: list[Domain] = [input_value.domain
                                   for input_value in inputs.values()
                                   if isinstance(input_value, Tensor)]

    input_union = canonicalised(union(*input_domains))

    return input_union  # assuming no addition/reduction