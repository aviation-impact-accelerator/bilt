import dataclasses
import functools
import typing

from bilt._backend._tensor_handling._canonical_ordering import (
    CanonicalizedDomain,
    canonicalized,
    transpose_and_align_to_canonical,
)
from bilt.core._tensors._tensors import Tensor, tensor
from bilt.symbolic._domains import Domain, union

if typing.TYPE_CHECKING:
    import numpy as np


@dataclasses.dataclass(frozen=True)
class LiftingPlan:
    output_domain: CanonicalizedDomain

    @functools.cached_property
    def should_lift_into_tensor(self) -> bool:
        return not self.output_domain.domain.is_scalar


def lower_inputs(inputs: dict[str, object]) -> dict[str, object]:
    lowered_inputs: dict[str, object] = {}

    for input_name, input_value in inputs.items():
        if isinstance(input_value, Tensor):
            lowered_inputs[input_name] = _lower_tensor(input_value)
        else:
            lowered_inputs[input_name] = input_value

    return lowered_inputs


def generate_lifting_plan(inputs: dict[str, object]) -> LiftingPlan:
    return LiftingPlan(output_domain=_derive_output_domain(inputs))


def lift_output(output_value: object, result: LiftingPlan) -> Tensor | object:
    if result.should_lift_into_tensor:
        output_value = tensor(output_value, over=result.output_domain.domain.dims)

    return output_value


def _lower_tensor(tensor: Tensor) -> np.ndarray:
    canonical_tensor = transpose_and_align_to_canonical(tensor)
    return canonical_tensor.values


def _derive_output_domain(inputs: dict[str, object]) -> CanonicalizedDomain:
    input_domains: list[Domain] = [
        input_value.domain for input_value in inputs.values() if isinstance(input_value, Tensor)
    ]

    return canonicalized(union(*input_domains))  # assuming no addition/reduction
