import dataclasses

from bilt.core._evaluation_context import current_evaluation
from bilt.core._tensors._tensors import Tensor


@dataclasses.dataclass(frozen=True)
class AlignmentPlan:
    target_to_source_dim_mappings: dict[str, tuple[int | None, ...]]


def generate_alignment_plan(inputs: dict[str, object]) -> AlignmentPlan:
    tensor_inputs = {
        input_name: input_value for input_name, input_value in inputs.items() if isinstance(input_value, Tensor)
    }

    canonical_ordering = current_evaluation().canonical_order

    target_to_source_dim_mappings: dict[str, tuple[int | None, ...]] = {}

    for tensor_name, tensor in tensor_inputs.items():
        mapping: list[int | None] = []

        for dim in canonical_ordering.dims:
            if dim in tensor.domain.dims:
                mapping.append(tensor.domain.dims.index(dim))
            else:
                mapping.append(None)

        target_to_source_dim_mappings[tensor_name] = tuple(mapping)

    return AlignmentPlan(target_to_source_dim_mappings)
