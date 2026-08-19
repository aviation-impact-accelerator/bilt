import dataclasses
import typing

import numpy as np

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._tensors._tensor_alignment import AlignmentPlan


@dataclasses.dataclass
class BackendCallable:
    name: str
    parameters: tuple[str, ...]
    callable: collections.abc.Callable

    def evaluate(self, inputs: collections.abc.Mapping[str, object], alignment_plan: AlignmentPlan) -> object:
        inputs_to_use = {param_name: inputs[param_name] for param_name in self.parameters}
        aligned_inputs = _apply_alignment_plan(inputs_to_use, alignment_plan)
        return self.callable(**aligned_inputs)


def _apply_alignment_plan(inputs_to_use: dict[str, object], alignment_plan: AlignmentPlan) -> dict[str, object]:
    aligned_inputs: dict[str, object] = {}

    for input_name, input_value in inputs_to_use.items():
        if target_to_source_dim_mapping := alignment_plan.target_to_source_dim_mappings.get(input_name):
            transpose_axes = tuple(
                source_index for source_index in target_to_source_dim_mapping if source_index is not None
            )
            expand_axes = tuple(
                i for i, source_index in enumerate(target_to_source_dim_mapping) if source_index is None
            )

            transposed_input = typing.cast("np.ndarray", input_value).transpose(transpose_axes)
            aligned_inputs[input_name] = np.expand_dims(transposed_input, expand_axes)
        else:
            aligned_inputs[input_name] = input_value

    return aligned_inputs
