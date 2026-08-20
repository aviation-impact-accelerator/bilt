import typing

import numpy as np

if typing.TYPE_CHECKING:
    from bilt.core._tensors._tensor_alignment import AlignmentPlan


class AlignmentMixin:
    """Makes a backend able to align raw array inputs."""

    def apply_alignment(self, lowered_inputs: dict[str, object], alignment_plan: AlignmentPlan) -> dict[str, object]:
        """Apply the given alignment plan on the provided lowered inputs."""
        aligned_inputs: dict[str, object] = {}

        for input_name, input_value in lowered_inputs.items():
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
