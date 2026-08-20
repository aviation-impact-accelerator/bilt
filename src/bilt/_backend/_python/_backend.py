import ast
import inspect
import typing

import numpy as np

from bilt._backend import BackendCallable, BackendCase
from bilt._backend._base_backend import ActivatableBackendMixin
from bilt._backend._in_process_python_mixin import InProcessPythonMixin

if typing.TYPE_CHECKING:
    from bilt.core._tensors._tensor_alignment import AlignmentPlan


class PythonBackend(InProcessPythonMixin, ActivatableBackendMixin):
    """Python backend implementation of the Bilt backend interface."""

    def __init__(self) -> None:
        self.ast_transformer = ast.NodeTransformer()
        self.compilation_namespace = {"__builtins__": __builtins__}

    def chain_case(self, case: BackendCase) -> BackendCallable:
        """Chain a BackendCase into a single callable function."""
        expected_signature = inspect.Signature(
            parameters=[inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in case.parameters]
        )

        def final_callable(*args, **kwargs) -> object:  # noqa: ANN002, ANN003
            bound = expected_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            inputs = dict(values)

            for intermediate in case.intermediates:
                args_for_intermediate = [inputs[name] for name in intermediate.parameters]
                result = intermediate.callable(*args_for_intermediate)
                inputs[intermediate.name] = result
            return inputs[case.return_name]

        final_callable.__signature__ = (  # pyrefly: ignore[missing-attribute]  # ty: ignore[unresolved-attribute]
            expected_signature
        )
        return BackendCallable(name=case.name, parameters=case.parameters, callable=final_callable)

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


PYTHON_BACKEND = PythonBackend()
