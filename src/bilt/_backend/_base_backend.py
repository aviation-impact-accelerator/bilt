"""Minimal interface for bilt backend, that will be utilised by the analysis layer."""

import abc
import ast
import typing

import numpy as np

from bilt._backend._callable import BackendCallable
from bilt._backend._errors import CompilationError

if typing.TYPE_CHECKING:
    from bilt.core._tensors._tensor_alignment import AlignmentPlan

    from ._case import BackendCase

_backend_stack: list[Backend] = []


class Backend(abc.ABC):
    """Interface for methods to implement by the JAX backend."""

    ast_transformer: ast.NodeTransformer
    compilation_namespace: dict[str, typing.Any]

    @abc.abstractmethod
    def __init__(
        self,
    ) -> None:
        """Must at minimum initialise the ast_transformer and compilation_namespace attributes."""

    def transpile(self, function_definition: ast.FunctionDef) -> BackendCallable:
        """Used to compile a single transform AST into a callable function, that can then be chained into the execution pipeline."""
        transformed = self.ast_transformer.visit(function_definition)
        module = ast.Module(body=[transformed], type_ignores=[])
        ast.fix_missing_locations(module)
        try:
            code = compile(module, "<backend_compile>", "exec")
        except Exception as e:
            msg = f"Error when trying to compile transformed function '{function_definition.name}' threw error: {e}"
            raise CompilationError(msg) from e

        exec(code, self.compilation_namespace)  # noqa: S102
        return BackendCallable(
            name=function_definition.name,
            parameters=tuple(arg.arg for arg in function_definition.args.args),
            callable=self.compilation_namespace[function_definition.name],
        )

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

    def __enter__(self) -> typing.Self:
        _backend_stack.append(self)
        return self

    def __exit__(self, *_: object) -> None:
        _backend_stack.pop()

    @abc.abstractmethod
    def chain_case(self, case: BackendCase) -> BackendCallable:
        """Execute the compiled function with the provided keyword arguments."""
