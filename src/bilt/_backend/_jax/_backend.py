import ast
import typing

import jax
import jax.numpy as jnp
import numpy as np

from bilt._backend._base_backend import ActivatableBackendMixin
from bilt._backend._in_process_python_mixin import InProcessPythonMixin
from bilt._backend._jax._ast import JAXASTTransformer
from bilt._backend._jax._supports import _validate

if typing.TYPE_CHECKING:
    from bilt._backend import BackendCallable, BackendCase
    from bilt.core._tensors._tensor_alignment import AlignmentPlan


class JAXBackend(InProcessPythonMixin, ActivatableBackendMixin):
    """JAX backend implementation of the Bilt backend interface."""

    def __init__(self) -> None:
        self.ast_transformer = JAXASTTransformer()
        self.compilation_namespace = {"jnp": jnp, "jax.jit": jax.jit, "__builtins__": __builtins__}

    def transpile(self, function_definition: ast.FunctionDef) -> BackendCallable:
        """Compile a single transform AST into a callable function."""
        _validate(function_definition)
        return super().transpile(function_definition)

    def chain_case(self, case: BackendCase) -> BackendCallable:
        new_namespace: dict[str, typing.Any] = {}
        chained_callable_as_string = f"def {case.name}_({', '.join(f'{x}' for x in case.parameters)}):\n"
        case_parameters = set(case.parameters)
        for interim_callable in case.intermediates:
            new_namespace[interim_callable.name] = interim_callable.callable
            chained_callable_as_string += f"    {interim_callable.name}_={interim_callable.name}({
                ', '.join(
                    parameter if parameter in case_parameters else f'{parameter}_'
                    for parameter in interim_callable.parameters
                )
            })\n"
        chained_callable_as_string += f"    return {case.return_name}_\n"
        chained_callable_ast = typing.cast("ast.FunctionDef", ast.parse(chained_callable_as_string).body[0])

        old_namespace = self.compilation_namespace.copy()
        self.compilation_namespace.update(new_namespace)
        final_callable = self.transpile(chained_callable_ast)
        self.compilation_namespace = old_namespace

        return final_callable

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


JAX_BACKEND = JAXBackend()
