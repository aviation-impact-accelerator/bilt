import ast
import typing

import jax
import jax.numpy as jnp

from bilt._backend import BackendLinearSystem
from bilt._backend._base_backend import ActivatableBackendMixin
from bilt._backend._in_process_python_mixin import InProcessPythonMixin
from bilt._backend._jax._ast import JAXASTTransformer
from bilt._backend._jax._supports import _validate
from bilt._backend._ndarray_alignment_mixin import NDArrayAlignmentMixin

if typing.TYPE_CHECKING:
    from bilt._backend import BackendCallable, BackendCase


class JAXBackend(
    InProcessPythonMixin,
    ActivatableBackendMixin,
    NDArrayAlignmentMixin,
):
    """JAX backend implementation of the Bilt backend interface."""

    def __init__(self) -> None:
        self.ast_transformer = JAXASTTransformer()
        self.compilation_namespace = {
            "jnp": jnp,
            "jax.jit": jax.jit,
            "__builtins__": __builtins__,
        }

    def transpile(self, function_definition: ast.FunctionDef) -> BackendCallable:
        """Compile a single transform AST into a callable function."""
        _validate(function_definition)
        return super().transpile(function_definition)

    def solve_linear(
        self,
        system: BackendLinearSystem,
        inputs: dict[str, object],
    ) -> dict[str, object]:
        coefficient_values = tuple(
            typing.cast("jax.typing.ArrayLike", coefficient.evaluate(inputs))
            for row in system.coefficients
            for coefficient in row
        )
        rhs_values = tuple(typing.cast("jax.typing.ArrayLike", rhs.evaluate(inputs)) for rhs in system.rhs)

        size = len(system.unknowns)

        values = jnp.broadcast_arrays(
            *coefficient_values,
            *rhs_values,
        )

        coefficients = values[: size * size]
        rhs_values = values[size * size :]

        batch_shape = coefficients[0].shape

        matrix = jnp.stack(
            coefficients,
            axis=-1,
        ).reshape((*batch_shape, size, size))

        rhs = jnp.stack(rhs_values, axis=-1)

        solution = jnp.linalg.solve(
            matrix,
            rhs[..., jnp.newaxis],
        )[..., 0]

        return {unknown: solution[..., index] for index, unknown in enumerate(system.unknowns)}

    def chain_case(self, case: BackendCase) -> BackendCallable:
        new_namespace: dict[str, typing.Any] = {}
        chained_callable_as_string = f"def {case.name}_({', '.join(case.parameters)}):\n"
        case_parameters = set(case.parameters)

        for index, intermediate in enumerate(case.intermediates):
            if isinstance(intermediate, BackendLinearSystem):
                callable_name = f"__bilt_linear_system_{index}"

                def solve_linear_system(
                    *args: object,
                    system: BackendLinearSystem = intermediate,
                ) -> tuple[object, ...]:
                    inputs = dict(zip(system.parameters, args, strict=True))
                    result = self.solve_linear(system, inputs)

                    return tuple(result[unknown] for unknown in system.unknowns)

                new_namespace[callable_name] = solve_linear_system

                arguments = ", ".join(
                    parameter if parameter in case_parameters else f"{parameter}_"
                    for parameter in intermediate.parameters
                )

                unknowns = ", ".join(f"{unknown}_" for unknown in intermediate.unknowns)

                if len(intermediate.unknowns) == 1:
                    unknowns += ","

                chained_callable_as_string += f"    {unknowns} = {callable_name}({arguments})\n"

                continue

            new_namespace[intermediate.name] = intermediate.callable

            arguments = ", ".join(
                parameter if parameter in case_parameters else f"{parameter}_" for parameter in intermediate.parameters
            )

            chained_callable_as_string += f"    {intermediate.name}_ = {intermediate.name}({arguments})\n"

        chained_callable_as_string += f"    return {case.return_name}_\n"

        chained_callable_ast = typing.cast(
            "ast.FunctionDef",
            ast.parse(chained_callable_as_string).body[0],
        )

        old_namespace = self.compilation_namespace.copy()
        self.compilation_namespace.update(new_namespace)

        try:
            final_callable = self.transpile(chained_callable_ast)
        finally:
            self.compilation_namespace = old_namespace

        return final_callable


JAX_BACKEND = JAXBackend()
