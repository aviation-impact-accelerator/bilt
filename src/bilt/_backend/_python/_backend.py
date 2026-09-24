import ast
import inspect

from bilt._backend import BackendCallable, BackendCase
from bilt._backend._base_backend import ActivatableBackendMixin
from bilt._backend._in_process_python_mixin import InProcessPythonMixin
from bilt._backend._ndarray_alignment_mixin import NDArrayAlignmentMixin
from bilt._backend import Backend, BackendCallable, BackendCase, BackendLinearSystem
from bilt._backend._python._numerics import ScipyNumerics


class PythonBackend(InProcessPythonMixin, ActivatableBackendMixin, NDArrayAlignmentMixin):
    """Python backend implementation of the Bilt backend interface."""

    def __init__(self) -> None:
        self.ast_transformer = ast.NodeTransformer()
        self.compilation_namespace = {"__builtins__": __builtins__}

    def solve_linear(
        self,
        system: BackendLinearSystem,
        inputs: dict[str, object],
    ) -> dict[str, object]:
        coefficient_values = tuple(
            coefficient.evaluate(inputs)
            for row in system.coefficients
            for coefficient in row
        )
        rhs_values = tuple(
            rhs.evaluate(inputs)
            for rhs in system.rhs
        )

        size = len(system.unknowns)

        values = np.broadcast_arrays(
            *coefficient_values,
            *rhs_values,
        )

        coefficients = values[: size * size]
        rhs_values = values[size * size :]

        batch_shape = coefficients[0].shape

        matrix = np.stack(
            coefficients,
            axis=-1,
        ).reshape((*batch_shape, size, size))

        rhs = np.stack(rhs_values, axis=-1)

        solution = np.linalg.solve(matrix, rhs)

        return {
            unknown: solution[..., index]
            for index, unknown in enumerate(system.unknowns)
        }

    def chain_case(self, case: BackendCase) -> BackendCallable:
        """Chain a BackendCase into a single callable function."""
        expected_signature = inspect.Signature(
            parameters=[
                inspect.Parameter(
                    name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
                for name in case.parameters
            ]
        )

        def final_callable(*args, **kwargs) -> object:  # noqa: ANN002, ANN003
            bound = expected_signature.bind(*args, **kwargs)
            bound.apply_defaults()

            inputs = dict(bound.arguments)

            for intermediate in case.intermediates:
                if isinstance(intermediate, BackendLinearSystem):
                    inputs.update(
                        self.solve_linear(intermediate, inputs)
                    )
                    continue

                args_for_intermediate = [
                    inputs[name]
                    for name in intermediate.parameters
                ]
                result = intermediate.callable(*args_for_intermediate)
                inputs[intermediate.name] = result

            return inputs[case.return_name]

        final_callable.__signature__ = (  # pyrefly: ignore[missing-attribute]  # ty: ignore[unresolved-attribute]
            expected_signature
        )

        return BackendCallable(
            name=case.name,
            parameters=case.parameters,
            callable=final_callable,
        )
