import ast
import inspect
import typing

from bilt._backend import Backend, BackendCallable

if typing.TYPE_CHECKING:
    from bilt._backend._case import BackendCase
    from bilt.core._system._analysis import BroadcastPlan


class PythonBackend(Backend):
    """Python backend implementation of the Bilt backend interface."""

    def __init__(self) -> None:
        self.ast_transformer = ast.NodeTransformer()
        self.compilation_namespace = {"__builtins__": __builtins__}

    def apply_broadcast_plan(
        self, function_definition: ast.FunctionDef, broadcast_plan: BroadcastPlan
    ) -> ast.FunctionDef:
        # Don't currently need to do anything as PythonBackend is using BiltTensor operator overloads. Will need in future.
        _ = broadcast_plan

        return function_definition

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
