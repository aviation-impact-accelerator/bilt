import ast
import inspect

from bilt._backend import BackendCallable, BackendCase
from bilt._backend._base_backend import ActivatableBackendMixin
from bilt._backend._in_process_python_mixin import InProcessPythonMixin


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


PYTHON_BACKEND = PythonBackend()
