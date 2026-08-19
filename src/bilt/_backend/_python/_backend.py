import ast
import inspect

from bilt._backend import BackendCallable, BackendCase
from bilt._backend._base_backend import ActivatableBackendMixin
from bilt._backend._in_process_python_mixin import InProcessPythonMixin
from bilt._backend._ndarray_alignment_mixin import NDArrayAlignmentMixin


class PythonBackend(InProcessPythonMixin, ActivatableBackendMixin, NDArrayAlignmentMixin):
    """Python backend implementation of the Bilt backend interface."""

    def __init__(self) -> None:
        self.ast_transformer = ast.NodeTransformer()
        self.compilation_namespace = {"__builtins__": __builtins__}

    def chain_case(self, case: BackendCase) -> BackendCallable:
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
            if not case.expose_intermediates:
                return inputs[case.return_name]
            return {name: inputs[name] for name in case.output_names}

        final_callable.__signature__ = (  # pyrefly: ignore[missing-attribute]  # ty: ignore[unresolved-attribute]
            expected_signature
        )
        return BackendCallable(
            name=case.return_name,
            parameters=case.parameters,
            callable=final_callable,
            outputs=case.output_names if case.expose_intermediates else None,
        )


PYTHON_BACKEND = PythonBackend()
