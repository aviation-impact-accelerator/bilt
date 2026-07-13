import ast

from bilt._backend import Backend, BackendCallable, BackendCase


class PythonBackend(Backend):
    """Python backend implementation of the Bilt backend interface."""

    def __init__(self) -> None:
        self.ast_transformer = ast.NodeTransformer()
        self.compilation_namespace = {"__builtins__": __builtins__}

    def chain_case(self, case: BackendCase) -> BackendCallable:
        """Chain a BackendCase into a single callable function."""

        def final_callable(**kwargs) -> object:  # noqa: ANN003
            inputs = dict(kwargs)
            for parameter in case.parameters:
                if parameter not in inputs:
                    msg = f"Missing required argument: {parameter}"
                    raise TypeError(msg)
            for extra in inputs:
                if extra not in case.parameters:
                    msg = f"Unexpected argument: {extra}"
                    raise TypeError(msg)

            for intermediate in case.intermediates:
                args_for_intermediate = [inputs[name] for name in intermediate.parameters]
                result = intermediate.callable(*args_for_intermediate)
                inputs[intermediate.name] = result
            return inputs[case.return_name]

        return BackendCallable(name=case.name, parameters=case.parameters, callable=final_callable)
