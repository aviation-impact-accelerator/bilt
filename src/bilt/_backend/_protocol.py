"""Minimal interface for bilt backend, that will be utilised by the analysis layer."""

import abc
import ast
import typing

if typing.TYPE_CHECKING:
    from ._case import BackendCase

from bilt._backend import BackendCallable


class Backend(abc.ABC):
    """Interface for methods to implement by the JAX backend."""

    ast_transformer: ast.NodeTransformer
    compilation_namespace: dict[str, typing.Any]

    @abc.abstractmethod
    def __init__(
        self,
    ) -> None:
        """Must at minimum initialise the ast_transformer and compilation_namespace attributes."""
        ...

    def transpile(self, function_definition: ast.FunctionDef) -> BackendCallable:
        """Used to compile a single transform AST into a callable function, that can then be chained into the execution pipeline."""
        transformed = self.ast_transformer.visit(function_definition)
        module = ast.Module(body=[transformed], type_ignores=[])
        ast.fix_missing_locations(module)
        code = compile(module, "<jax_emit>", "exec")
        exec(code, self.compilation_namespace)  # noqa: S102

        return BackendCallable(
            name=function_definition.name,
            parameters=tuple(arg.arg for arg in function_definition.args.args),
            callable=self.compilation_namespace[function_definition.name],
        )

    @abc.abstractmethod
    def chain_case(self, case: BackendCase) -> BackendCallable:
        """Execute the compiled function with the provided keyword arguments."""
        ...
