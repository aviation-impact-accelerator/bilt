"""Minimal interface for bilt backend, that will be utilised by the analysis layer."""

import abc
import ast
import typing

from bilt._backend._callable import BackendCallable, CallableType
from bilt._backend._errors import CompilationError
from bilt.core._tensors._tensor_analysis import *

if typing.TYPE_CHECKING:
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

    def transpile(self, function_definition: ast.FunctionDef) -> CallableType:
        """Used to compile a single transform AST into a callable function, that can then be chained into the execution pipeline."""
        transformed = typing.cast(ast.FunctionDef, self.ast_transformer.visit(function_definition))

        module = ast.Module(body=[transformed], type_ignores=[])
        ast.fix_missing_locations(module)
        try:
            code = compile(module, "<backend_compile>", "exec")
        except Exception as e:
            msg = f"Error when trying to compile transformed function '{function_definition.name}' threw error: {e}"
            raise CompilationError(msg) from e

        exec(code, self.compilation_namespace)  # noqa: S102

        print()
        print("===== TRANSPILE =====")
        print(ast.unparse(transformed))

        return typing.cast(CallableType, self.compilation_namespace[function_definition.name])

    def __enter__(self) -> typing.Self:
        _backend_stack.append(self)
        return self

    def __exit__(self, *_: object) -> None:
        _backend_stack.pop()

    @abc.abstractmethod
    def chain_case(self, case: BackendCase) -> BackendCallable:
        """Execute the compiled function with the provided keyword arguments."""
