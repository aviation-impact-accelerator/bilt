"""Minimal interface for Bilt backends used by the analysis layer."""

import abc
import ast
import typing

from bilt._backend._callable import BackendCallable
from bilt._backend._errors import CompilationError

if typing.TYPE_CHECKING:
    from ._case import BackendCase

_backend_stack: list[Backend] = []


class Backend(abc.ABC):
    """Interface implemented by Bilt backends."""

    ast_transformer: ast.NodeTransformer
    compilation_namespace: dict[str, typing.Any]

    @abc.abstractmethod
    def __init__(self) -> None:
        """Initialise the AST transformer and compilation namespace."""

    def transpile(
        self,
        function_definition: ast.FunctionDef,
    ) -> BackendCallable:
        """Compile a function AST into a backend callable."""
        transformed = self.ast_transformer.visit(function_definition)
        module = ast.Module(
            body=[transformed],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)

        try:
            code = compile(
                module,
                "<backend_compile>",
                "exec",
            )
        except Exception as e:
            msg = f"Error when trying to compile transformed function {function_definition.name!r} threw error: {e}"
            raise CompilationError(msg) from e

        exec(code, self.compilation_namespace)  # noqa: S102

        return BackendCallable(
            name=function_definition.name,
            parameters=tuple(arg.arg for arg in function_definition.args.args),
            callable=self.compilation_namespace[function_definition.name],
        )

    def linear_solve(
        self,
        coefficients: object,
        rhs: object,
    ) -> object:
        """Solve a dense linear system A x = b."""
        msg = f"{type(self).__name__} does not support linear solves."
        raise NotImplementedError(msg)

    def __enter__(self) -> typing.Self:
        _backend_stack.append(self)
        return self

    def __exit__(self, *_: object) -> None:
        _backend_stack.pop()

    @abc.abstractmethod
    def chain_case(
        self,
        case: BackendCase,
    ) -> BackendCallable:
        """Chain a backend case into a single callable."""
