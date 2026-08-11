"""Minimal interface for bilt backend, that will be utilised by the analysis layer."""

import abc
import ast
import typing

from bilt._backend._callable import BackendCallable
from bilt._backend._errors import CompilationError

if typing.TYPE_CHECKING:
    import collections.abc

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

    def transpile(self, function_definition: ast.FunctionDef) -> BackendCallable:
        """Used to compile a single transform AST into a callable function, that can then be chained into the execution pipeline."""
        transformed = self.ast_transformer.visit(function_definition)
        module = ast.Module(body=[transformed], type_ignores=[])
        ast.fix_missing_locations(module)
        try:
            code = compile(module, "<backend_compile>", "exec")
        except Exception as e:
            msg = f"Error when trying to compile transformed function '{function_definition.name}' threw error: {e}"
            raise CompilationError(msg) from e

        exec(code, self.compilation_namespace)  # noqa: S102
        return BackendCallable(
            name=function_definition.name,
            parameters=tuple(arg.arg for arg in function_definition.args.args),
            callable=self.compilation_namespace[function_definition.name],
        )

    def __enter__(self) -> typing.Self:
        _backend_stack.append(self)
        return self

    def __exit__(self, *_: object) -> None:
        _backend_stack.pop()

    @abc.abstractmethod
    def chain_case(self, case: BackendCase) -> BackendCallable:
        """Execute the compiled function with the provided keyword arguments."""

    def _identity(self) -> tuple[collections.abc.Hashable, ...]:
        """Define when two backend instances are interchangeable for cache keying.."""
        return (type(self),)

    def __eq__(self, other: object) -> bool:
        """Compare two backend instances for equality based on their identity."""
        if not isinstance(other, Backend):
            return NotImplemented
        return self._identity() == other._identity()

    def __hash__(self) -> int:
        """Return hash of this backend instance's identity."""
        return hash(self._identity())
