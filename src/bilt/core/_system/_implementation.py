__all__ = ["BiltSystemEvaluationError", "Implementation"]

import ast
import inspect
import textwrap
import types
import typing

from bilt._backend import Backend, BackendCallable, BackendCase
from bilt._backend._python._backend import PythonBackend
from bilt.core._equation import BasicEquation
from bilt.core._error import BiltCoreError
from bilt.core._transform import BasicTransform

from ._interface import Interface

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._operation import BiltOperation


class BiltSystemEvaluationError(BiltCoreError):
    pass


class Implementation:
    """Register a collection of operations as an implementation of a system."""

    def __init__(
        self,
        system: Interface,
        operations: collections.abc.Iterable[BiltOperation],
        backend: Backend = PythonBackend(),  # noqa: B008
    ) -> None:
        self._system = system
        self._operations = types.MappingProxyType({operation.__bilt_name__: operation for operation in operations})
        self._backend = backend
        self.callable_cache: dict[tuple[str, tuple[str, ...]], BackendCallable] = {}
        system.__bilt_system_registry__.add(self)

    @property
    def system(self) -> Interface:
        return self._system

    def evaluate(self, inputs: dict[str, object], output: str) -> object:
        if self.callable_cache.get((output, tuple(inputs.keys()))):
            return self.callable_cache[(output, tuple(inputs.keys()))].callable(**inputs)
        if output in inputs:
            return inputs[output]

        if output not in self._operations:
            message = f"No operation registered to produce output {output!r}"
            raise BiltSystemEvaluationError(message)

        backend_case = self._find_successful_case(inputs, output)

        self.callable_cache[(output, tuple(inputs.keys()))] = self._backend.chain_case(backend_case)
        return self.callable_cache[(output, tuple(inputs.keys()))].callable(**inputs)

    def _find_successful_case(self, inputs: dict[str, object], output: str) -> BackendCase:
        return BackendCase(
            name=output,
            parameters=tuple(inputs.keys()),
            intermediates=list(
                map(
                    self._operation_to_callable,
                    (self._operations[x] for x in self._path_through_graph(inputs, output)[::-1]),
                )
            ),
            return_name=output,
        )

    def _path_through_graph(self, inputs: dict[str, object], output: str) -> list[str]:
        ordered_callables = []
        operation = self._operations[output]

        parameters = operation.__bilt_parameters__
        for parameter in parameters:
            if parameter not in inputs:
                ordered_callables.append(parameter)
                ordered_callables.extend(self._path_through_graph(inputs, parameter))
                inputs[parameter] = self.evaluate(inputs=inputs, output=parameter)
        ordered_callables.append(output)

        return ordered_callables

    def _operation_to_callable(self, operation: BiltOperation) -> BackendCallable:
        match operation:
            case BasicTransform():
                function_ast = typing.cast(
                    "ast.FunctionDef",
                    ast.parse(textwrap.dedent(inspect.getsource(operation._function))).body[0],  # noqa: SLF001
                )
                function_ast.decorator_list = [
                    d for d in function_ast.decorator_list if not (isinstance(d, ast.Name) and d.id == "transform")
                ]
                return self._backend.transpile(function_ast)

            case BasicEquation():
                msg = "Equation evaluation is not yet implemented."
                raise NotImplementedError(msg)
            case Interface():
                msg = "Interface evaluation is not yet implemented."
                raise NotImplementedError(msg)
            case _:
                msg = f"Operation type {type(operation)} is not supported for evaluation."
                raise NotImplementedError(msg)

    def __enter__(self) -> typing.Self:
        self._system.__bilt_system_implementation_default__.append(self)
        return self

    def __exit__(self, *_: object) -> None:
        self._system.__bilt_system_implementation_default__.pop()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Implementation):
            return NotImplemented
        return self.system == other.system and self._operations == other._operations

    def __hash__(self) -> int:
        return hash((self.system, frozenset(self._operations.items())))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(system={self.system}, operations={frozenset(self._operations.values())!r})"
