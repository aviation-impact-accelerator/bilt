__all__ = ["transform"]

import ast
import inspect
import textwrap
import typing

from bilt.core._enum_matching import enum_port_types
from bilt.core._errors import BiltCategoricalPortError, BiltUnresolvedAnnotationError
from bilt.core._port_declaration import PortDeclaration, port_declarations
from bilt.core._unit_erasure import erase_units

if typing.TYPE_CHECKING:
    import collections.abc


class Transform[R, **P](typing.Protocol):
    __bilt_name__: str
    __bilt_parameters__: tuple[str, ...]
    __bilt_port_declarations__: dict[str, PortDeclaration]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


class BasicTransform[R, **P]:
    def __init__(self, function: collections.abc.Callable[P, R]) -> None:
        self.__bilt_name__ = function.__name__  # ty: ignore[missing-attribute]
        try:
            signature = inspect.signature(function, eval_str=True)
        except NameError as error:
            raise BiltUnresolvedAnnotationError(self.__bilt_name__, error) from error
        self.__bilt_parameters__ = tuple(signature.parameters.keys())
        input_annotations = {port: parameter.annotation for port, parameter in signature.parameters.items()}
        output_annotation = {self.__bilt_name__: signature.return_annotation}
        input_types = enum_port_types(self.__bilt_name__, input_annotations)
        output_types = enum_port_types(self.__bilt_name__, output_annotation)
        if self.__bilt_name__ in input_annotations and input_types.get(self.__bilt_name__) is not output_types.get(
            self.__bilt_name__
        ):
            msg = "has different categorical types for its input and output of the same name."
            raise BiltCategoricalPortError(self.__bilt_name__, self.__bilt_name__, msg)
        self.__bilt_enum_ports__ = input_types | output_types
        self.__bilt_port_declarations__ = port_declarations(self.__bilt_name__, input_annotations | output_annotation)
        self._function = function
        self._function_ast = erase_units(
            typing.cast("ast.FunctionDef", ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]),
            self,
        )

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._function(*args, **kwargs)


def transform[R, **P](function: collections.abc.Callable[P, R]) -> BasicTransform[R, P]:
    """Decorator to denote a function as a transform."""
    return BasicTransform(function)
