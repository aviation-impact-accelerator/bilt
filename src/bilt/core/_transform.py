__all__ = ["transform"]

import inspect
import typing

from bilt.core._port_declaration import port_declarations

if typing.TYPE_CHECKING:
    import collections.abc


class Transform[R, **P](typing.Protocol):
    __bilt_name__: str
    __bilt_parameters__: tuple[str, ...]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


class BasicTransform[R, **P]:
    def __init__(self, function: collections.abc.Callable[P, R]) -> None:
        self.__bilt_name__ = function.__name__  # ty: ignore[unresolved-attribute]
        signature = inspect.signature(function)
        self.__bilt_parameters__ = tuple(signature.parameters.keys())
        self.__bilt_port_declarations__ = port_declarations(
            self.__bilt_name__,
            {port: parameter.annotation for port, parameter in signature.parameters.items()}
            | {self.__bilt_name__: signature.return_annotation},
        )
        self._function = function

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._function(*args, **kwargs)


def transform[R, **P](function: collections.abc.Callable[P, R]) -> BasicTransform[R, P]:
    """Decorator to denote a function as a transform."""
    return BasicTransform(function)
