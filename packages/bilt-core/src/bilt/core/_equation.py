__all__ = ["equation"]

import inspect
import typing

if typing.TYPE_CHECKING:
    import collections.abc


class Equation[R, **P](typing.Protocol):
    __bilt_transform_name__: str
    __bilt_transform_parameters__: tuple[str, ...]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> bool: ...


class BasicEquation[**P]:
    def __init__(self, function: collections.abc.Callable[P, bool]) -> None:
        self.__bilt_transform_name__ = function.__name__  # ty: ignore[unresolved-attribute]
        self.__bilt_transform_parameters__ = tuple(inspect.signature(function).parameters.keys())
        self._function = function

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> bool:
        return self._function(*args, **kwargs)


def equation[**P](function: collections.abc.Callable[P, bool]) -> BasicEquation[P]:
    """Decorator to denote a function as an equation."""
    return BasicEquation(function)
