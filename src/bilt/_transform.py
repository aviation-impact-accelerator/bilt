"""Decorate functions as transforms."""

from __future__ import annotations

__all__ = ("BasicTransform", "transform")

import inspect
import typing

if typing.TYPE_CHECKING:
    import collections.abc


class Transform[R, **P](typing.Protocol):
    __bilt_transform_name__: str
    __bilt_transform_parameters__: tuple[str, ...]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


class BasicTransform[R, **P]:
    def __init__(self, function: collections.abc.Callable[P, R]) -> None:
        self.__bilt_transform_name__ = function.__name__  # ty: ignore[unresolved-attribute]
        self.__bilt_transform_parameters__ = tuple(inspect.signature(function).parameters.keys())
        self._function = function

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._function(*args, **kwargs)


transform = BasicTransform
