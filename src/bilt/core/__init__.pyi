__all__ = [
    "system",
    "transform",
]

import collections.abc
import typing

def system[T](cls: type[T]) -> type[T]: ...

class Transform[**P, R](typing.Protocol):
    __bilt_transform_name__: str
    __bilt_transform_parameters__: tuple[str, ...]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...

def transform[**P, R](func: collections.abc.Callable[P, R]) -> Transform[P, R]: ...
