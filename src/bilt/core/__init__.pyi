__all__ = [
    "Implementation",
    "system",
    "transform",
]

import collections.abc
import typing

class Implementation:
    def __init__(
        self, system: type[typing.Any], transforms: collections.abc.Iterable[Transform[..., typing.Any]]
    ) -> None: ...
    @property
    def system(self) -> type[typing.Any]: ...
    @property
    def transforms(self) -> dict[str, Transform[..., typing.Any]]: ...
    def evaluate(self, inputs: dict[str, object], output: str) -> object: ...
    def __enter__(self) -> typing.Self: ...
    def __exit__(self, *_: object) -> None: ...

def system[T](cls: type[T]) -> type[T]: ...

class Transform[**P, R](typing.Protocol):
    __bilt_transform_name__: str
    __bilt_transform_parameters__: tuple[str, ...]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...

def transform[**P, R](func: collections.abc.Callable[P, R]) -> Transform[P, R]: ...
