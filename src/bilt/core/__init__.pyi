"""Only needed to satisfy Pyright."""
from collections.abc import Callable
from typing import Protocol

class Transform[**P, R](Protocol):
    __bilt_transform_name__: str
    __bilt_transform_parameters__: tuple[str, ...]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...

def transform[**P, R](func: Callable[P, R]) -> Transform[P, R]: ...

__all__ = ("transform",)
