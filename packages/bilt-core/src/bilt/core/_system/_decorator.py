__all__ = ["system"]

import dataclasses
import typing

from ._interface import Interface


@dataclasses.dataclass(frozen=True, slots=True)
class Attribute[T]:
    name: str
    annotation: type[T]


def system[T](cls: type[T]) -> type[T]:
    """Decorate a class as the interface of a system."""
    interface = Interface(cls.__name__, cls.__bases__, dict(cls.__dict__))
    return typing.cast("type[T]", interface)
