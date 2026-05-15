__all__ = ["system"]

import dataclasses


@dataclasses.dataclass(frozen=True, slots=True)
class Attribute[T]:
    name: str
    annotation: type[T]


def system[T](cls: type[T]) -> type[T]:
    """Decorate a class as the interface of a system."""
    return cls
