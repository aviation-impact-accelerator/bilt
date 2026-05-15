__all__ = ["system"]


def system[T](cls: type[T]) -> type[T]:
    """Decorate a class as the interface of a system."""
    return cls
