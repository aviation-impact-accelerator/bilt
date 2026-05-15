__all__ = ["system"]


def system[T](cls: type[T]) -> type[T]:
    """Decorate a class as the interface of system."""
    return cls
