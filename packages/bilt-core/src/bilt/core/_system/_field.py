__all__ = ["field"]

from ._sentinel import MISSING, _Missing


def field() -> _Missing:
    return MISSING
