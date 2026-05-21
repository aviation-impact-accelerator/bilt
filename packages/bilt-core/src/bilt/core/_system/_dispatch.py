__all__ = []

import typing

if typing.TYPE_CHECKING:
    from ._decorator import Attribute
    from ._interface import Interface


def _dispatch_evaluation[T](instance: Interface, attribute: Attribute[T]) -> T:
    raise NotImplementedError
