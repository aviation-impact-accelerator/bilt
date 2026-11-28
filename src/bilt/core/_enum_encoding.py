"""Pure ordinal encoding for scalar enum members at backend boundaries."""

import enum
import operator
import typing


def enum_code(member: enum.Enum) -> int:
    """Encode a member by declaration order, not by its Python value."""
    return tuple(type(member)).index(member)


def enum_from_code[T: enum.Enum](enum_type: type[T], code: typing.SupportsIndex) -> T:
    """Decode an ordinal to a member of the *declared* enum class."""
    index = operator.index(code)
    members = tuple(enum_type)
    if not 0 <= index < len(members):
        msg = f"Enum code {index} is outside the valid range for {enum_type.__qualname__}."
        raise ValueError(msg)
    return members[index]
