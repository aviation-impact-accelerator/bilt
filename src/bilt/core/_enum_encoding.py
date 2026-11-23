"""Pure ordinal encoding for scalar enum members at backend boundaries."""

import enum
import operator
import typing

import numpy as np


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


def enum_codes(values: np.ndarray) -> np.ndarray:
    """Encode an enum-containing ordinary Tensor as integer ordinals."""
    return np.fromiter((enum_code(member) for member in values.flat), dtype=np.int64, count=values.size).reshape(
        values.shape
    )


def enum_from_codes[T: enum.Enum](enum_type: type[T], codes: np.ndarray) -> np.ndarray:
    """Decode backend results while preserving enum objects, including IntEnum."""
    array = np.asarray(codes)
    result = np.empty(array.shape, dtype=object)
    for index in np.ndindex(array.shape):
        result[index] = enum_from_code(enum_type, array[index])
    return result
