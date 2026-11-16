"""Restricted categorical enum support for symbolic match/case equations."""

import enum
import typing

from bilt.core._port_declaration import annotated_parts


def enum_port_types(annotations: typing.Mapping[str, object]) -> dict[str, type[enum.Enum]]:
    """Collect exact enum classes from annotations (not enum members or arbitrary unions)."""
    result: dict[str, type[enum.Enum]] = {}
    for name, annotation in annotations.items():
        base, _ = annotated_parts(annotation)
        if isinstance(base, type) and issubclass(base, enum.Enum):
            result[name] = base
    return result


def enum_code(member: enum.Enum) -> int:
    """Use a stable ordinal rather than the enum's potentially non-numeric value."""
    return tuple(type(member)).index(member)
