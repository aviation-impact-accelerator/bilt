"""Validate scalar enum port annotations on systems and transforms."""

import enum
import types
import typing

from bilt.core._errors import BiltCategoricalPortError
from bilt.core._port_declaration import annotated_parts
from bilt.core._tensors._tensors import Tensor
from bilt.core._validation import validate_independently
from bilt.symbolic._domains import Domain
from bilt.units import Dimension, Metadata, Unit
from bilt.units.dimensions import ABSOLUTE


def _contains_enum(annotation: object) -> bool:
    pending = [annotation]
    seen: set[int] = set()
    while pending:
        item, _ = annotated_parts(pending.pop())
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, enum.Enum) or (isinstance(item, type) and issubclass(item, enum.Enum)):
            return True
        pending.extend(typing.get_args(item))
    return False


def enum_port_types(owner: str, annotations: typing.Mapping[str, object]) -> dict[str, type[enum.Enum]]:
    """Collect nominal enum types for scalar and scalar-or-Tensor declarations."""
    result: dict[str, type[enum.Enum]] = {}

    def validate_port(name: str, annotation: object) -> None:
        base, metadata = annotated_parts(annotation)
        enum_type, _ = _enum_shape(base)
        if enum_type is None:
            if _contains_enum(base):
                msg = "must use a concrete enum class or a union of that class with Tensor."
                raise BiltCategoricalPortError(owner, name, msg)
            return
        base = enum_type
        if issubclass(base, enum.Flag):
            msg = f"declares {base.__name__}. Flag and IntFlag are unsupported enum port types."
            raise BiltCategoricalPortError(owner, name, msg)
        if not base.__members__:
            msg = f"declares {base.__name__}, which has no enum members. Use a concrete enum class."
            raise BiltCategoricalPortError(owner, name, msg)
        if any(isinstance(item, Dimension | Domain | Metadata | Unit) or item is ABSOLUTE for item in metadata):
            msg = "declares units, dimensionality or a tensor domain on a categorical port."
            raise BiltCategoricalPortError(owner, name, msg)
        result[name] = base

    validate_independently(
        lambda name=name, annotation=annotation: validate_port(name, annotation)
        for name, annotation in annotations.items()
    )
    return result


def _enum_shape(annotation: object) -> tuple[type[enum.Enum] | None, bool]:
    """Recognise a scalar enum or the same enum paired with ordinary Tensor."""
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return annotation, False
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        options = typing.get_args(annotation)
        if len(options) == 2 and Tensor in options:  # noqa: PLR2004
            other = next(option for option in options if option is not Tensor)
            if isinstance(other, type) and issubclass(other, enum.Enum):
                return other, True
    return None, False


def enum_tensor_ports(annotations: typing.Mapping[str, object]) -> frozenset[str]:
    """Ports that explicitly accept a tensor of their declared enum type."""
    return frozenset(name for name, annotation in annotations.items() if _enum_shape(annotated_parts(annotation)[0])[1])
