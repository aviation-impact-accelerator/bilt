"""Validate scalar enum port annotations on systems and transforms."""

import enum
import typing

from bilt.core._errors import BiltCategoricalPortError
from bilt.core._port_declaration import annotated_parts
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
    """Collect concrete scalar enum classes, rejecting unsupported annotations."""
    result: dict[str, type[enum.Enum]] = {}

    def validate_port(name: str, annotation: object) -> None:
        base, metadata = annotated_parts(annotation)
        if not isinstance(base, type) or not issubclass(base, enum.Enum):
            if not isinstance(base, type) and _contains_enum(base):
                msg = "must use a concrete enum class directly for a categorical port."
                raise BiltCategoricalPortError(owner, name, msg)
            return
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
