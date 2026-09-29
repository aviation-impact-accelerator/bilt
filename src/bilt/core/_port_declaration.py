"""What a port's annotation declares about the quantity it carries."""

__all__ = ["PortDeclaration", "port_declarations"]

import dataclasses
import typing

from bilt.core._errors import BiltPortDeclarationError
from bilt.units import Dimension, Quantity, Unit

if typing.TYPE_CHECKING:
    import collections.abc


@dataclasses.dataclass
class PortDeclaration:
    """What a port's annotation declares: whether it carries a quantity, and of what dimensionality."""

    carries_quantity: bool = False
    dimensionality: Dimension | None = None


def port_declarations(
    owner_name: str,
    port_annotations: collections.abc.Mapping[str, object],
) -> dict[str, PortDeclaration]:
    """Read the value off each port's annotation to keep information about the port."""
    return {port: _declaration(owner_name, port, annotation) for port, annotation in port_annotations.items()}


def _declaration(owner_name: str, port: str, annotation: object) -> PortDeclaration:
    """Classify one annotation, ignoring `Annotated` metadata that is not a dimensionality."""
    while isinstance(annotation, typing.TypeAliasType):
        annotation = annotation.__value__
    base_type, *metadata = (
        typing.get_args(annotation) if typing.get_origin(annotation) is typing.Annotated else (annotation,)
    )
    unit = next((item for item in metadata if isinstance(item, Unit)), None)
    if unit is not None:
        msg = (
            f"declares unit {unit.name}, but a port declares a dimensionality, not a unit. "
            f"Declare its dimensionality ({unit.dimension}) instead."
        )
        raise BiltPortDeclarationError(owner_name, port, msg)

    dimensionalities = [item for item in metadata if isinstance(item, Dimension)]
    if len(dimensionalities) > 1:
        msg = f"declares more than one dimensionality ({', '.join(map(str, dimensionalities))}). Declare exactly one."
        raise BiltPortDeclarationError(owner_name, port, msg)

    dimensionality = dimensionalities[0] if dimensionalities else None
    carries_quantity = base_type is Quantity
    if dimensionality is not None and not carries_quantity:
        msg = (
            f"declares {dimensionality} on {getattr(base_type, '__name__', repr(base_type))}, "
            "but a dimensionality can only annotate Quantity. Use `typing.Annotated[Quantity, ...]`."
        )
        raise BiltPortDeclarationError(owner_name, port, msg)

    return PortDeclaration(carries_quantity=carries_quantity, dimensionality=dimensionality)
