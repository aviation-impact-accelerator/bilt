"""What a port's annotation declares about the quantity it carries."""

__all__ = ["port_declarations"]

import typing

from bilt.core._errors import BiltPortDeclarationError
from bilt.core._tensors._tensors import Tensor
from bilt.units import Dimension, Quantity, Unit

if typing.TYPE_CHECKING:
    import collections.abc


def port_declarations(
    owner_name: str,
    port_annotations: collections.abc.Mapping[str, object],
) -> dict[str, Dimension]:
    """Read the value off each port's annotation to keep information about the port."""
    declarations = {port: _declaration(owner_name, port, annotation) for port, annotation in port_annotations.items()}
    return {port: dimensionality for port, dimensionality in declarations.items() if dimensionality is not None}


def _declaration(owner_name: str, port: str, annotation: object) -> Dimension | None:
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
    if dimensionality is None and carries_quantity:
        msg = (
            "is a Quantity without a dimensionality. Declare one with `typing.Annotated[Quantity, DIM]`, "
            "or write `float` if the port is outside the unit system."
        )
        raise BiltPortDeclarationError(owner_name, port, msg)
    if dimensionality is not None and not carries_quantity:
        msg = (
            f"declares {dimensionality} on {getattr(base_type, '__name__', repr(base_type))}, "
            "but a dimensionality can only annotate Quantity. Use `typing.Annotated[Quantity, ...]`."
        )
        if base_type is Tensor:
            msg = f"declares {dimensionality} on Tensor, but tensors are outside the unit system."
        raise BiltPortDeclarationError(owner_name, port, msg)

    return dimensionality
