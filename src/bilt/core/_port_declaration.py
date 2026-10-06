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
    """Read the dimensionality each port's annotation declares, leaving out plain ports that declare none."""
    declarations: dict[str, Dimension] = {}
    for port, annotation in port_annotations.items():
        dimensionality = _dimensionality(owner_name, port, annotation)
        if dimensionality is not None:
            declarations[port] = dimensionality
    return declarations


def _dimensionality(owner_name: str, port: str, annotation: object) -> Dimension | None:
    """Read one annotation's dimensionality, ignoring `Annotated` metadata that is not one; plain ports give `None`."""
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

    if not dimensionalities:
        if base_type is Quantity:
            msg = (
                "is a Quantity without a dimensionality. Declare one with `typing.Annotated[Quantity, DIM]`, "
                "or write `float` if the port is outside the unit system."
            )
            raise BiltPortDeclarationError(owner_name, port, msg)
        return None

    dimensionality = dimensionalities[0]
    if base_type is not Quantity:
        if base_type is Tensor:
            msg = (
                f"declares {dimensionality} on Tensor, but tensors are outside the unit system. "
                "Remove the dimensionality."
            )
        else:
            msg = (
                f"declares {dimensionality} on {getattr(base_type, '__name__', repr(base_type))}, "
                "but a dimensionality can only annotate Quantity. Use `typing.Annotated[Quantity, ...]`."
            )
        raise BiltPortDeclarationError(owner_name, port, msg)

    return dimensionality
