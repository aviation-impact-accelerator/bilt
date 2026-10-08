"""What a port's annotation declares about the quantity it carries."""

__all__ = ["PortDeclaration", "port_declarations"]

import dataclasses
import typing

from bilt.core._errors import BiltPortDeclarationError
from bilt.core._tensors._tensors import Tensor
from bilt.units import Dimension, Quantity, Unit
from bilt.units._errors import BiltUnitError
from bilt.units._units import BASE_UNIT_SYSTEM
from bilt.units.dimensions import ABSOLUTE

if typing.TYPE_CHECKING:
    import collections.abc


@dataclasses.dataclass
class PortDeclaration:
    """What a declared port holds: a quantity of its dimensionality, absolute if marked and relative otherwise."""

    dimensionality: Dimension
    absolute: bool = False

    @property
    def coherent_unit(self) -> Unit:
        """The unit the port answers in: the coherent unit of its dimensionality, or that unit's absolute counterpart."""
        coherent_unit = BASE_UNIT_SYSTEM.coherent_unit(self.dimensionality)
        return abs(coherent_unit) if self.absolute else coherent_unit

    def __str__(self) -> str:
        return f"absolute {self.dimensionality}" if self.absolute else str(self.dimensionality)


def port_declarations(
    owner_name: str,
    port_annotations: collections.abc.Mapping[str, object],
) -> dict[str, PortDeclaration]:
    """Read the value off each port's annotation to keep information about the port."""
    declarations = {port: _declaration(owner_name, port, annotation) for port, annotation in port_annotations.items()}
    return {port: declaration for port, declaration in declarations.items() if declaration is not None}


def _declaration(owner_name: str, port: str, annotation: object) -> PortDeclaration | None:
    """Classify one annotation, ignoring `Annotated` metadata that is neither a dimensionality nor the absolute marker."""
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
    absolute = any(item is ABSOLUTE for item in metadata)
    if dimensionality is None and (carries_quantity or absolute):
        msg = (
            f"is {'a Quantity' if carries_quantity else 'marked ABSOLUTE'} without a dimensionality. Declare one "
            "with `typing.Annotated[Quantity, DIM]`, or write `float` if the port is outside the unit system."
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

    if dimensionality is None:
        return None

    declaration = PortDeclaration(dimensionality, absolute=absolute)
    if absolute:
        try:
            _ = declaration.coherent_unit
        except BiltUnitError as error:
            msg = f"is marked ABSOLUTE, but {dimensionality} cannot be absolute: {error}."
            raise BiltPortDeclarationError(owner_name, port, msg) from error
    return declaration
