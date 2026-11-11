"""What a port's annotation declares about the quantity it carries."""

__all__ = ["PortDeclaration", "annotated_parts", "port_declarations"]

import dataclasses
import types
import typing

from bilt.core._errors import BiltPortDeclarationError
from bilt.core._tensors._tensors import QuantityTensor, Tensor
from bilt.core._validation import validate_independently
from bilt.symbolic._domains import Domain
from bilt.units import Dimension, Metadata, Quantity, Unit
from bilt.units._errors import BiltUnitError
from bilt.units._units import BASE_UNIT_SYSTEM
from bilt.units.dimensions import ABSOLUTE

if typing.TYPE_CHECKING:
    import collections.abc


@dataclasses.dataclass
class PortDeclaration:
    """What a declared port holds: a quantity of its dimensionality, absolute if marked and relative otherwise.

    `over_a_domain` records that the port requires a tensor rather than a scalar. It says what a caller may
    pass at this port, not what the port measures, so it takes no part in a declaration's identity: a constituent
    written over scalars is reusable in a tensor system under the same port name.
    """

    dimensionality: Dimension
    absolute: bool = False
    over_a_domain: bool = dataclasses.field(default=False, compare=False)
    scalar_only: bool = dataclasses.field(default=False, compare=False)
    required_domain: Domain | None = dataclasses.field(default=None, compare=False)

    @property
    def coherent_unit(self) -> Unit:
        """The unit the port answers in: the coherent unit of its dimensionality, or that unit's absolute counterpart."""
        coherent_unit = BASE_UNIT_SYSTEM.coherent_unit(self.dimensionality)
        return abs(coherent_unit) if self.absolute else coherent_unit

    def __str__(self) -> str:
        return f"absolute {self.dimensionality}" if self.absolute else str(self.dimensionality)


def annotated_parts(annotation: object) -> tuple[object, list[object]]:
    """Split an annotation into the type it is written over and its `Annotated` metadata, through any type alias."""
    while isinstance(annotation, typing.TypeAliasType):
        annotation = annotation.__value__
    if typing.get_origin(annotation) is typing.Annotated:
        base_type, *metadata = typing.get_args(annotation)
        return base_type, metadata
    return annotation, []


def _is_pointwise_quantity_union(base_type: object) -> bool:
    """Recognise scalar-or-tensor quantities in pointwise annotations."""
    return typing.get_origin(base_type) in (typing.Union, types.UnionType) and frozenset(
        typing.get_args(base_type)
    ) == {Quantity, QuantityTensor}


def quantity_magnitude_kind(base_type: object) -> typing.Literal["scalar", "tensor", "either"] | None:
    """Classify concrete unit-bearing quantities by accepted shape."""
    if _is_pointwise_quantity_union(base_type):
        return "either"
    if base_type is Quantity:
        return "scalar"
    if base_type is QuantityTensor:
        return "tensor"
    return None


def _required_domain(  # noqa: PLR0913
    owner_name: str,
    port: str,
    metadata: list[object],
    base_type: object,
    dimensionality: Dimension | None,
    *,
    system_boundary: bool,
) -> Domain | None:
    """Distinguish exact equation hints from minimum tensor-domain requirements."""
    # Existing dimensioned Tensor annotations remain valid until the migration is done.
    # Keep existing scalar hints until the system-field migration is complete.
    # Pointwise equation hints are exact; tensor field requirements are minimum.
    if (
        dimensionality is None
        or base_type in (Quantity, Tensor)
        or (not system_boundary and _is_pointwise_quantity_union(base_type))
    ):
        return None

    domains = [item for item in metadata if isinstance(item, Domain)]
    if len(domains) > 1:
        msg = "declares more than one required domain. Declare at most one Domain in Annotated."
        raise BiltPortDeclarationError(owner_name, port, msg)
    if not domains:
        return None
    if base_type is not QuantityTensor:
        msg = "declares a required Domain on a non-tensor quantity; use QuantityTensor."
        raise BiltPortDeclarationError(owner_name, port, msg)
    return domains[0]


def port_declarations(
    owner_name: str,
    port_annotations: collections.abc.Mapping[str, object],
    *,
    system_boundary: bool = False,
) -> dict[str, PortDeclaration]:
    """Read the value off each port's annotation to keep information about the port."""
    declarations: dict[str, PortDeclaration] = {}

    def validate_port(port: str, annotation: object) -> None:
        if (declaration := _declaration(owner_name, port, annotation, system_boundary=system_boundary)) is not None:
            declarations[port] = declaration

    validate_independently(
        lambda port=port, annotation=annotation: validate_port(port, annotation)
        for port, annotation in port_annotations.items()
    )
    return declarations


def _declaration(
    owner_name: str, port: str, annotation: object, *, system_boundary: bool = False
) -> PortDeclaration | None:
    """Classify one annotation, ignoring `Annotated` metadata that is neither a dimensionality nor the absolute marker."""
    base_type, metadata = annotated_parts(annotation)
    unit = next((item for item in metadata if isinstance(item, Unit)), None)
    if unit is not None:
        msg = (
            f"declares unit {unit.name}, but a port declares a dimensionality, not a unit. "
            f"Declare its dimensionality ({unit.dimension}) instead."
        )
        raise BiltPortDeclarationError(owner_name, port, msg)

    tag = next((item for item in metadata if isinstance(item, Metadata)), None)
    if tag is not None:
        msg = (
            f"declares tag {tag} on its own, but a tag is a factor of a dimensionality. "
            "Multiply the dimensionality by it, as in `typing.Annotated[Quantity, DIM * tag]`."
        )
        raise BiltPortDeclarationError(owner_name, port, msg)

    dimensionalities = [item for item in metadata if isinstance(item, Dimension)]
    if len(dimensionalities) > 1:
        msg = f"declares more than one dimensionality ({', '.join(map(str, dimensionalities))}). Declare exactly one."
        raise BiltPortDeclarationError(owner_name, port, msg)

    dimensionality = dimensionalities[0] if dimensionalities else None
    magnitude_kind = quantity_magnitude_kind(base_type)
    carries_quantity = base_type is Quantity or magnitude_kind is not None
    over_a_domain = base_type in (Tensor, QuantityTensor) or magnitude_kind == "tensor"
    required_domain = _required_domain(
        owner_name, port, metadata, base_type, dimensionality, system_boundary=system_boundary
    )
    absolute = any(item is ABSOLUTE for item in metadata)
    if dimensionality is None and (carries_quantity or absolute):
        msg = (
            f"is {'a Quantity' if carries_quantity else 'marked ABSOLUTE'} without a dimensionality. Declare one "
            "with `typing.Annotated[Quantity, DIM]`, or write `float` if the port is outside the unit system."
        )
        raise BiltPortDeclarationError(owner_name, port, msg)
    if dimensionality is not None and not (carries_quantity or over_a_domain):
        msg = (
            f"declares {dimensionality} on {getattr(base_type, '__name__', repr(base_type))}, "
            "but a dimensionality can only annotate Quantity or Tensor. "
            "Use `typing.Annotated[Quantity, ...]`, or `typing.Annotated[Tensor, ...]` if the port varies "
            "over a domain."
        )
        raise BiltPortDeclarationError(owner_name, port, msg)

    if dimensionality is None:
        return None

    declaration = PortDeclaration(
        dimensionality,
        absolute=absolute,
        over_a_domain=over_a_domain,
        scalar_only=magnitude_kind == "scalar",
        required_domain=required_domain,
    )
    if absolute:
        try:
            _ = declaration.coherent_unit
        except BiltUnitError as error:
            msg = f"is marked ABSOLUTE, but {dimensionality} cannot be absolute: {error}."
            raise BiltPortDeclarationError(owner_name, port, msg) from error
    return declaration
