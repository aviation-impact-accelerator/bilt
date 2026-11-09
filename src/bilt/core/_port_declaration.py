"""What a port's annotation declares about the quantity it carries."""

__all__ = ["PortDeclaration", "annotated_parts", "port_declarations"]

import dataclasses
import types
import typing

from bilt.core._errors import BiltPortDeclarationError
from bilt.core._tensors._tensors import QuantityTensor, Tensor
from bilt.symbolic._domains import Domain
from bilt.units import Dimension, Metadata, Quantity, Unit
from bilt.units._errors import BiltUnitError
from bilt.units._quantity_protocol import QuantityLike
from bilt.units._units import BASE_UNIT_SYSTEM
from bilt.units.dimensions import ABSOLUTE

if typing.TYPE_CHECKING:
    import collections.abc


@dataclasses.dataclass
class PortDeclaration:
    """What a declared port holds: a quantity of its dimensionality, absolute if marked and relative otherwise.

    `over_a_domain` records that the port accepts a tensor quantity. It says what a caller may
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
    """Check the exact scalar-or-tensor quantity union, not arbitrary unions."""
    return typing.get_origin(base_type) in (typing.Union, types.UnionType) and frozenset(typing.get_args(base_type)) == {
        Quantity,
        QuantityTensor,
    }


def quantity_magnitude_kind(base_type: object) -> typing.Literal["scalar", "tensor", "either"] | None:
    """Identify a shared quantity contract for scalar/tensor system boundaries.

    Bare QuantityLike is intentionally liftable at system boundaries. Its
    float specialisation means scalar *values* at system boundaries, but a
    scalar-local symbolic expression when used by an equation.
    """
    # A union of the two concrete quantity representations is a flexible
    # pointwise quantity. It does not include plain, unitless Tensor.
    if _is_pointwise_quantity_union(base_type):
        return "either"
    if base_type is QuantityLike:
        return "either"
    if typing.get_origin(base_type) is not QuantityLike:
        return None

    args = typing.get_args(base_type)
    if args == (float,):
        return "scalar"
    if args == (Tensor,):
        return "tensor"
    if args == (object,):
        return "either"
    msg = f"Unsupported QuantityLike magnitude argument {args!r}; use float or Tensor."
    raise TypeError(msg)


def _required_domain(
    owner_name: str,
    port: str,
    metadata: list[object],
    base_type: object,
    dimensionality: Dimension | None,
    *,
    system_boundary: bool,
) -> Domain | None:
    """Read minimum tensor-domain constraints, preserving exact pointwise equation hints."""
    # Exact Domain hints on pointwise equations remain for the linear solver.
    if dimensionality is None or base_type is Quantity or (
        not system_boundary and _is_pointwise_quantity_union(base_type)
    ):
        return None

    domains = [item for item in metadata if isinstance(item, Domain)]
    if len(domains) > 1:
        msg = "declares more than one required domain. Declare at most one Domain in Annotated."
        raise BiltPortDeclarationError(owner_name, port, msg)
    if not domains:
        return None
    if base_type is not QuantityTensor and quantity_magnitude_kind(base_type) != "tensor":
        msg = "declares a required Domain on a scalar or generic quantity; use QuantityTensor or QuantityLike[Tensor]."
        raise BiltPortDeclarationError(owner_name, port, msg)
    return domains[0]


def port_declarations(
    owner_name: str,
    port_annotations: collections.abc.Mapping[str, object],
    *,
    system_boundary: bool = False,
) -> dict[str, PortDeclaration]:
    """Read the value off each port's annotation to keep information about the port."""
    declarations = {
        port: _declaration(owner_name, port, annotation, system_boundary=system_boundary)
        for port, annotation in port_annotations.items()
    }
    return {port: declaration for port, declaration in declarations.items() if declaration is not None}


def _reject_legacy_annotation(
    owner_name: str,
    port: str,
    base_type: object,
    dimensionality: Dimension | None,
    *,
    system_boundary: bool,
) -> None:
    if dimensionality is not None and base_type is Tensor:
        msg = (
            "uses the old dimensioned Tensor annotation. "
            "Use Annotated[QuantityLike[Tensor], DIM] on system fields or "
            "Annotated[QuantityTensor, DIM] in equations and transforms."
        )
        raise BiltPortDeclarationError(owner_name, port, msg)
    if dimensionality is not None and system_boundary and base_type is Quantity:
        msg = (
            "uses the old Quantity system-field annotation. "
            "Use Annotated[QuantityLike, DIM] or Annotated[QuantityLike[float], DIM]."
        )
        raise BiltPortDeclarationError(owner_name, port, msg)


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
            "Multiply the dimensionality by it, as in `typing.Annotated[QuantityLike, DIM * tag]`."
        )
        raise BiltPortDeclarationError(owner_name, port, msg)

    dimensionalities = [item for item in metadata if isinstance(item, Dimension)]
    if len(dimensionalities) > 1:
        msg = f"declares more than one dimensionality ({', '.join(map(str, dimensionalities))}). Declare exactly one."
        raise BiltPortDeclarationError(owner_name, port, msg)

    dimensionality = dimensionalities[0] if dimensionalities else None
    _reject_legacy_annotation(owner_name, port, base_type, dimensionality, system_boundary=system_boundary)

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
            "but a dimensionality requires a quantity annotation. "
            "Use `typing.Annotated[QuantityLike, ...]` on system fields or a concrete "
            "Quantity / QuantityTensor annotation for equations."
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
