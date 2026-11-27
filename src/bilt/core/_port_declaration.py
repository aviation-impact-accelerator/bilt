"""What a port's annotation declares about the quantity it carries."""

__all__ = ["PortDeclaration", "annotated_parts", "port_declarations"]

import dataclasses
import types
import typing

from bilt.core._errors import BiltPortDeclarationError
from bilt.core._tensors._protocols import QuantityTensor, Tensor
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

    `over_a_domain` records that the port requires a tensor quantity (`QuantityTensor`); otherwise it takes a
    quantity of any shape (`Quantity`). It says what a caller may pass at this port, not what the port measures, so
    it takes no part in a declaration's identity: a constituent written over scalars is reusable in a tensor system
    under the same port name.
    """

    dimensionality: Dimension
    absolute: bool = False
    over_a_domain: bool = dataclasses.field(default=False, compare=False)
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
    metadata: list[object] = []
    seen: set[int] = set()
    while id(annotation) not in seen:
        seen.add(id(annotation))
        if isinstance(annotation, typing.TypeAliasType):
            annotation = annotation.__value__
        elif typing.get_origin(annotation) is typing.Annotated:
            annotation, *items = typing.get_args(annotation)
            metadata[:0] = items
        else:
            break
    return annotation, metadata


def _is_old_pointwise_quantity_union(base_type: object) -> bool:
    """Whether this is `Quantity | QuantityTensor`, which means the same as `Quantity`."""
    return typing.get_origin(base_type) in (typing.Union, types.UnionType) and frozenset(
        typing.get_args(base_type)
    ) == {
        Quantity,
        QuantityTensor,
    }


def quantity_magnitude_kind(base_type: object) -> typing.Literal["tensor", "any"] | None:
    """Classify a quantity annotation by the shape it accepts: `QuantityTensor` a tensor, `Quantity` any shape."""
    if base_type is Quantity or _is_old_pointwise_quantity_union(base_type):
        return "any"
    if base_type is QuantityTensor:
        return "tensor"
    return None


_DECLARING_METADATA = (Dimension, Domain, Metadata, Unit)


def _hides_a_declaration(annotation: object, seen: frozenset[int] = frozenset()) -> bool:
    """Whether a dimensionality, `ABSOLUTE` or required domain sits somewhere Bilt does not read it.

    Bilt reads `Annotated` metadata only at the top of a port's annotation (through plain type aliases).
    Metadata inside a union, a generic or a subscripted generic type alias would otherwise be dropped silently.
    """
    if id(annotation) in seen:
        return False
    seen = seen | {id(annotation)}
    if isinstance(annotation, typing.TypeAliasType):
        return _hides_a_declaration(annotation.__value__, seen)
    origin = typing.get_origin(annotation)
    if isinstance(origin, typing.TypeAliasType):
        return _hides_a_declaration(origin.__value__, seen) or any(
            _hides_a_declaration(item, seen) for item in typing.get_args(annotation)
        )
    if origin is typing.Annotated:
        base_type, *metadata = typing.get_args(annotation)
        return any(isinstance(item, _DECLARING_METADATA) or item is ABSOLUTE for item in metadata) or (
            _hides_a_declaration(base_type, seen)
        )
    return any(_hides_a_declaration(item, seen) for item in typing.get_args(annotation))


def _required_domain(  # noqa: PLR0913
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
    if dimensionality is None or (not system_boundary and quantity_magnitude_kind(base_type) == "any"):
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


def _reject_dimensioned_tensor_annotation(
    owner_name: str,
    port: str,
    base_type: object,
    dimensionality: Dimension | None,
) -> None:
    if dimensionality is not None and base_type is Tensor:
        msg = (
            "declares a dimensionality on Tensor, which has no unit. "
            "Use Annotated[QuantityTensor, DIM] for a tensor-only port or "
            "Annotated[Quantity, DIM] for a port of any shape."
        )
        raise BiltPortDeclarationError(owner_name, port, msg)


def _declaration(
    owner_name: str, port: str, annotation: object, *, system_boundary: bool = False
) -> PortDeclaration | None:
    """Classify one annotation, ignoring `Annotated` metadata that is neither a dimensionality nor the absolute marker."""
    base_type, metadata = annotated_parts(annotation)
    if _hides_a_declaration(base_type):
        msg = (
            f"is annotated `{annotation}`, where Bilt cannot read its dimensionality. Put `Annotated` at the top "
            "of the annotation, as in `typing.Annotated[Quantity, DIM]`, directly or through a type alias without "
            "type parameters. Unions such as `Annotated[...] | None` and subscripted generic aliases such as "
            "`Length[Quantity]` are not supported."
        )
        raise BiltPortDeclarationError(owner_name, port, msg)
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
    _reject_dimensioned_tensor_annotation(owner_name, port, base_type, dimensionality)

    magnitude_kind = quantity_magnitude_kind(base_type)
    carries_quantity = magnitude_kind is not None
    over_a_domain = base_type in (Tensor, QuantityTensor) or magnitude_kind == "tensor"
    required_domain = _required_domain(
        owner_name, port, metadata, base_type, dimensionality, system_boundary=system_boundary
    )
    absolute = any(item is ABSOLUTE for item in metadata)
    if dimensionality is None and (carries_quantity or absolute):
        msg = (
            f"is {'a Quantity' if carries_quantity else 'marked ABSOLUTE'} without a dimensionality. Declare one "
            "with `typing.Annotated[Quantity, DIM]`, or `typing.Annotated[QuantityTensor, DIM]` for a tensor-only "
            "port; write `Number` (or `float`) if the port is outside the unit system."
        )
        raise BiltPortDeclarationError(owner_name, port, msg)
    if dimensionality is not None and not (carries_quantity or over_a_domain):
        msg = (
            f"declares {dimensionality} on {getattr(base_type, '__name__', repr(base_type))}, "
            "but a dimensionality requires a quantity annotation. "
            "Use `typing.Annotated[Quantity, ...]` for any shape or `typing.Annotated[QuantityTensor, ...]` "
            "for a tensor-only port."
        )
        raise BiltPortDeclarationError(owner_name, port, msg)

    if dimensionality is None:
        return None

    declaration = PortDeclaration(
        dimensionality,
        absolute=absolute,
        over_a_domain=over_a_domain,
        required_domain=required_domain,
    )
    if absolute:
        try:
            _ = declaration.coherent_unit
        except BiltUnitError as error:
            msg = f"is marked ABSOLUTE, but {dimensionality} cannot be absolute: {error}."
            raise BiltPortDeclarationError(owner_name, port, msg) from error
    return declaration
