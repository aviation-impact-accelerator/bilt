"""Check dimensionality across an implementation's interface and constituents by shared port name."""

__all__ = ["check_categorical_consistency", "check_dimensional_consistency"]

import typing

from bilt.core._errors import BiltCategoricalConsistencyError, BiltImplementationDimensionalityError
from bilt.core._transform import BasicTransform
from bilt.core._validation import validate_independently

if typing.TYPE_CHECKING:
    import collections.abc
    import enum

    from bilt.core._interface import Interface
    from bilt.core._model_constituent import BiltModelConstituent


def check_dimensional_consistency(
    system: Interface,
    constituents: collections.abc.Iterable[BiltModelConstituent],
) -> None:
    """Demand that every owner of a port agrees on its declaration, declaring it or leaving it plain together.

    The owners are the system interface and each constituent, and a port is named by an owner when it is one of
    its parameters, one it declares, or, for a transform, its output.

    Raises:
        BiltImplementationDimensionalityError: Owners declare a port differently, or only some of them declare it.
    """
    owners = (system, *sorted(constituents, key=lambda constituent: constituent.__bilt_name__))
    named = {owner: {*owner.__bilt_parameters__, *owner.__bilt_port_declarations__} for owner in owners}
    for owner in owners:
        if isinstance(owner, BasicTransform):
            named[owner].add(owner.__bilt_name__)
        named[owner].update(getattr(owner, "__bilt_enum_ports__", {}))
    validate_independently(
        lambda port=port: _check_port(system, port, owners, named) for port in sorted(set().union(*named.values()))
    )


def _check_port(
    system: Interface,
    port: str,
    owners: tuple[Interface | BiltModelConstituent, ...],
    named: collections.abc.Mapping[Interface | BiltModelConstituent, set[str]],
) -> None:
    """Validate a single port independently of the declarations of other ports."""
    naming = [owner for owner in owners if port in named[owner]]
    declaring = [owner for owner in naming if port in owner.__bilt_port_declarations__]
    enum_declarations = tuple(
        (owner.__bilt_name__, getattr(owner, "__bilt_enum_ports__", {}).get(port)) for owner in naming
    )
    if check_categorical_consistency(system.__bilt_name__, port, enum_declarations):
        if declaring:
            msg = f"gives categorical port {port!r} a physical dimensionality."
            raise BiltCategoricalConsistencyError(system.__bilt_name__, msg)
        return
    if not declaring:
        return

    first = declaring[0]
    declaration = first.__bilt_port_declarations__[port]
    for owner in declaring[1:]:
        other = owner.__bilt_port_declarations__[port]
        if other != declaration:
            msg = (
                f"gives port {port!r} conflicting declarations:\n"
                f"  {declaration}: declared by {first.__bilt_name__}\n"
                f"  {other}: declared by {owner.__bilt_name__}"
            )
            raise BiltImplementationDimensionalityError(system.__bilt_name__, msg)

    plain = [owner.__bilt_name__ for owner in naming if port not in owner.__bilt_port_declarations__]
    if plain:
        msg = (
            f"gives port {port!r} a dimensionality in only some places:\n"
            f"  {declaration}: declared by {first.__bilt_name__}\n"
            f"  left plain by {', '.join(sorted(plain))}"
        )
        raise BiltImplementationDimensionalityError(system.__bilt_name__, msg)


def check_categorical_consistency(
    system: str, port: str, declarations: tuple[tuple[str, type[enum.Enum] | None], ...]
) -> bool:
    """Return whether the port is categorical, requiring all owners to declare the same enum class."""
    expected = next((enum_type for _, enum_type in declarations if enum_type is not None), None)
    if expected is None:
        return False
    if any(enum_type is not expected for _, enum_type in declarations):
        details = ", ".join(
            f"{owner!r} declares {enum_type.__qualname__ if enum_type is not None else 'a non-enum port'}"
            for owner, enum_type in declarations
        )
        msg = f"gives categorical port {port!r} inconsistent enum types: {details}."
        raise BiltCategoricalConsistencyError(system, msg)
    return True
