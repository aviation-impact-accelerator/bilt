"""Validate supplied values against a system or implementation's port declarations."""

import typing

from bilt.core._system._errors import BiltInputDimensionalityError
from bilt.core._system._sentinel import MISSING
from bilt.core._tensors._tensors import Tensor
from bilt.units import Quantity
from bilt.units._category import UnitCategory

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._port_declaration import PortDeclaration


def validate_inputs(
    system_name: str,
    declarations: collections.abc.Mapping[str, PortDeclaration],
    values: collections.abc.Mapping[str, object],
) -> None:
    """Ensure every supplied value satisfies the declaration of its port."""
    for port, value in values.items():
        if value is MISSING:
            continue

        unit = value.unit if isinstance(value, Quantity | Tensor) else None
        expected = declarations.get(port)
        if expected is None:
            if unit is not None:
                raise BiltInputDimensionalityError(system_name, port, expected, value)
        elif (
            unit is None
            or unit.dimension != expected.dimensionality
            or not (unit.category is UnitCategory.ABSOLUTE if expected.absolute else unit.is_relative)
            or (expected.over_a_domain and not isinstance(value, Tensor))
        ):
            raise BiltInputDimensionalityError(system_name, port, expected, value)
