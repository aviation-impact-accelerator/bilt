"""Validate supplied values against a system or implementation's port declarations."""

import enum
import typing

import numpy as np

from bilt.core._errors import BiltEvaluationError
from bilt.core._system._errors import BiltInputDimensionalityError
from bilt.core._system._sentinel import MISSING
from bilt.core._tensors._tensors import Tensor
from bilt.symbolic._dimensions import CoordinateAxis
from bilt.units import Quantity
from bilt.units._category import UnitCategory

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._port_declaration import PortDeclaration


def validate_inputs(  # noqa: C901
    system_name: str,
    declarations: collections.abc.Mapping[str, PortDeclaration],
    values: collections.abc.Mapping[str, object],
    *,
    enum_types: collections.abc.Mapping[str, type[enum.Enum]] | None = None,
    enum_tensor_ports: collections.abc.Container[str] = (),
) -> None:
    """Ensure every supplied value satisfies the declaration of its port."""
    enum_types = enum_types or {}
    for port, value in values.items():
        if value is MISSING:
            continue
        enum_type = enum_types.get(port)
        if enum_type is not None:
            if isinstance(value, CoordinateAxis) and port in enum_tensor_ports:
                if not value.coordinates or any(type(member) is not enum_type for member in value.coordinates):
                    msg = (
                        f"Input {port!r} of {system_name!r} requires a CoordinateAxis of {enum_type.__name__} members."
                    )
                    raise BiltEvaluationError(msg)
                continue
            if isinstance(value, Tensor) and port in enum_tensor_ports:
                if (
                    value.unit is not None
                    or value.values.size == 0
                    or value.values.dtype != np.dtype(object)
                    or any(type(member) is not enum_type for member in value.values.flat)
                ):
                    msg = f"Input {port!r} of {system_name!r} requires a Tensor of {enum_type.__name__} members."
                    raise BiltEvaluationError(msg)
                continue
            if type(value) is not enum_type:
                msg = f"Input {port!r} of {system_name!r} requires a {enum_type.__name__} enum member, got {value!r}."
                raise BiltEvaluationError(msg)
            continue
        if isinstance(value, enum.Enum):
            msg = f"Input {port!r} of {system_name!r} does not declare an enum type; got {value!r}."
            raise BiltEvaluationError(msg)

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
            or (expected.scalar_only and isinstance(value, Tensor))
            or (
                expected.required_domain is not None
                and isinstance(value, Tensor)
                and not set(expected.required_domain.dims).issubset(value.domain.dims)
            )
        ):
            raise BiltInputDimensionalityError(system_name, port, expected, value)
