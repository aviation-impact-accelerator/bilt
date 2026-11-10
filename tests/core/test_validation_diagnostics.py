"""Independent definition checks collect errors only when explicitly requested."""

import typing

import pytest

from bilt import Implementation, Interface, equation, field, system, transform
from bilt.core._validation import collecting_validation_errors
from bilt.errors import (
    BiltCollectedValidationError,
    BiltDefinitionError,
    BiltEquationDimensionalityError,
    BiltImplementationDimensionalityError,
)
from bilt.units import Quantity
from bilt.units.dimensions import ABSOLUTE, LENGTH, MASS, TEMPERATURE, TIME

Length = typing.Annotated[Quantity, LENGTH]
Mass = typing.Annotated[Quantity, MASS]
Time = typing.Annotated[Quantity, TIME]
AbsoluteTemperature = typing.Annotated[Quantity, TEMPERATURE, ABSOLUTE]
EXPECTED_ERRORS = 2


@system
class Model(metaclass=Interface):
    a: Length = field()
    b: Length = field()


@transform
def a() -> Mass:
    return 0  # pyrefly: ignore[bad-return]  # ty: ignore[invalid-return-type]


@transform
def b() -> Time:
    return 0  # pyrefly: ignore[bad-return]  # ty: ignore[invalid-return-type]


def test_default_construction_still_raises_the_first_error() -> None:
    with pytest.raises(BiltImplementationDimensionalityError, match="port 'a'"):
        Implementation(Model, [a, b])


def test_checker_collects_independent_port_conflicts() -> None:
    with collecting_validation_errors(), pytest.raises(BiltCollectedValidationError) as raised:
        Implementation(Model, [a, b])

    assert isinstance(raised.value, BiltDefinitionError)
    errors = raised.value.errors
    assert len(errors) == EXPECTED_ERRORS
    assert all(isinstance(error, BiltImplementationDimensionalityError) for error in errors)
    assert "port 'a'" in str(errors[0])
    assert "port 'b'" in str(errors[1])
    assert not Model.__bilt_system_registry__


def test_collection_mode_is_restored_after_failure() -> None:
    with collecting_validation_errors(), pytest.raises(BiltCollectedValidationError):
        Implementation(Model, [a, b])

    with pytest.raises(BiltImplementationDimensionalityError):
        Implementation(Model, [a, b])


def test_equation_checks_collect_independent_errors() -> None:
    def wrong(out: AbsoluteTemperature, hot: AbsoluteTemperature, cold: AbsoluteTemperature, mass: Mass) -> bool:
        return out == hot + cold + mass

    with collecting_validation_errors(), pytest.raises(BiltCollectedValidationError) as raised:
        equation(wrong)

    assert len(raised.value.errors) == EXPECTED_ERRORS
    assert all(isinstance(error, BiltEquationDimensionalityError) for error in raised.value.errors)


def test_equation_still_raises_first_error_normally() -> None:
    def wrong(out: AbsoluteTemperature, hot: AbsoluteTemperature, cold: AbsoluteTemperature, mass: Mass) -> bool:
        return out == hot + cold + mass

    with pytest.raises(BiltEquationDimensionalityError):
        equation(wrong)
