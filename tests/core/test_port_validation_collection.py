"""Collect independent annotation errors without changing normal construction."""

from typing import Annotated

import pytest

from bilt import Interface, equation, field, system, transform
from bilt.core._validation import collecting_validation_errors
from bilt.errors import BiltCollectedValidationError, BiltPortDeclarationError
from bilt.units import Quantity  # noqa: TC001 - Bilt evaluates annotations when decorating models
from bilt.units.dimensions import LENGTH

EXPECTED_ERRORS = 2


def test_system_annotations_still_fail_fast_and_leave_no_partial_metadata() -> None:
    class Broken(metaclass=Interface):
        first: Quantity = field()
        second: Annotated[float, LENGTH] = field()

    with pytest.raises(BiltPortDeclarationError, match="Port 'first'"):
        system(Broken)

    assert "__bilt_system_attributes__" not in vars(Broken)
    assert "__bilt_parameters__" not in vars(Broken)
    assert "__bilt_port_declarations__" not in vars(Broken)


def test_system_annotations_collect_independent_errors() -> None:
    class Broken(metaclass=Interface):
        first: Quantity = field()
        second: Annotated[float, LENGTH] = field()

    with collecting_validation_errors(), pytest.raises(BiltCollectedValidationError) as raised:
        system(Broken)

    errors = raised.value.errors
    assert len(errors) == EXPECTED_ERRORS
    assert all(isinstance(error, BiltPortDeclarationError) for error in errors)
    assert "Port 'first'" in str(errors[0])
    assert "Port 'second'" in str(errors[1])
    assert "__bilt_port_declarations__" not in vars(Broken)


def test_transform_annotations_collect_independent_errors() -> None:
    def broken(first: Quantity, second: Annotated[float, LENGTH]) -> float:
        return 1.0 if first is second else 0.0

    with collecting_validation_errors(), pytest.raises(BiltCollectedValidationError) as raised:
        transform(broken)

    assert len(raised.value.errors) == EXPECTED_ERRORS
    assert all(isinstance(error, BiltPortDeclarationError) for error in raised.value.errors)


def test_equation_annotations_collect_independent_errors() -> None:
    def broken(first: Quantity, second: Annotated[float, LENGTH]) -> bool:
        return first == second

    with collecting_validation_errors(), pytest.raises(BiltCollectedValidationError) as raised:
        equation(broken)

    assert len(raised.value.errors) == EXPECTED_ERRORS
    assert all(isinstance(error, BiltPortDeclarationError) for error in raised.value.errors)
