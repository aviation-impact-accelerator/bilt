import re

import pytest

from bilt.errors import BiltDimensionalityError, BiltUnitError
from bilt.units._metadata import Metadata
from bilt.units._units import kilogram, meter
from bilt.units._validation import (
    validate_addition,
    validate_cast,
    validate_comparison,
)
from bilt.units.units import celsius, gram, kilometer, mile, newton, second


def test_validate_cast_accepts_same_unit() -> None:
    validate_cast(1.0 * kilogram, kilogram)


def test_validate_cast_accepts_equivalent_unit_after_stripping_metadata() -> None:
    validate_cast(
        1.0 * kilogram * Metadata("mass"),
        kilogram,
    )


def test_validate_cast_rejects_non_equivalent_unit() -> None:
    with pytest.raises(BiltUnitError, match="Cannot cast quantity"):
        validate_cast(1.0 * gram, kilogram)


def test_validate_addition_accepts_same_unit() -> None:
    validate_addition(
        1.0 * meter,
        2.0 * meter,
    )


def test_validate_addition_accepts_absolute_and_relative_units() -> None:
    validate_addition(
        1.0 * abs(celsius),
        1.0 * celsius,
    )


def test_validate_addition_rejects_two_absolute_units() -> None:
    with pytest.raises(
        BiltUnitError,
        match="Cannot add two quantities with absolute units",
    ):
        validate_addition(
            1.0 * abs(celsius),
            2.0 * abs(celsius),
        )


def test_validate_addition_rejects_incompatible_dimensions() -> None:
    with pytest.raises(
        BiltDimensionalityError,
        match="dimensionally inconsistent",
    ):
        validate_addition(
            1.0 * meter,
            1.0 * second,
        )


def test_validate_addition_rejects_inconsistent_metadata() -> None:
    with pytest.raises(
        BiltDimensionalityError,
        match="dimensionally inconsistent",
    ):
        validate_addition(
            1.0 * kilogram * Metadata("co2"),
            1.0 * kilogram * Metadata("nox"),
        )


def test_validate_addition_requests_conversion_for_scaled_units() -> None:
    with pytest.raises(
        BiltUnitError,
        match=re.escape("Quantity.convert_to()"),
    ):
        validate_addition(
            1.0 * kilometer,
            1.0 * mile,
        )


def test_validate_addition_requests_cast_for_equivalent_units() -> None:
    with pytest.raises(
        BiltUnitError,
        match=re.escape("Quantity.cast_to()"),
    ):
        validate_addition(
            1.0 * newton,
            1.0 * kilogram * meter / second**2,
        )


def test_validate_comparison_accepts_identical_units() -> None:
    validate_comparison(
        1.0 * meter,
        2.0 * meter,
    )


def test_validate_comparison_rejects_different_units() -> None:
    with pytest.raises(BiltUnitError, match="Cannot compare quantity"):
        validate_comparison(
            1.0 * meter,
            1.0 * kilometer,
        )
