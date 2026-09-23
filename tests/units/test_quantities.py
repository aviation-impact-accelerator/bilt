import math
import re

import pytest

from bilt.errors import BiltDimensionalityError, BiltUnitError
from bilt.units._metadata import Metadata
from bilt.units._quantities import Quantity
from bilt.units._units import (
    UNITLESS,
    Unit,
    ampere,
    kelvin,
    kilogram,
    meter,
    second,
)
from bilt.units.units import (
    absolute_year,
    celsius,
    degree,
    exajoule,
    fahrenheit,
    fraction,
    gigawatthour,
    gram,
    hour,
    kilometer,
    kilowatt,
    kilowatthour,
    megagram,
    mile,
    newton,
    percent,
    radian,
    watt,
    year,
)

co2 = Metadata("co2")
nox = Metadata("nox")


class TestQuantity:
    @staticmethod
    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        (
            (1.0 * gram, 1.0 * gram, True),
            (1.0 * gram * co2, 1.0 * gram * co2, True),
            (1.0 * abs(celsius), 1.0 * abs(celsius), True),
            (1.0 * gram, 2.0 * gram, False),
        ),
    )
    def test_equality(
        left: Quantity,
        right: Quantity,
        expected: bool,
    ) -> None:
        assert (left == right) is expected

    @staticmethod
    def test_positive_and_rounding() -> None:
        quantity = -2.345 * meter

        assert +quantity == -2.345 * meter
        assert round(quantity, 2) == -2.35 * meter

    @staticmethod
    @pytest.mark.parametrize(
        ("quantity", "expected"),
        (
            (1.0 * meter, -1.0 * meter),
            (-2.0 * kilometer / second, 2.0 * kilometer / second),
        ),
    )
    def test_negation(
        quantity: Quantity,
        expected: Quantity,
    ) -> None:
        assert -quantity == expected

    @staticmethod
    @pytest.mark.parametrize(
        ("quantity", "expected"),
        (
            (1.0 * meter, 1.0 * meter),
            (-2.0 * kilometer / second, 2.0 * kilometer / second),
        ),
    )
    def test_absolute(
        quantity: Quantity,
        expected: Quantity,
    ) -> None:
        assert abs(quantity) == expected

    @staticmethod
    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        (
            (1.0 * meter, 2.0 * meter, 3.0 * meter),
            (
                1.0 * kilogram / meter,
                2.0 * kilogram / meter,
                3.0 * kilogram / meter,
            ),
        ),
    )
    def test_addition(
        left: Quantity,
        right: Quantity,
        expected: Quantity,
    ) -> None:
        assert left + right == expected

    @staticmethod
    def test_sum_uses_reflected_addition() -> None:
        assert sum((1.0 * meter, 2.0 * meter)) == 3.0 * meter

    @staticmethod
    @pytest.mark.parametrize(
        ("left", "right"),
        (
            (1.0 * meter, 1.0 * kilogram),
            (1.0 * watt, 1.0 * newton),
            (
                1.0 * kilogram * meter / second**2,
                1.0 * watt,
            ),
        ),
    )
    def test_addition_dimensionality_error_if_units_dimensionally_inconsistent(
        left: Quantity,
        right: Quantity,
    ) -> None:
        with pytest.raises(
            BiltDimensionalityError,
            match="as they are dimensionally inconsistent",
        ):
            _ = left + right

    @staticmethod
    @pytest.mark.parametrize(
        ("left", "right"),
        (
            (
                1.0 * newton,
                1.0 * kilogram * meter / second**2,
            ),
            (
                1.0 * kilowatt * hour,
                1.0 * kilowatthour,
            ),
        ),
    )
    def test_addition_unit_error_if_units_not_equal(
        left: Quantity,
        right: Quantity,
    ) -> None:
        with pytest.raises(
            BiltUnitError,
            match=re.escape("as the units are inconsistent, use the `Quantity.cast_to()` method"),
        ):
            _ = left + right

    @staticmethod
    def test_addition_unit_error_if_units_not_equivalent() -> None:
        with pytest.raises(
            BiltUnitError,
            match=re.escape("as the units are inconsistent, use the `Quantity.convert_to()` method"),
        ):
            _ = 1.0 * kilometer + 1.0 * mile

    @staticmethod
    @pytest.mark.parametrize(
        ("left", "right"),
        (
            (
                1.0 * kilogram,
                1.0 * kilogram * Metadata("co2"),
            ),
            (
                1.0 * kilogram * Metadata("nox"),
                1.0 * kilogram * Metadata("co2"),
            ),
        ),
    )
    def test_addition_unit_error_if_unit_metadata_not_equal(
        left: Quantity,
        right: Quantity,
    ) -> None:
        with pytest.raises(
            BiltUnitError,
            match="because the units do not have consistent metadata",
        ):
            _ = left + right

    @staticmethod
    def test_addition_rejects_two_absolute_quantities() -> None:
        with pytest.raises(
            BiltUnitError,
            match="Cannot add two quantities with absolute units",
        ):
            _ = 1.0 * abs(celsius) + 1.0 * abs(celsius)

    @staticmethod
    @pytest.mark.parametrize(
        ("left", "right"),
        (
            (1.0, 1.0 * kilogram),
            (1.0 * kilogram, 1.0),
            (1.0, 1.0 * kilogram * Metadata("co2")),
            (1.0 * kilogram * Metadata("co2"), 1.0),
        ),
    )
    def test_addition_type_error_if_other_is_not_quantity(
        left: Quantity,
        right: Quantity,
    ) -> None:
        with pytest.raises(TypeError):
            _ = left + right

    @staticmethod
    def test_addition_rejects_absolute_and_differently_scaled_relative_units() -> None:
        with pytest.raises(
            BiltUnitError,
            match="relative unit",
        ):
            _ = 1.0 * abs(celsius) + 1.0 * fahrenheit

    @staticmethod
    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        (
            (
                20.0 * abs(celsius),
                5.0 * celsius,
                25.0 * abs(celsius),
            ),
            (
                5.0 * celsius,
                20.0 * abs(celsius),
                25.0 * abs(celsius),
            ),
        ),
    )
    def test_addition_absolute_and_relative_returns_absolute_quantity(
        left: Quantity,
        right: Quantity,
        expected: Quantity,
    ) -> None:
        assert left + right == expected

    @staticmethod
    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        (
            (1.0 * meter, 3.0 * meter, -2.0 * meter),
            (
                1.0 * kilogram / meter,
                3.0 * kilogram / meter,
                -2.0 * kilogram / meter,
            ),
            (
                100.0 * abs(kelvin),
                10.0 * abs(kelvin),
                90.0 * kelvin,
            ),
        ),
    )
    def test_subtraction(
        left: Quantity,
        right: Quantity,
        expected: Quantity,
    ) -> None:
        assert left - right == expected

    @staticmethod
    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        (
            (
                Quantity(2.0, second),
                Quantity(2.0, ampere),
                Quantity(4.0, ampere * second),
            ),
            (
                Quantity(2.0, second),
                ampere,
                Quantity(2.0, ampere * second),
            ),
            (
                Quantity(2.0, kilogram),
                co2,
                Quantity(2.0, kilogram * co2),
            ),
            (
                Quantity(2.0, kilogram),
                2.0,
                Quantity(4.0, kilogram),
            ),
            (
                Quantity(2.0, kilogram * co2),
                2.0,
                Quantity(4.0, kilogram * co2),
            ),
        ),
    )
    def test_multiplication_quantity_left(
        left: Quantity,
        right: Unit | Quantity | Metadata | float,
        expected: Quantity,
    ) -> None:
        assert left * right == expected

    @staticmethod
    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        (
            (
                Quantity(2.0, second),
                Quantity(2.0, ampere),
                Quantity(4.0, ampere * second),
            ),
            (
                2.0,
                Quantity(2.0, kilogram),
                Quantity(4.0, kilogram),
            ),
            (
                2.0,
                Quantity(2.0, kilogram * co2),
                Quantity(4.0, kilogram * co2),
            ),
        ),
    )
    def test_multiplication_quantity_right(
        left: Quantity | float,
        right: Quantity,
        expected: Quantity,
    ) -> None:
        assert left * right == expected

    @staticmethod
    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        (
            (
                Quantity(4.0, meter),
                Quantity(2.0, second),
                Quantity(2.0, meter / second),
            ),
            (
                Quantity(2.0, meter / second),
                2.0,
                Quantity(1.0, meter / second),
            ),
            (
                Quantity(2.0, meter),
                second,
                Quantity(2.0, meter / second),
            ),
        ),
    )
    def test_division_quantity_numerator(
        left: Quantity,
        right: Quantity | Unit | float,
        expected: Quantity,
    ) -> None:
        assert left / right == expected

    @staticmethod
    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        (
            (
                Quantity(4.0, meter),
                Quantity(2.0, second),
                Quantity(2.0, meter / second),
            ),
            (
                1.0,
                Quantity(2.0, second),
                Quantity(0.5, 1 / second),
            ),
            (
                meter,
                Quantity(2.0, second),
                Quantity(0.5, meter / second),
            ),
        ),
    )
    def test_division_quantity_denominator(
        left: Quantity | Unit | float,
        right: Quantity,
        expected: Quantity,
    ) -> None:
        assert left / right == expected

    @staticmethod
    @pytest.mark.parametrize(
        ("quantity", "exponent", "expected"),
        (
            (
                4.7 * UNITLESS,
                0,
                1.0 * UNITLESS,
            ),
            (
                3.4 * UNITLESS,
                1,
                3.4 * UNITLESS,
            ),
            (
                2.0 * UNITLESS,
                -3,
                0.125 * UNITLESS,
            ),
            (
                4.7 * UNITLESS,
                0.0,
                1.0 * UNITLESS,
            ),
            (
                9.0 * UNITLESS,
                1.5,
                27.0 * UNITLESS,
            ),
            (
                16.0 * UNITLESS,
                -0.5,
                0.25 * UNITLESS,
            ),
        ),
    )
    def test_exponentiation_unitless_unit(
        quantity: Quantity,
        exponent: float,
        expected: Quantity,
    ) -> None:
        assert quantity**exponent == expected

    @staticmethod
    @pytest.mark.parametrize(
        ("quantity", "exponent", "expected"),
        (
            (
                1.5 * kilogram,
                0,
                1.0 * UNITLESS,
            ),
            (
                2.5 * kilogram * meter / second**2,
                1,
                2.5 * kilogram * meter / second**2,
            ),
            (
                -2.0 * kilowatthour,
                -2,
                0.25 * (1 / kilowatthour**2),
            ),
        ),
    )
    def test_exponentiation_int_exponent(
        quantity: Quantity,
        exponent: int,
        expected: Quantity,
    ) -> None:
        assert quantity**exponent == expected

    @staticmethod
    @pytest.mark.parametrize(
        ("quantity", "exponent"),
        (
            (1.0 * kilogram, 0.0),
            (2.5 * kilogram * meter / second**2, 1.0),
            (-2.35 * kilowatthour, -2.5),
        ),
    )
    def test_exponentiation_float_exponent_raises_type_error(
        quantity: Quantity,
        exponent: float,
    ) -> None:
        message = re.escape("unsupported operand type(s) for ** or pow(): 'Quantity' and 'float'")

        with pytest.raises(TypeError, match=message):
            _ = quantity**exponent

    @staticmethod
    @pytest.mark.parametrize(
        ("base", "quantity", "expected"),
        (
            (
                2.0,
                0.0 * UNITLESS,
                1.0 * UNITLESS,
            ),
            (
                2.0,
                1.0 * UNITLESS,
                2.0 * UNITLESS,
            ),
            (
                2.0,
                2.0 * UNITLESS,
                4.0 * UNITLESS,
            ),
            (
                4.0,
                0.5 * UNITLESS,
                2.0 * UNITLESS,
            ),
        ),
    )
    def test_exponentiation_float_base(
        base: float,
        quantity: Quantity,
        expected: Quantity,
    ) -> None:
        assert base**quantity == expected

    @staticmethod
    @pytest.mark.parametrize(
        ("base", "quantity"),
        (
            (2.0, 1.0 * percent),
            (2.0, 1.0 * meter),
        ),
    )
    def test_exponentiation_float_base_non_dimensionless_quantity_raises_type_error(
        base: float,
        quantity: Quantity,
    ) -> None:
        message = re.escape("unsupported operand type(s) for ** or pow(): 'float' and 'Quantity'")

        with pytest.raises(TypeError, match=message):
            _ = base**quantity

    @staticmethod
    def test_comparison() -> None:
        assert 1.0 * meter < 2.0 * meter

        with pytest.raises(BiltUnitError, match="Cannot compare quantity"):
            _ = 1.0 * meter < 1.0 * kilometer

    @staticmethod
    @pytest.mark.parametrize(
        ("quantity", "unit", "expected"),
        (
            (
                1.0 * kilogram,
                kilogram,
                1.0 * kilogram,
            ),
            (
                1.0 * kilogram * meter / second**2,
                newton,
                1.0 * newton,
            ),
            (
                1.0 * newton,
                kilogram * meter / second**2,
                1.0 * kilogram * meter / second**2,
            ),
            (
                1.0 * newton * meter / second,
                watt,
                1.0 * watt,
            ),
            (
                1.0 * kilowatthour,
                kilowatt * hour,
                1.0 * kilowatt * hour,
            ),
            (
                1.0 * kilowatt * hour,
                kilowatthour,
                1.0 * kilowatthour,
            ),
            (
                1.0 * kilowatthour / kilogram,
                kilowatt * hour / kilogram,
                1.0 * kilowatt * hour / kilogram,
            ),
            (
                1.0 * kilowatt * hour / kilogram,
                kilowatthour / kilogram,
                1.0 * kilowatthour / kilogram,
            ),
            (
                1.0 * radian / second,
                1 / second,
                1.0 * (1 / second),
            ),
            (
                1.0 * (fraction * Metadata("mass")) * kilogram / second,
                kilogram / second,
                1.0 * kilogram / second,
            ),
        ),
    )
    def test_quantity_cast_to(
        quantity: Quantity,
        unit: Unit,
        expected: Quantity,
    ) -> None:
        result = quantity.cast_to(unit)

        assert result.value == expected.value
        assert result.unit == expected.unit

    @staticmethod
    @pytest.mark.parametrize(
        ("quantity", "unit"),
        (
            (1.0 * meter, kilogram),
            (1.0 * gram, kilogram),
            (1.0 * kilogram, gram),
            (1.0 * degree / second, 1 / second),
            (
                1.0 * (percent * Metadata("mass")) * kilogram / second,
                kilogram / second,
            ),
        ),
    )
    def test_quantity_cast_to_unit_error_if_units_not_equivalent(
        quantity: Quantity,
        unit: Unit,
    ) -> None:
        with pytest.raises(BiltUnitError, match="Cannot cast quantity"):
            quantity.cast_to(unit)

    @staticmethod
    @pytest.mark.parametrize(
        ("quantity", "unit", "expected"),
        (
            (
                1.0 * kilogram,
                kilogram,
                1.0 * kilogram,
            ),
            (
                1.0 * kilogram,
                gram,
                1_000.0 * gram,
            ),
            (
                1.0 * kilogram,
                megagram,
                0.001 * megagram,
            ),
            (
                1.0 * gigawatthour,
                exajoule,
                3.6e-6 * exajoule,
            ),
            (
                1.0 * year,
                second,
                365.25 * 24.0 * 60.0 * 60.0 * second,
            ),
            (
                1.0 * radian,
                degree,
                180.0 / math.pi * degree,
            ),
            (
                1.0 * degree,
                radian,
                math.pi / 180.0 * radian,
            ),
            (
                1.0 * percent,
                fraction,
                0.01 * fraction,
            ),
            (
                1.0 * fraction,
                percent,
                100.0 * percent,
            ),
        ),
    )
    def test_quantity_convert_to(
        quantity: Quantity,
        unit: Unit,
        expected: Quantity,
    ) -> None:
        result = quantity.convert_to(unit)

        assert result.value == pytest.approx(expected.value)
        assert result.unit == expected.unit

    @staticmethod
    def test_quantity_convert_absolute() -> None:
        result = (0.0 * abs(celsius)).convert_to(abs(kelvin))

        assert result.value == pytest.approx(273.15)
        assert result.unit == abs(kelvin)

    @staticmethod
    def test_quantity_convert_same_absolute_unit_returns_self() -> None:
        quantity = 273.15 * abs(kelvin)

        assert quantity.convert_to(quantity.unit) is quantity

    @staticmethod
    @pytest.mark.parametrize(
        ("quantity", "unit"),
        (
            (1.0 * meter, kilogram),
            (1.0 * watt, newton),
        ),
    )
    def test_quantity_convert_to_dimensionality_error_if_units_not_dimensionally_consistent(
        quantity: Quantity,
        unit: Unit,
    ) -> None:
        with pytest.raises(
            BiltDimensionalityError,
            match="Cannot convert",
        ):
            quantity.convert_to(unit)

    @staticmethod
    def test_quantity_convert_to_unit_error_if_units_have_different_metadata() -> None:
        with pytest.raises(BiltUnitError, match="Cannot convert"):
            (1.0 * kilogram * co2).convert_to(kilogram * nox)

    @staticmethod
    def test_quantity_convert_to_unit_error_if_converting_absolute_to_relative() -> None:
        with pytest.raises(BiltUnitError, match="Cannot convert"):
            (1.0 * absolute_year).convert_to(second)

    @staticmethod
    @pytest.mark.parametrize(
        ("quantity", "expected"),
        (
            (
                1.23 * kilogram * Metadata("carbon dioxide"),
                1.23 * kilogram,
            ),
            (
                -4.56 * kilogram * meter / second**2 * Metadata("vertical"),
                -4.56 * kilogram * meter / second**2,
            ),
            (
                10_000.0 * kilowatthour * Metadata("thermal"),
                10_000.0 * kilowatthour,
            ),
        ),
    )
    def test_strip_metadata(
        quantity: Quantity,
        expected: Quantity,
    ) -> None:
        assert quantity.strip_metadata() == expected
