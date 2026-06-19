import math
import re
import typing

import pytest

from bilt.units import (
    DIMENSIONLESS,
    USD,
    absolute_celsius,
    absolute_fahrenheit,
    absolute_kelvin,
    acre,
    ampere,
    candela,
    celsius,
    fahrenheit,
    foot,
    fraction,
    gram,
    hectare,
    hour,
    joule,
    kelvin,
    kilogram,
    kilojoule,
    kilometer,
    kilowatt,
    kilowatthour,
    megagram,
    meter,
    mile,
    milligram,
    millimeter,
    millisecond,
    minute,
    mole,
    newton,
    percent,
    radian,
    second,
    watt,
    watthour,
    yard,
)
from bilt.units._dimensions import LENGTH, TEMPERATURE, Dimension
from bilt.units._errors import UnitError
from bilt.units._metadata import Metadata
from bilt.units._prefix import Prefix
from bilt.units._units import Unit, UnitComposition, UnitOrder

if typing.TYPE_CHECKING:
    from bilt.units._quantities import Quantity

electric = Metadata("electric")
thermal = Metadata("thermal")

co2 = Metadata("co2")
nox = Metadata("nox")


class TestUnitOrder:
    @staticmethod
    def test_caching() -> None:
        assert UnitOrder(meter, -1) is UnitOrder(meter, -1)
        assert UnitOrder(watt, 3) is UnitOrder(watt, 3)

    @staticmethod
    @pytest.mark.parametrize(
        ("unit_order", "expected_unit", "expected_order", "expected_dimension"),
        (
            (UnitOrder(meter, 1), meter, 1, LENGTH),
            (UnitOrder(watt, -2), watt, -2, Dimension(mass=-2, length=-4, time=6)),
        ),
    )
    def test_properties(
        unit_order: UnitOrder,
        expected_unit: Unit,
        expected_order: int,
        expected_dimension: Dimension,
    ) -> None:
        assert unit_order.unit is expected_unit
        assert unit_order.order == expected_order
        assert unit_order.dimension is expected_dimension

    @staticmethod
    @pytest.mark.parametrize(
        ("left", "right", "is_less_than"),
        (
            (UnitOrder(meter, 1), UnitOrder(second, 1), True),
            (UnitOrder(second, 1), UnitOrder(meter, 2), False),
            (UnitOrder(meter, 1), UnitOrder(meter, 2), True),
            (UnitOrder(meter, 3), UnitOrder(meter, 2), False),
            (UnitOrder(kilogram, 1), UnitOrder(kilogram, 1), False),
            (UnitOrder(kilogram, 1), UnitOrder(kilogram * nox, 1), True),
            (UnitOrder(kilogram * co2, 1), UnitOrder(kilogram, 1), False),
            (UnitOrder(kilogram * co2, 1), UnitOrder(kilogram * nox, 1), True),
            (UnitOrder(kilogram * nox, 1), UnitOrder(kilogram * co2, 1), False),
        ),
    )
    def test_ordering(left: UnitOrder, right: UnitOrder, is_less_than: bool) -> None:
        assert (left < right) is is_less_than

    @staticmethod
    @pytest.mark.parametrize(
        ("unit_order", "expected_str", "expected_repr"),
        (
            (UnitOrder(meter, 1), "meter", "UnitOrder(unit=meter, order=1)"),
            (UnitOrder(second, -2), "second^(-2)", "UnitOrder(unit=second, order=-2)"),
            (UnitOrder(kilogram * co2, 1), "kilogram|co2", "UnitOrder(unit=kilogram|co2, order=1)"),
            (
                UnitOrder(kilogram * co2, -1),
                "kilogram|co2^(-1)",
                "UnitOrder(unit=kilogram|co2, order=-1)",
            ),
        ),
    )
    def test_representation(unit_order: Metadata, expected_str: str, expected_repr: str) -> None:
        assert repr(unit_order) == expected_repr
        assert str(unit_order) == expected_str


class TestUnitComposition:
    @staticmethod
    @pytest.mark.parametrize(
        ("left", "right"),
        (
            (UnitComposition(UnitOrder(kilogram, 1)), UnitComposition(UnitOrder(kilogram, 1))),
            (UnitComposition(UnitOrder(meter, 1)), UnitComposition(UnitOrder(meter, 1))),
            (UnitComposition(UnitOrder(second, 1)), UnitComposition(UnitOrder(second, 1))),
            (UnitComposition(UnitOrder(kelvin, 1)), UnitComposition(UnitOrder(kelvin, 1))),
            (
                UnitComposition(UnitOrder(meter, 1), UnitOrder(second, -1)),
                UnitComposition(UnitOrder(second, -1), UnitOrder(meter, 1)),
            ),
            (
                UnitComposition(UnitOrder(kilogram, 1), UnitOrder(meter, 1), UnitOrder(second, -2)),
                UnitComposition(UnitOrder(meter, 1), UnitOrder(second, -2), UnitOrder(kilogram, 1)),
            ),
            (UnitComposition(UnitOrder(newton, 1)), UnitComposition(UnitOrder(newton, 1))),
        ),
    )
    def test_caching(left: UnitComposition, right: UnitComposition) -> None:
        assert left is right

    @staticmethod
    @pytest.mark.parametrize(
        ("composition", "expected"),
        (
            (UnitComposition(UnitOrder(DIMENSIONLESS, 1)), UnitComposition()),
            (
                UnitComposition(UnitOrder(DIMENSIONLESS, 1), UnitOrder(fraction, 1)),
                UnitComposition(UnitOrder(fraction, 1)),
            ),
            (
                UnitComposition(UnitOrder(DIMENSIONLESS, 1), UnitOrder(fraction * Metadata("mass"), 1)),
                UnitComposition(UnitOrder(fraction * Metadata("mass"), 1)),
            ),
        ),
    )
    def test_remove_dimensionless(composition: UnitComposition, expected: UnitComposition) -> None:
        assert composition is expected

    @staticmethod
    @pytest.mark.parametrize("indices", ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)))
    @pytest.mark.parametrize(
        "terms",
        (
            (UnitOrder(kilogram, 1), UnitOrder(meter, 1), UnitOrder(second, -2)),
            (UnitOrder(kilogram, 1), UnitOrder(kilogram * co2, 1), UnitOrder(kilogram * nox, 1)),
        ),
    )
    def test_composition_sorting(terms: tuple[UnitOrder, UnitOrder, UnitOrder], indices: tuple[int, int, int]) -> None:
        unit_composition = UnitComposition(terms[indices[0]], terms[indices[1]], terms[indices[2]])
        assert unit_composition.composition == terms

    @staticmethod
    @pytest.mark.parametrize(
        ("terms", "expected"),
        (
            ((), ()),
            ((UnitOrder(meter, -1), UnitOrder(meter, 1)), ()),
            ((UnitOrder(meter, 1), UnitOrder(meter, 2)), (UnitOrder(meter, 3),)),
            (
                (UnitOrder(meter, 1), UnitOrder(second, -2), UnitOrder(meter, 2)),
                (UnitOrder(meter, 3), UnitOrder(second, -2)),
            ),
        ),
    )
    def test_unit_order_folding(terms: tuple[UnitOrder, ...], expected: tuple[UnitOrder, ...]) -> None:
        instance = UnitComposition(*terms)
        assert instance.composition == expected

    @staticmethod
    @pytest.mark.parametrize("scale", (1.0, 1_609.344))
    def test_scale(scale: float) -> None:
        instance = UnitComposition(UnitOrder(meter, 1), scale=scale)
        assert instance.scale == scale

    @staticmethod
    @pytest.mark.parametrize(
        ("instance", "expected"),
        (
            (UnitComposition(UnitOrder(meter, 1), UnitOrder(meter, -1)), Dimension()),
            (
                UnitComposition(UnitOrder(kilogram, 1), UnitOrder(meter, 1), UnitOrder(second, -2)),
                Dimension(mass=1, length=1, time=-2),
            ),
            (UnitComposition(UnitOrder(watt, 1)), Dimension(mass=1, length=2, time=-3)),
        ),
    )
    def test_dimension(instance: UnitComposition, expected: Dimension) -> None:
        assert instance.dimension == expected

    @staticmethod
    @pytest.mark.parametrize(
        ("instance", "expected"),
        (
            (UnitComposition(), UnitComposition()),
            (UnitComposition(UnitOrder(kilogram, 1)), UnitComposition(UnitOrder(kilogram, 1))),
            (
                UnitComposition(UnitOrder(kilogram, 1), scale=1_000.0),
                UnitComposition(UnitOrder(kilogram, 1), scale=1_000.0),
            ),
            (
                UnitComposition(UnitOrder(kilogram, 1), UnitOrder(meter, 1), UnitOrder(second, -2)),
                UnitComposition(UnitOrder(kilogram, 1), UnitOrder(meter, 1), UnitOrder(second, -2)),
            ),
            (
                UnitComposition(UnitOrder(watt, 1)),
                UnitComposition(UnitOrder(kilogram, 1), UnitOrder(meter, 2), UnitOrder(second, -3)),
            ),
            (
                UnitComposition(UnitOrder(kilowatt, 1)),
                UnitComposition(
                    UnitOrder(kilogram, 1),
                    UnitOrder(meter, 2),
                    UnitOrder(second, -3),
                    scale=1_000.0,
                ),
            ),
            (
                UnitComposition(UnitOrder(minute, 1)),
                UnitComposition(UnitOrder(second, 1), scale=60.0),
            ),
            (
                UnitComposition(UnitOrder(millimeter, 1), UnitOrder(millisecond, -1)),
                UnitComposition(UnitOrder(meter, 1), UnitOrder(second, -1)),
            ),
        ),
    )
    def test_si_base_units(instance: UnitComposition, expected: UnitComposition) -> None:
        assert instance.si_base_units == expected

    @staticmethod
    @pytest.mark.parametrize(
        ("instance", "expected"),
        (
            (UnitComposition(), True),
            (UnitComposition(UnitOrder(kilogram, 1)), True),
            (UnitComposition(UnitOrder(watt, 1)), True),
            (UnitComposition(UnitOrder(gram, 1)), False),
            (UnitComposition(UnitOrder(kilogram, 1), UnitOrder(meter, 1)), True),
            (UnitComposition(UnitOrder(gram, 1), UnitOrder(kilometer, 1)), False),
            (UnitComposition(UnitOrder(millimeter, 1), UnitOrder(millisecond, -1)), False),
        ),
    )
    def test_is_coherent(instance: UnitComposition, expected: bool) -> None:
        assert instance.is_coherent is expected

    @staticmethod
    @pytest.mark.parametrize(
        ("left", "right"),
        (
            (UnitComposition(), UnitComposition()),
            (
                UnitComposition(UnitOrder(kilogram, 1), UnitOrder(meter, 2), UnitOrder(second, -3)),
                UnitComposition(UnitOrder(watt, 1)),
            ),
            (
                UnitComposition(UnitOrder(millimeter, 1), UnitOrder(millisecond, -1)),
                UnitComposition(UnitOrder(meter, 1), UnitOrder(second, -1)),
            ),
            (
                UnitComposition(UnitOrder(kilogram, 1), UnitOrder(millisecond, -1)),
                UnitComposition(UnitOrder(megagram, 1), UnitOrder(second, -1)),
            ),
            (
                UnitComposition(UnitOrder(kilogram, 1), UnitOrder(meter, 1)),
                UnitComposition(UnitOrder(gram, 1), UnitOrder(kilometer, 1)),
            ),
        ),
    )
    def test_equality(left: UnitComposition, right: UnitComposition) -> None:
        assert left == right

    @staticmethod
    def test_is_hashable() -> None:
        hash(UnitComposition())
        hash(UnitComposition(UnitOrder(meter, 1)))

    @staticmethod
    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        (
            (UnitComposition(), UnitComposition(), UnitComposition()),
            (
                UnitComposition(UnitOrder(meter, 1)),
                UnitComposition(UnitOrder(meter, 1)),
                UnitComposition(UnitOrder(meter, 2)),
            ),
            (
                UnitComposition(UnitOrder(kilogram, 1)),
                UnitComposition(UnitOrder(gram, 1)),
                UnitComposition(UnitOrder(kilogram, 1), UnitOrder(gram, 1)),
            ),
            (
                UnitComposition(UnitOrder(meter, 1)),
                UnitComposition(UnitOrder(meter, -1)),
                UnitComposition(),
            ),
            (
                UnitComposition(UnitOrder(kilogram, 1), UnitOrder(meter, 1), UnitOrder(second, -2)),
                UnitComposition(UnitOrder(second, 1), UnitOrder(kilogram, -1)),
                UnitComposition(UnitOrder(meter, 1), UnitOrder(second, -1)),
            ),
            (
                UnitComposition(UnitOrder(mile, 1)),
                UnitComposition(UnitOrder(hour, -1)),
                UnitComposition(UnitOrder(hour, -1), UnitOrder(mile, 1)),
            ),
            (
                UnitComposition(UnitOrder(meter, 1), scale=1_609.344),
                UnitComposition(UnitOrder(second, -1), scale=1.0 / 3_600.0),
                UnitComposition(
                    UnitOrder(meter, 1),
                    UnitOrder(second, -1),
                    scale=1_609.344 / 3_600.0,
                ),
            ),
            (
                60.0,
                UnitComposition(UnitOrder(second, 1)),
                UnitComposition(UnitOrder(second, 1), scale=60.0),
            ),
        ),
    )
    def test_multiplication(left: UnitComposition | float, right: UnitComposition, expected: UnitComposition) -> None:
        assert (left * right) == expected

    @staticmethod
    def test_multiply_by_zero_is_error() -> None:
        with pytest.raises(
            UnitError,
            match="Cannot multiply unit composition .* by 0.0 as result is ambiguous",  # noqa: RUF043
        ):
            _ = 0.0 * UnitComposition()

    @staticmethod
    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        (
            (UnitComposition(), UnitComposition(), UnitComposition()),
            (
                UnitComposition(UnitOrder(meter, 1)),
                UnitComposition(UnitOrder(meter, 1)),
                UnitComposition(),
            ),
            (
                UnitComposition(UnitOrder(kilogram, 1)),
                UnitComposition(UnitOrder(gram, 1)),
                UnitComposition(UnitOrder(kilogram, 1), UnitOrder(gram, -1)),
            ),
            (
                UnitComposition(UnitOrder(meter, 1)),
                UnitComposition(UnitOrder(meter, -1)),
                UnitComposition(UnitOrder(meter, 2)),
            ),
            (
                UnitComposition(UnitOrder(kilogram, 1), UnitOrder(meter, 1), UnitOrder(second, -2)),
                UnitComposition(UnitOrder(second, -1), UnitOrder(kilogram, 1)),
                UnitComposition(UnitOrder(meter, 1), UnitOrder(second, -1)),
            ),
            (
                UnitComposition(UnitOrder(mile, 1)),
                UnitComposition(UnitOrder(hour, 1)),
                UnitComposition(UnitOrder(hour, -1), UnitOrder(mile, 1)),
            ),
            (
                UnitComposition(UnitOrder(meter, 1), scale=1_609.344),
                UnitComposition(UnitOrder(second, 1), scale=3_600.0),
                UnitComposition(
                    UnitOrder(meter, 1),
                    UnitOrder(second, -1),
                    scale=1_609.344 / 3_600.0,
                ),
            ),
            (
                1,
                UnitComposition(UnitOrder(second, 1), scale=60.0),
                UnitComposition(UnitOrder(second, -1), scale=(1 / 60.0)),
            ),
        ),
    )
    def test_division(
        left: UnitComposition | typing.Literal[1], right: UnitComposition, expected: UnitComposition
    ) -> None:
        assert (left / right) == expected

    @staticmethod
    @pytest.mark.parametrize(
        ("something", "exponent", "expected"),
        (
            (UnitComposition(), 4, UnitComposition()),
            (UnitComposition(UnitOrder(meter, 2)), 4, UnitComposition(UnitOrder(meter, 8))),
            (
                UnitComposition(UnitOrder(second, 1), scale=60.0),
                -2,
                UnitComposition(UnitOrder(second, -2), scale=(1 / 60.0**2)),
            ),
            (
                UnitComposition(UnitOrder(watt, 1), scale=1_000.0),
                -2,
                UnitComposition(UnitOrder(watt, -2), scale=10**-6),
            ),
        ),
    )
    def test_exponentiation(something: UnitComposition, exponent: int, expected: UnitComposition) -> None:
        assert (something**exponent) == expected

    @staticmethod
    @pytest.mark.parametrize(
        ("unit_composition", "expected"),
        (
            (UnitComposition(), 0),
            (UnitComposition(UnitOrder(meter, 1), UnitOrder(meter, 2)), 1),
            (UnitComposition(UnitOrder(meter, 1), UnitOrder(millimeter, -1)), 2),
        ),
    )
    def test_length(unit_composition: UnitComposition, expected: int) -> None:
        assert len(unit_composition) == expected

    @staticmethod
    @pytest.mark.parametrize(
        ("unit_composition", "expected_str", "expected_repr"),
        (
            (UnitComposition(), "DIMENSIONLESS", "UnitComposition(scale=1.0)"),
            (
                UnitComposition(scale=1_000.0),
                "1000.0 DIMENSIONLESS",
                "UnitComposition(scale=1000.0)",
            ),
            (
                UnitComposition(UnitOrder(meter, 1)),
                "meter",
                "UnitComposition(UnitOrder(unit=meter, order=1), scale=1.0)",
            ),
        ),
    )
    def test_representation(unit_composition: Metadata, expected_str: str, expected_repr: str) -> None:
        assert repr(unit_composition) == expected_repr
        assert str(unit_composition) == expected_str


class TestUnit:
    @staticmethod
    @pytest.mark.parametrize(
        ("left", "right"),
        (
            (kilogram, kilogram),
            (meter, meter),
            (second, second),
            (kelvin, kelvin),
            (meter / second, meter / second),
            (kilogram * meter / second**2, kilogram * meter / second**2),
            (newton, newton),
            (Unit(UnitComposition(UnitOrder(meter, 1))), meter),
            (Unit(UnitComposition(UnitOrder(kilogram, 1))), kilogram),
            (Unit.new_named("kilowatthour", watthour, kilowatthour.prefix), kilowatthour),
            (Unit(UnitComposition(UnitOrder(kilowatthour, 1))), kilowatthour),
        ),
    )
    def test_caching(left: Unit, right: Unit) -> None:
        assert left is right

    @staticmethod
    @pytest.mark.parametrize(
        ("unit", "expected"),
        (
            (meter, "meter"),
            (kilogram, "kilogram"),
            (USD[2024], "USD2024"),
            (meter / second, "meter / second"),
            (kilowatt, "kilowatt"),
            (meter / meter, "DIMENSIONLESS"),
            (kilogram / gram, "kilogram / gram"),
            (kilogram / gram * meter / millimeter, "kilogram meter / gram millimeter"),
            (kilogram * meter / second**2, "kilogram meter / second^2"),
            (mile / hour, "mile / hour"),
            (yard**2, "yard^2"),
            (acre, "acre"),
            (hectare, "hectare"),
            ((kilogram * co2) / (kilogram * nox), "kilogram|co2 / kilogram|nox"),
        ),
    )
    def test_name(unit: Unit, expected: str) -> None:
        assert unit.name == expected

    @staticmethod
    @pytest.mark.parametrize(
        ("unit", "expected"),
        (
            (meter, UnitComposition(UnitOrder(meter, 1))),
            (kilogram, UnitComposition(UnitOrder(kilogram, 1))),
            (second, UnitComposition(UnitOrder(second, 1))),
            (kelvin, UnitComposition(UnitOrder(kelvin, 1))),
            (ampere, UnitComposition(UnitOrder(ampere, 1))),
            (mole, UnitComposition(UnitOrder(mole, 1))),
            (candela, UnitComposition(UnitOrder(candela, 1))),
            (USD[2025], UnitComposition(UnitOrder(USD[2025], 1))),
            (gram, UnitComposition(UnitOrder(kilogram, 1), scale=0.001)),
            (megagram, UnitComposition(UnitOrder(kilogram, 1), scale=1_000.0)),
            (millimeter, UnitComposition(UnitOrder(meter, 1), scale=0.001)),
            (
                watt,
                UnitComposition(UnitOrder(kilogram, 1), UnitOrder(meter, 2), UnitOrder(second, -3)),
            ),
            (
                kilowatt,
                UnitComposition(
                    UnitOrder(kilogram, 1),
                    UnitOrder(meter, 2),
                    UnitOrder(second, -3),
                    scale=1_000.0,
                ),
            ),
            (hour, UnitComposition(UnitOrder(minute, 1), scale=60.0)),
            (mile, UnitComposition(UnitOrder(meter, 1), scale=1_609.344)),
            (
                Unit.new_named("megamile", relation=mile, prefix=Prefix.MEGA),
                UnitComposition(UnitOrder(meter, 1), scale=1_609_344_000.0),
            ),
            (foot, UnitComposition(UnitOrder(meter, 1), scale=0.3048)),
            (yard, UnitComposition(UnitOrder(foot, 1), scale=3.0)),
            (
                acre,
                UnitComposition(UnitOrder(yard, 2), scale=4_840.0),
            ),
            (
                hectare,
                UnitComposition(UnitOrder(meter, 2), scale=10_000.0),
            ),
            (kilogram * co2, UnitComposition(UnitOrder(kilogram, 1))),
            (
                (kilogram * co2) / (kilogram * nox),
                UnitComposition(UnitOrder(kilogram * co2, 1), UnitOrder(kilogram * nox, -1)),
            ),
        ),
    )
    def test_relation(unit: Unit, expected: UnitComposition) -> None:
        assert unit.relation.composition == expected.composition
        assert unit.relation.scale == expected.scale

    @staticmethod
    def test_cannot_instantiate_unit_with_prefix_where_name_does_not_match() -> None:
        with pytest.raises(UnitError, match="does not start with its defined prefix"):
            _ = Unit.new_named(
                "faulty",
                relation=UnitComposition(),
                prefix=Prefix.KILO,
            )

    @staticmethod
    def test_cannot_instantiate_unit_with_inconsistent_relation() -> None:
        with pytest.raises(UnitError, match="Cannot instantiate unit .* because relation .* does not match"):  # noqa: RUF043
            _ = Unit.new_named("meter", relation=UnitComposition(UnitOrder(second, 1)))

    @staticmethod
    @pytest.mark.parametrize(
        ("unit", "expected"),
        (
            (meter, True),
            (kilogram, True),
            (second, True),
            (ampere, True),
            (mole, True),
            (kelvin, True),
            (candela, True),
            (USD[2025], True),
            (USD[2024], False),
            (gram, False),
            (megagram, False),
            (minute, False),
            (mile, False),
            (watt, False),
            (meter / second, False),
        ),
    )
    def test_is_si_base_unit(unit: Unit, expected: bool) -> None:
        assert unit.is_si_base_unit is expected

    @staticmethod
    @pytest.mark.parametrize(
        ("unit", "expected"),
        (
            (meter, True),
            (kilogram, True),
            (second, True),
            (ampere, True),
            (mole, True),
            (kelvin, True),
            (candela, True),
            (USD[2025], True),
            (USD[2024], False),
            (milligram, False),
            (gram, False),
            (megagram, False),
            (minute, False),
            (mile, False),
            (watt, True),
            (meter / second, True),
            (kilogram * meter, True),
            (gram * kilometer, False),
            (joule, True),
            (kilojoule, False),
            (kilogram * meter / second**2, True),
            (millimeter / millisecond, False),
            (gram / millisecond, False),
            (gram * kilometer / second**2, False),
        ),
    )
    def test_is_coherent(unit: Unit, expected: bool) -> None:
        assert unit.is_coherent is expected

    @staticmethod
    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        (
            (kilogram, kilogram, True),
            (meter, meter, True),
            (second, second, True),
            (kelvin, kelvin, True),
            (meter / second, meter / second, True),
            (newton, kilogram * meter / second**2, True),
            (watt, kilogram * meter**2 / second**3, True),
            (kilowatt, megagram * meter**2 / second**3, True),
            (kilometer / second, meter / millisecond, True),
            (kilogram * meter, gram * kilometer, True),
            (kilogram * co2, kilogram * nox, False),
            (kilowatt * hour, kilowatthour, True),
            (kilowatt * hour / kilogram, kilowatthour / kilogram, True),
        ),
    )
    def test_equality(left: Unit, right: Unit, expected: bool) -> None:
        assert (left == right) is expected

    @staticmethod
    @pytest.mark.parametrize("unit", (joule, milligram, kilowatthour, meter / second))
    def test_unit_error_if_no_corresponding_absolute_unit(unit: Unit) -> None:
        message = "No absolute unit corresponding to .* exists"
        with pytest.raises(UnitError, match=message):
            _ = abs(unit)

    @staticmethod
    @pytest.mark.parametrize(
        ("left", "right", "expected_name", "expected_composition"),
        (
            (
                kilogram,
                gram,
                "gram kilogram",
                UnitComposition(UnitOrder(kilogram, 1), UnitOrder(gram, 1)),
            ),
            (
                newton,
                meter,
                "meter newton",
                UnitComposition(UnitOrder(meter, 1), UnitOrder(newton, 1)),
            ),
            (
                kilowatt,
                hour,
                "hour kilowatt",
                UnitComposition(UnitOrder(hour, 1), UnitOrder(kilowatt, 1)),
            ),
        ),
    )
    def test_multiplication(left: Unit, right: Unit, expected_name: str, expected_composition: UnitComposition) -> None:
        result = left * right
        assert result.name == expected_name
        assert result.relation == expected_composition

    @staticmethod
    @pytest.mark.parametrize(
        ("left", "right", "expected_name", "expected_composition"),
        (
            (kilogram, kilogram, "DIMENSIONLESS", UnitComposition()),
            (
                kilogram * co2,
                kilogram * nox,
                "kilogram|co2 / kilogram|nox",
                UnitComposition(UnitOrder(kilogram * co2, 1), UnitOrder(kilogram * nox, -1)),
            ),
            (
                kilogram,
                gram,
                "kilogram / gram",
                UnitComposition(UnitOrder(kilogram, 1), UnitOrder(gram, -1)),
            ),
            (mile, hour, "mile / hour", UnitComposition(UnitOrder(mile, 1), UnitOrder(hour, -1))),
            (
                kilometer,
                meter,
                "kilometer / meter",
                UnitComposition(UnitOrder(kilometer, 1), UnitOrder(meter, -1)),
            ),
        ),
    )
    def test_division(left: Unit, right: Unit, expected_name: str, expected_composition: UnitComposition) -> None:
        result = left / right
        assert result.name == expected_name
        assert result.relation == expected_composition

    @staticmethod
    @pytest.mark.parametrize(
        ("unit", "exponent", "expected"),
        (
            (meter, 2, Unit(UnitComposition(UnitOrder(meter, 2)))),
            (kilogram, 2, Unit(UnitComposition(UnitOrder(kilogram, 2)))),
            (kilowatt, 2, Unit(UnitComposition(UnitOrder(kilowatt, 2)))),
            (
                kilogram * meter / second**2,
                -2,
                Unit(
                    UnitComposition(UnitOrder(kilogram, -2), UnitOrder(meter, -2), UnitOrder(second, 4)),
                ),
            ),
            (DIMENSIONLESS, -2, Unit(UnitComposition())),
        ),
    )
    def test_exponentiation_with_int_exponent(unit: Unit, exponent: int, expected: Unit) -> None:
        assert unit**exponent == expected

    @staticmethod
    def test_exponentiation_of_dimensionless_with_float_exponent() -> None:
        assert DIMENSIONLESS**1.5 == DIMENSIONLESS

    @staticmethod
    @pytest.mark.parametrize(
        ("unit", "exponent"),
        (
            (meter, 1.5),
            (kilogram, -0.5),
            (kilogram * meter / second**2, 0.0),
            (kilowatt, 1.0),
        ),
    )
    def test_exponentiation_with_float_exponent(unit: Unit, exponent: float) -> None:
        message = re.escape("unsupported operand type(s) for ** or pow(): 'Unit' and 'float'")
        with pytest.raises(TypeError, match=message):
            _ = unit**exponent

    @staticmethod
    @pytest.mark.parametrize(
        ("unit", "expected"),
        (
            (kilogram * Metadata("carbon dioxide"), kilogram),
            ((kilogram * Metadata("co2")) / (kilogram * Metadata("nox")), DIMENSIONLESS),
            (kilogram * meter / second**2 * Metadata("vertical"), kilogram * meter / second**2),
            (kilowatthour * Metadata("thermal"), kilowatthour),
        ),
    )
    def test_strip_metadata(unit: Unit, expected: Unit) -> None:
        assert unit.strip_metadata() is expected

    @staticmethod
    @pytest.mark.parametrize(
        ("unit", "expected"),
        (
            (meter, UnitComposition(UnitOrder(meter, 1), scale=1.0)),
            (millimeter, UnitComposition(UnitOrder(meter, 1), scale=0.001)),
            (kilogram, UnitComposition(UnitOrder(kilogram, 1), scale=1.0)),
            (gram, UnitComposition(UnitOrder(kilogram, 1), scale=0.001)),
            (megagram, UnitComposition(UnitOrder(kilogram, 1), scale=1_000.0)),
            (fraction, UnitComposition(scale=1.0)),
            (percent, UnitComposition(scale=0.01)),
            (percent * Metadata("mass"), UnitComposition(scale=0.01)),
        ),
    )
    def test_to_si_base_units(unit: Unit, expected: UnitComposition) -> None:
        assert unit.to_si_base_units().relation == expected


class TestDimensionless:
    @staticmethod
    @pytest.mark.parametrize(
        ("name", "relation", "expected_relation"),
        (
            ("radian", DIMENSIONLESS, UnitComposition()),
            (
                "degree",
                (math.pi / 180.0) * radian,
                UnitComposition(UnitOrder(radian, 1), scale=math.pi / 180.0),
            ),
            ("count", DIMENSIONLESS, UnitComposition()),
        ),
    )
    def test_new_named(name: str, relation: Unit | Quantity, expected_relation: UnitComposition) -> None:
        unit = Unit.new_named(name, relation)
        assert unit.name == name
        assert unit.relation == expected_relation

    @staticmethod
    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        (
            (DIMENSIONLESS, meter, meter),
            (meter, DIMENSIONLESS, meter),
            (DIMENSIONLESS, kilogram * meter / second**2, kilogram * meter / second**2),
            (kilogram * meter / second**2, DIMENSIONLESS, kilogram * meter / second**2),
            (DIMENSIONLESS, fraction * Metadata("mass"), fraction * Metadata("mass")),
            (fraction * Metadata("mass"), DIMENSIONLESS, fraction * Metadata("mass")),
        ),
    )
    def test_multiplication(left: Unit, right: Unit, expected: Unit) -> None:
        assert left * right == expected

    @staticmethod
    @pytest.mark.parametrize(
        ("numerator", "denominator", "expected"),
        (
            (DIMENSIONLESS, meter, 1 / meter),
            (meter, DIMENSIONLESS, meter),
            (DIMENSIONLESS, kilogram * meter / second**2, second**2 / (kilogram * meter)),
            (kilogram * meter / second**2, DIMENSIONLESS, kilogram * meter / second**2),
        ),
    )
    def test_division(numerator: Unit, denominator: Unit, expected: Unit) -> None:
        assert numerator / denominator == expected

    @staticmethod
    @pytest.mark.parametrize(
        ("exponent"),
        (0, 1, -2, 0.0, 1.5, -0.7, -2.0),
    )
    def test_exponentiation(exponent: float) -> None:
        assert DIMENSIONLESS**exponent is DIMENSIONLESS

    @staticmethod
    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        (
            (fraction, DIMENSIONLESS, True),
            (fraction * kilogram / second, kilogram / second, True),
            ((fraction * Metadata("mass")) * kilogram / second, kilogram / second, True),
            (fraction, percent, False),
        ),
    )
    def test_equality(left: Unit, right: Unit, expected: bool) -> None:
        assert (left == right) is expected


class TestAbsoluteUnit:
    @staticmethod
    @pytest.mark.parametrize(
        ("name", "scale", "zero", "expected_relation", "expected_offset"),
        (
            ("|kelvin|", kelvin, None, UnitComposition(UnitOrder(kelvin, 1), scale=1.0), None),
            (
                "|celsius|",
                celsius,
                273.15 * abs(kelvin),
                UnitComposition(UnitOrder(celsius, 1)),
                UnitComposition(UnitOrder(abs(kelvin), 1), scale=273.15),
            ),
            (
                "|fahrenheit|",
                fahrenheit,
                (-160 / 9) * abs(celsius),
                UnitComposition(UnitOrder(fahrenheit, 1)),
                UnitComposition(UnitOrder(abs(celsius), 1), scale=-160 / 9),
            ),
        ),
    )
    def test_instantiate(
        name: str,
        scale: Unit,
        zero: None,
        expected_relation: UnitComposition,
        expected_offset: UnitComposition | None,
    ) -> None:
        absolute = Unit.new_absolute(name, scale, zero)
        assert absolute.name == name
        assert absolute.relation == expected_relation
        assert absolute.offset == expected_offset
        assert absolute.dimension is TEMPERATURE

    @staticmethod
    @pytest.mark.parametrize(
        ("unit", "absolute_unit"),
        (
            (kelvin, absolute_kelvin),
            (celsius, absolute_celsius),
            (fahrenheit, absolute_fahrenheit),
        ),
    )
    def test_unit_abs(unit: Unit, absolute_unit: Unit) -> None:
        assert abs(unit) is absolute_unit

    @staticmethod
    def test_unit_error_if_reinstantiation_with_different_zero() -> None:
        _absolute_celsius = Unit.new_absolute("|celsius|", scale=celsius, zero=273.15 * abs(kelvin))
        message = (
            "Cannot instantiate new .*Unit.* with zero .* because one already exists with the same "
            "name .* and scale .* but a different zero .*"
        )
        with pytest.raises(UnitError, match=message):
            _ = Unit.new_absolute("|celsius|", scale=celsius, zero=0.0 * abs(kelvin))

    @staticmethod
    def test_unit_error_if_reinstantiation_with_different_scale() -> None:
        _absolute_celsius = Unit.new_absolute("|celsius|", scale=celsius, zero=273.15 * abs(kelvin))
        message = "Cannot instantiate new .*Unit.* with scale .* because one already exists with the same scale"
        with pytest.raises(UnitError, match=message):
            _ = Unit.new_absolute("|celsius|", scale=kelvin, zero=273.15 * abs(kelvin))

    @staticmethod
    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        (
            (1.0 * abs(kelvin), 1.0 * kelvin, 2.0 * abs(kelvin)),
            (1.0 * kelvin, 1.0 * abs(kelvin), 2.0 * abs(kelvin)),
            (1.0 * kelvin, 1.0 * kelvin, 2.0 * kelvin),
            (1.0 * abs(celsius), 1.0 * celsius, 2.0 * abs(celsius)),
            (1.0 * celsius, 1.0 * abs(celsius), 2.0 * abs(celsius)),
            (1.0 * celsius, 1.0 * celsius, 2.0 * celsius),
        ),
    )
    def test_addition(left: Quantity, right: Quantity, expected: Quantity) -> None:
        assert (left + right) == expected

    @staticmethod
    @pytest.mark.parametrize(
        ("left", "right"),
        (
            (0.0 * abs(kelvin), 0.0 * abs(kelvin)),
            (1.0 * abs(kelvin), 1.0 * abs(kelvin)),
            (1.0 * abs(celsius), 1.0 * abs(celsius)),
            (1.0 * abs(fahrenheit), 1.0 * abs(fahrenheit)),
        ),
    )
    def test_unit_error_for_addition_of_two_absolute_units(left: Quantity, right: Quantity) -> None:
        message = "Cannot add two quantities with absolute units of .* and .*"
        with pytest.raises(UnitError, match=message):
            _ = left + right

    @staticmethod
    @pytest.mark.parametrize(
        ("quantity", "unit", "expected"),
        (
            (273.15 * abs(kelvin), abs(kelvin), 273.15 * abs(kelvin)),
            (273.15 * abs(kelvin), abs(celsius), 0.0 * abs(celsius)),
            (273.15 * abs(kelvin), abs(fahrenheit), 32.0 * abs(fahrenheit)),
            (0.0 * abs(celsius), abs(kelvin), 273.15 * abs(kelvin)),
            (50.0 * abs(celsius), abs(celsius), 50.0 * abs(celsius)),
            (0.0 * abs(celsius), abs(fahrenheit), 32.0 * abs(fahrenheit)),
            (50.0 * abs(fahrenheit), abs(fahrenheit), 50.0 * abs(fahrenheit)),
            (50.0 * abs(fahrenheit), abs(kelvin), 283.15 * abs(kelvin)),
            (0.0 * abs(fahrenheit), abs(celsius), -160 / 9 * abs(celsius)),
            (100.0 * abs(fahrenheit), abs(kelvin), 2798.35 / 9 * abs(kelvin)),
            (100.0 * abs(fahrenheit), abs(celsius), 340 / 9 * abs(celsius)),
        ),
    )
    def test_convert_to(quantity: Quantity, unit: Unit, expected: Quantity) -> None:
        assert quantity.convert_to(unit) == expected
