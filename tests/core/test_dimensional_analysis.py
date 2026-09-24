import pytest

from bilt.core._dimensional_analysis import dimensionality_of
from bilt.errors import BiltDimensionalityError
from bilt.math import mean
from bilt.symbolic import dim
from bilt.symbolic._expressions import symbol_expr
from bilt.units.dimensions import AREA, DIMENSIONLESS, FORCE, LENGTH, MASS, TIME

x = symbol_expr("x")
y = symbol_expr("y")
z = symbol_expr("z")


def test_product_composes_dimensionalities() -> None:
    assert dimensionality_of((x * y).tree, {"x": MASS, "y": LENGTH / TIME**2}) is FORCE


def test_quotient_divides_dimensionalities() -> None:
    assert dimensionality_of((x / y).tree, {"x": LENGTH, "y": TIME}) is LENGTH / TIME


def test_difference_of_like_dimensionalities_keeps_it() -> None:
    assert dimensionality_of((x - y).tree, {"x": LENGTH, "y": LENGTH}) is LENGTH


def test_difference_in_a_denominator_composes() -> None:
    assert dimensionality_of((x / (y - z)).tree, {"x": LENGTH, "y": TIME, "z": TIME}) is LENGTH / TIME


def test_nonzero_literal_is_dimensionless() -> None:
    assert dimensionality_of((2 * x).tree, {"x": MASS}) is MASS
    assert dimensionality_of((x + 1).tree, {"x": DIMENSIONLESS}) is DIMENSIONLESS


def test_integer_powers_raise_the_dimensionality() -> None:
    assert dimensionality_of((x**2).tree, {"x": LENGTH}) is AREA
    assert dimensionality_of((x**-1).tree, {"x": LENGTH}) is LENGTH**-1


def test_dimensionless_base_accepts_a_dimensionless_exponent() -> None:
    assert dimensionality_of((x**y).tree, {"x": DIMENSIONLESS, "y": DIMENSIONLESS}) is DIMENSIONLESS


def test_mean_preserves_the_dimensionality() -> None:
    assert dimensionality_of(mean(x, dim("t")).tree, {"x": MASS}) is MASS


def test_adding_unlike_dimensionalities_raises() -> None:
    with pytest.raises(BiltDimensionalityError) as raised:
        dimensionality_of((x + y).tree, {"x": MASS, "y": LENGTH})

    assert str(MASS) in str(raised.value)
    assert str(LENGTH) in str(raised.value)


def test_adding_a_nonzero_literal_to_a_dimensional_symbol_raises() -> None:
    with pytest.raises(BiltDimensionalityError):
        dimensionality_of((x + 1).tree, {"x": LENGTH})


def test_zero_takes_the_dimensionality_of_its_neighbours() -> None:
    assert dimensionality_of((x + 0).tree, {"x": LENGTH}) is LENGTH
    assert dimensionality_of((x - 0.0).tree, {"x": LENGTH}) is LENGTH


def test_dimensional_exponent_raises() -> None:
    with pytest.raises(BiltDimensionalityError, match="exponent must be dimensionless") as raised:
        dimensionality_of((x**y).tree, {"x": LENGTH, "y": MASS})

    assert str(MASS) in str(raised.value)


@pytest.mark.parametrize("exponent", (y, 0.5))
def test_dimensional_base_with_a_non_integer_literal_exponent_raises(exponent: object) -> None:
    with pytest.raises(BiltDimensionalityError, match="integer literal") as raised:
        dimensionality_of((x**exponent).tree, {"x": LENGTH, "y": DIMENSIONLESS})

    assert str(LENGTH) in str(raised.value)


def test_unknown_symbol_makes_its_product_unknown() -> None:
    assert dimensionality_of((x * y).tree, {"x": LENGTH}) is None


def test_unknown_addend_takes_the_dimensionality_of_its_neighbours() -> None:
    with pytest.raises(BiltDimensionalityError):
        dimensionality_of((x + y + z).tree, {"x": MASS, "z": LENGTH})


def test_unknown_base_with_a_dimensional_exponent_raises() -> None:
    with pytest.raises(BiltDimensionalityError, match="exponent must be dimensionless"):
        dimensionality_of((x**y).tree, {"y": MASS})


def test_dimensional_base_with_an_unknown_exponent_raises() -> None:
    with pytest.raises(BiltDimensionalityError, match="integer literal"):
        dimensionality_of((x**y).tree, {"x": LENGTH})
