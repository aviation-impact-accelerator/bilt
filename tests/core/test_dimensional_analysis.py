import operator
import typing

import pytest

from bilt.core._dimensional_analysis import dimensionality_of, implied_dimensionality
from bilt.errors import BiltDimensionalityError
from bilt.symbolic import dim
from bilt.symbolic._atoms import Atom, OpType
from bilt.symbolic._expressions import symbol_expr, tensor_symbol_expr
from bilt.symbolic._tree import Tree, symbol_tree
from bilt.units.dimensions import AREA, DIMENSIONLESS, FORCE, LENGTH, MASS, TIME

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.symbolic._expressions import Expr

x = symbol_expr("x")
y = symbol_expr("y")
z = symbol_expr("z")
tensor_x = tensor_symbol_expr("x")
tensor_y = tensor_symbol_expr("y")


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


def test_a_zeroth_power_is_dimensionless_whatever_its_base() -> None:
    assert dimensionality_of((x**0).tree, {}) is DIMENSIONLESS


def test_dimensionless_base_accepts_a_dimensionless_exponent() -> None:
    assert dimensionality_of((x**y).tree, {"x": DIMENSIONLESS, "y": DIMENSIONLESS}) is DIMENSIONLESS


@pytest.mark.parametrize("reduction", ("mean", "sum", "min", "max"))
def test_reductions_but_a_product_preserve_the_dimensionality(reduction: str) -> None:
    assert dimensionality_of(getattr(tensor_x, reduction)(dim("t")).tree, {"x": MASS}) is MASS


def test_a_product_along_an_axis_and_a_selection_are_unknown() -> None:
    assert dimensionality_of(tensor_x.prod(dim("t")).tree, {"x": LENGTH}) is None
    assert dimensionality_of(tensor_x.isel(dim("t"), y).tree, {"x": MASS, "y": TIME}) is None


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


def test_an_addend_takes_the_dimensionality_of_the_other_side() -> None:
    assert implied_dimensionality(x == y + z, "y", {"x": LENGTH}) is LENGTH


def test_an_addend_takes_the_dimensionality_of_the_addends_beside_it() -> None:
    assert implied_dimensionality(x == y + z, "y", {"z": LENGTH}) is LENGTH


def test_a_factor_takes_the_other_side_divided_by_the_other_factors() -> None:
    assert implied_dimensionality(x == y * z, "y", {"x": FORCE, "z": MASS}) is LENGTH / TIME**2


def test_a_quotient_implies_its_dividend_and_divisor() -> None:
    assert implied_dimensionality(x == y / z, "y", {"x": LENGTH / TIME, "z": TIME}) is LENGTH
    assert implied_dimensionality(x == y / z, "z", {"x": LENGTH / TIME, "y": LENGTH}) is TIME


def test_an_inverse_implies_the_inverse_dimensionality() -> None:
    assert implied_dimensionality(x == 1 / y, "y", {"x": TIME}) is TIME**-1


def test_negation_and_mean_pass_the_dimensionality_through() -> None:
    assert implied_dimensionality(x == -y, "y", {"x": MASS}) is MASS
    assert implied_dimensionality(x == tensor_y.mean(dim("t")), "y", {"x": MASS}) is MASS


def test_an_integer_power_implies_its_root() -> None:
    assert implied_dimensionality(x == y**2, "y", {"x": AREA}) is LENGTH
    assert implied_dimensionality(x == y**-1, "y", {"x": LENGTH}) is LENGTH**-1


def test_a_power_with_no_exact_root_raises() -> None:
    with pytest.raises(BiltDimensionalityError):
        implied_dimensionality(x == y**2, "y", {"x": LENGTH})


def test_a_zeroth_power_implies_nothing_but_demands_dimensionless() -> None:
    assert implied_dimensionality(x == y**0, "y", {"x": DIMENSIONLESS}) is None
    with pytest.raises(BiltDimensionalityError):
        implied_dimensionality(x == y**0, "y", {"x": LENGTH})


def test_a_non_literal_exponent_implies_a_dimensionless_base_and_exponent() -> None:
    assert implied_dimensionality(x == y**z, "y", {"x": DIMENSIONLESS}) is DIMENSIONLESS
    assert implied_dimensionality(x == y**z, "z", {"x": DIMENSIONLESS}) is DIMENSIONLESS
    assert implied_dimensionality(x == y**0.5, "y", {}) is DIMENSIONLESS


def test_a_symbol_on_both_sides_is_implied_when_they_agree() -> None:
    assert implied_dimensionality(x + y == 2 * x, "x", {"y": LENGTH}) is LENGTH


def test_a_symbol_on_both_sides_raises_when_they_disagree() -> None:
    with pytest.raises(BiltDimensionalityError):
        implied_dimensionality(x + y == z * x, "x", {"y": MASS, "z": LENGTH})


def test_an_absent_symbol_implies_nothing() -> None:
    assert implied_dimensionality(x == y, "w", {"x": LENGTH}) is None


def test_an_undetermined_symbol_implies_nothing() -> None:
    assert implied_dimensionality(x == y * z, "y", {"x": LENGTH}) is None
    assert implied_dimensionality(x * x == z, "x", {"z": AREA}) is None


def test_an_impossible_demand_elsewhere_does_not_block_the_symbol() -> None:
    assert implied_dimensionality(x + y**2 == z, "x", {"z": LENGTH}) is LENGTH


@pytest.mark.parametrize(
    "relation",
    (operator.lt, operator.le, operator.gt, operator.ge, operator.ne),
    ids=("<", "<=", ">", ">=", "!="),
)
def test_inequalities_imply_like_equality(relation: collections.abc.Callable[[Expr, Expr], Tree]) -> None:
    assert implied_dimensionality(relation(x, y), "y", {"x": LENGTH}) is LENGTH


def test_a_literal_zero_implies_nothing() -> None:
    assert implied_dimensionality(x - y == 0, "x", {}) is None
    assert implied_dimensionality(x == 0, "x", {}) is None


def test_a_nonzero_literal_implies_dimensionless() -> None:
    assert implied_dimensionality(x == 1, "x", {}) is DIMENSIONLESS


def test_inhomogeneous_known_parts_raise() -> None:
    with pytest.raises(BiltDimensionalityError):
        implied_dimensionality(x == y + z, "y", {"x": LENGTH, "z": MASS})


def test_an_unknown_operation_raises() -> None:
    with pytest.raises(NotImplementedError):
        implied_dimensionality(Tree(Atom(OpType, "unknown"), (symbol_tree("y"),)), "y", {})
