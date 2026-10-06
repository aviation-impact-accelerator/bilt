import operator
import typing

import pytest

from bilt.core._dimensional_analysis import dimensionality_of
from bilt.errors import BiltDimensionalityError
from bilt.symbolic import dim
from bilt.symbolic._atoms import Atom, DimensionType, FloatType, IntType, OpType
from bilt.symbolic._dimensions import Dim
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


def test_negation_keeps_the_dimensionality() -> None:
    assert dimensionality_of((-x).tree, {"x": MASS}) is MASS


def test_an_inverse_inverts_the_dimensionality() -> None:
    assert dimensionality_of((1 / x).tree, {"x": TIME}) is TIME**-1


def test_an_unknown_factor_makes_an_inverse_unknown() -> None:
    assert dimensionality_of((1 / x).tree, {}) is None


@pytest.mark.parametrize(
    "relation",
    (operator.eq, operator.lt, operator.le, operator.gt, operator.ge, operator.ne),
    ids=("==", "<", "<=", ">", ">=", "!="),
)
def test_a_relation_of_like_dimensionalities_is_dimensionless(
    relation: collections.abc.Callable[[Expr, Expr], Tree],
) -> None:
    assert dimensionality_of(relation(x, y), {"x": LENGTH, "y": LENGTH}) is DIMENSIONLESS


def test_a_relation_between_unlike_dimensionalities_raises() -> None:
    with pytest.raises(BiltDimensionalityError):
        dimensionality_of(x == y, {"x": LENGTH, "y": MASS})


def test_a_relation_with_an_unknown_side_is_dimensionless() -> None:
    assert dimensionality_of(x == y, {"x": LENGTH}) is DIMENSIONLESS


def test_a_bare_literal_zero_is_unknown_and_a_nonzero_literal_is_dimensionless() -> None:
    assert dimensionality_of(Tree(Atom(IntType, 0)), {}) is None
    assert dimensionality_of(Tree(Atom(FloatType, 0.0)), {}) is None
    assert dimensionality_of(Tree(Atom(IntType, 2)), {}) is DIMENSIONLESS


def test_an_axis_is_unknown() -> None:
    assert dimensionality_of(Tree(Atom(DimensionType, Dim("t"))), {}) is None


def test_an_unknown_operation_raises() -> None:
    with pytest.raises(NotImplementedError):
        dimensionality_of(Tree(Atom(OpType, "unknown"), (symbol_tree("y"),)), {"y": LENGTH})
