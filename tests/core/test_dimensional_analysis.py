import operator
import typing

import pytest

from bilt.core._dimensional_analysis import absoluteness_of, dimensionality_of
from bilt.core._port_declaration import PortDeclaration
from bilt.errors import BiltDimensionalityError, BiltEquationDimensionalityError
from bilt.symbolic import dim
from bilt.symbolic._atoms import Atom, DimensionType, OpType
from bilt.symbolic._dimensions import Dim
from bilt.symbolic._expressions import as_tree, symbol_expr, tensor_symbol_expr
from bilt.symbolic._tree import Tree, symbol_tree
from bilt.units import Quantity
from bilt.units.dimensions import AREA, DIMENSIONLESS, FORCE, LENGTH, MASS, TEMPERATURE, TIME
from bilt.units.units import absolute_celsius, kelvin

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.symbolic._expressions import Expr

x = symbol_expr("x")
y = symbol_expr("y")
z = symbol_expr("z")
tensor_x = tensor_symbol_expr("x")
tensor_y = tensor_symbol_expr("y")

DECLARATIONS = {
    "x": PortDeclaration(TEMPERATURE, absolute=True),
    "y": PortDeclaration(TEMPERATURE),
    "z": PortDeclaration(TEMPERATURE, absolute=True),
}


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


def test_an_axis_is_unknown() -> None:
    assert dimensionality_of(Tree(Atom(DimensionType, Dim("t"))), {}) is None


def test_an_unknown_operation_raises() -> None:
    with pytest.raises(NotImplementedError):
        dimensionality_of(Tree(Atom(OpType, "unknown"), (symbol_tree("y"),)), {"y": LENGTH})


def test_a_declared_port_is_absolute_if_marked_and_relative_otherwise() -> None:
    assert absoluteness_of("balance", x.tree, DECLARATIONS) == {1}
    assert absoluteness_of("balance", y.tree, DECLARATIONS) == {0}


def test_a_plain_port_and_a_literal_are_read_either_way() -> None:
    assert absoluteness_of("balance", x.tree, {}) == {0, 1}
    assert absoluteness_of("balance", as_tree(2.0), DECLARATIONS) == {0, 1}
    assert absoluteness_of("balance", as_tree(0), DECLARATIONS) == {0, 1}


def test_a_quantity_constant_is_absolute_if_its_unit_is() -> None:
    assert absoluteness_of("balance", as_tree(Quantity(0.0, absolute_celsius)), {}) == {1}
    assert absoluteness_of("balance", as_tree(2.0 * kelvin), {}) == {0}


@pytest.mark.parametrize(
    ("expression", "expected"),
    (
        (x + y, {1}),
        (y + x, {1}),
        (x - y, {1}),
        (x - z, {0}),
        (y + y, {0}),
        (y - y, {0}),
        (-x + z, {0}),
        (z + (x - z) / 2, {1}),
    ),
    ids=str,
)
def test_a_sum_adds_the_fixed_zeros_of_its_addends(expression: Expr, expected: set[int]) -> None:
    assert absoluteness_of("balance", expression.tree, DECLARATIONS) == expected


@pytest.mark.parametrize("expression", (x * z, x / z, x**2, 2 * x), ids=str)
def test_a_product_a_quotient_and_a_power_are_read_either_way(expression: Expr) -> None:
    assert absoluteness_of("balance", expression.tree, DECLARATIONS) == {0, 1}


def test_a_sum_holding_an_addend_read_either_way_keeps_every_legal_reading() -> None:
    assert absoluteness_of("balance", (x + y * y).tree, DECLARATIONS) == {1}
    assert absoluteness_of("balance", (x - y * y).tree, DECLARATIONS) == {0, 1}
    assert absoluteness_of("balance", (y + 0).tree, DECLARATIONS) == {0, 1}


@pytest.mark.parametrize("expression", (x + z, (x + z) / 2, x + y + z), ids=str)
def test_adding_two_absolute_quantities_raises(expression: Expr) -> None:
    with pytest.raises(BiltEquationDimensionalityError, match="'balance' adds two absolute quantities"):
        absoluteness_of("balance", expression.tree, DECLARATIONS)


@pytest.mark.parametrize("expression", (y - x, -x + y, -x - z), ids=str)
def test_subtracting_an_absolute_quantity_from_a_relative_one_raises(expression: Expr) -> None:
    with pytest.raises(
        BiltEquationDimensionalityError, match="'balance' subtracts an absolute quantity from a relative one"
    ):
        absoluteness_of("balance", expression.tree, DECLARATIONS)


def test_a_relation_between_sides_read_alike_is_relative() -> None:
    assert absoluteness_of("balance", x == z, DECLARATIONS) == {0}
    assert absoluteness_of("balance", x > y * y, DECLARATIONS) == {0}


def test_a_relation_between_an_absolute_and_a_relative_quantity_raises() -> None:
    with pytest.raises(
        BiltEquationDimensionalityError, match="'balance' relates an absolute quantity to a relative one"
    ):
        absoluteness_of("balance", x == y, DECLARATIONS)


@pytest.mark.parametrize("reduction", ("mean", "sum", "min", "max"))
def test_reductions_but_a_product_preserve_the_absoluteness(reduction: str) -> None:
    assert absoluteness_of("balance", getattr(tensor_x, reduction)(dim("t")).tree, DECLARATIONS) == {1}


def test_a_product_along_an_axis_and_a_selection_are_read_either_way() -> None:
    assert absoluteness_of("balance", tensor_x.prod(dim("t")).tree, DECLARATIONS) == {0, 1}
    assert absoluteness_of("balance", tensor_x.isel(dim("t"), y).tree, DECLARATIONS) == {0, 1}
