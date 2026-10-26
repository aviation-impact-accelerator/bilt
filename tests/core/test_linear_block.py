import itertools
import typing

import pytest

from bilt import Implementation, Interface, equation, field, system, transform
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._structural_analysis._linear_block import (
    BiltLinearBlockError,
    LinearBlock,
    extract_linear_block,
)
from bilt.core._structural_analysis._plan import PlanBlock, PlanStep
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._to_ast import tree_to_functiondef
from bilt.symbolic._algebra._affine import AffineAnalysisError
from bilt.symbolic._algebra._expr_ops import same_tree

if typing.TYPE_CHECKING:
    from bilt.symbolic._tree import Tree


@equation
def first(x: float, y: float, p: float) -> bool:
    return 2 * x + 3 * y == p


@equation
def second(x: float, y: float, q: float) -> bool:
    return -x + 4 * y == q


def _value(tree: Tree, **inputs: float) -> float:
    function = PYTHON_BACKEND.transpile(
        tree_to_functiondef(tree, name="coefficient", parameter_order=tuple(inputs)),
    )
    return typing.cast("float", function.evaluate(inputs))


def _matrix(linear: LinearBlock, **inputs: float) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(_value(value, **inputs) for value in row) for row in linear.coefficients)


@pytest.mark.parametrize("reverse", (False, True))
def test_extracts_a_coupled_system_with_stable_axes_and_source_identity(reverse: bool) -> None:
    steps = (PlanStep(operation=first, output="x"), PlanStep(operation=second, output="y"))
    block = PlanBlock(steps=steps[::-1] if reverse else steps, is_cyclic=True)
    original_trees = (first.__bilt_equation_tree__, second.__bilt_equation_tree__)

    linear = extract_linear_block(block)

    assert linear.block is block
    assert linear.unknowns == ("x", "y")
    assert linear.row_steps[0] is steps[0]
    assert linear.row_steps[1] is steps[1]
    assert linear.inputs == frozenset({"p", "q"})
    assert _matrix(linear) == ((2, 3), (-1, 4))
    assert tuple(_value(value, p=7, q=5) for value in linear.rhs) == (7, 5)
    assert first.__bilt_equation_tree__ is original_trees[0]
    assert second.__bilt_equation_tree__ is original_trees[1]


def test_reconstructed_residuals_agree_with_original_equations() -> None:
    block = PlanBlock(
        steps=(PlanStep(operation=first, output="x"), PlanStep(operation=second, output="y")),
        is_cyclic=True,
    )
    linear = extract_linear_block(block)
    matrix = _matrix(linear)

    for x, y in itertools.product((-2.0, 0.0, 3.0), repeat=2):
        expected = (2 * x + 3 * y - 7, -x + 4 * y - 5)
        for row, rhs, residual in zip(matrix, linear.rhs, expected, strict=True):
            actual = row[0] * x + row[1] * y - _value(rhs, p=7, q=5)
            assert actual == pytest.approx(residual)


def test_repeated_unknown_on_both_sides_is_collected() -> None:
    @equation
    def repeated(x: float, y: float) -> bool:
        return x + x + y == x + 3 * y

    linear = extract_linear_block(PlanBlock(steps=(PlanStep(operation=repeated, output="x"),)))

    assert linear.unknowns == ("x",)
    assert _matrix(linear) == ((1,),)
    expected = 8
    assert _value(linear.rhs[0], y=4) == expected


def test_external_inputs_can_appear_nonlinearly_in_coefficients_and_rhs() -> None:
    @equation
    def parameterised(x: float, y: float, a: float, b: float, c: float) -> bool:
        return a**2 * x + y / b == c**2

    @equation
    def link(x: float, y: float, d: float) -> bool:
        return y == x + d

    block = PlanBlock(
        steps=(PlanStep(operation=parameterised, output="x"), PlanStep(operation=link, output="y")),
        is_cyclic=True,
    )
    linear = extract_linear_block(block)

    assert linear.inputs == frozenset({"a", "b", "c", "d"})
    assert _matrix(linear, a=2, b=4) == ((4, 0.25), (-1, 1))
    assert tuple(_value(value, c=3, d=5) for value in linear.rhs) == (9, 5)


def test_linearity_depends_on_which_symbols_are_unknown() -> None:
    @equation
    def product(x: float, y: float, p: float) -> bool:
        return x * y == p

    # y is an external coefficient when only x is unknown.
    linear = extract_linear_block(PlanBlock(steps=(PlanStep(operation=product, output="x"),)))
    assert _matrix(linear, y=3) == ((3,),)

    # The same primitive is not affine when both x and y are unknown.
    coupled = PlanBlock(
        steps=(PlanStep(operation=product, output="x"), PlanStep(operation=second, output="y")),
        is_cyclic=True,
    )
    with pytest.raises(BiltLinearBlockError, match="supported affine form") as raised:
        extract_linear_block(coupled)
    assert isinstance(raised.value.__cause__, AffineAnalysisError)


@pytest.mark.parametrize("kind", ("power", "reciprocal", "power_one"))
def test_unsupported_affine_forms_retain_the_cause_without_claiming_nonlinearity(kind: str) -> None:
    @equation
    def power(x: float, p: float) -> bool:
        return x**2 == p

    @equation
    def reciprocal(x: float, p: float) -> bool:
        return 1 / x == p

    @equation
    def power_one(x: float, p: float) -> bool:
        return x**1 == p

    source = {"power": power, "reciprocal": reciprocal, "power_one": power_one}[kind]
    block = PlanBlock(steps=(PlanStep(operation=source, output="x"),))

    with pytest.raises(BiltLinearBlockError, match="Cannot extract a supported affine form") as raised:
        extract_linear_block(block)

    assert isinstance(raised.value.__cause__, AffineAnalysisError)
    assert source.__bilt_name__ in str(raised.value)
    assert "nonlinear" not in str(raised.value)


@pytest.mark.parametrize("offset", (0.0, 1.0))
def test_cancelled_unknown_leaves_a_zero_row_for_later_solve_diagnostics(offset: float) -> None:
    @equation
    def cancelled(x: float) -> bool:
        return x + offset == x

    block = PlanBlock(steps=(PlanStep(operation=cancelled, output="x"),), is_cyclic=True)

    linear = extract_linear_block(block)

    assert _matrix(linear) == ((0,),)
    assert _value(linear.rhs[0]) == -offset


def test_distinct_but_dependent_equations_are_retained() -> None:
    @equation
    def duplicate_relation(x: float, y: float, p: float) -> bool:
        return 2 * x + 3 * y == p

    block = PlanBlock(
        steps=(PlanStep(operation=first, output="x"), PlanStep(operation=duplicate_relation, output="y")),
        is_cyclic=True,
    )

    linear = extract_linear_block(block)

    assert _matrix(linear) == ((2, 3), (2, 3))
    assert same_tree(linear.rhs[0], linear.rhs[1])


def test_qualified_ports_are_preserved() -> None:
    @system
    class Pair(metaclass=Interface):
        x: float = field()
        y: float = field()

    @equation
    def sum_ports(pair: Pair, p: float) -> bool:
        return pair.x + pair.y == p

    @equation
    def difference_ports(pair: Pair, q: float) -> bool:
        return pair.x - pair.y == q

    block = PlanBlock(
        steps=(
            PlanStep(operation=sum_ports, output="pair.x"),
            PlanStep(operation=difference_ports, output="pair.y"),
        ),
        is_cyclic=True,
    )

    linear = extract_linear_block(block)

    assert linear.unknowns == ("pair.x", "pair.y")
    assert linear.inputs == frozenset({"p", "q"})
    assert _matrix(linear) == ((1, 1), (1, -1))
    assert linear.row_steps[0].operation is sum_ports


def test_transforms_are_not_treated_as_equations() -> None:
    @transform
    def x(y: float) -> float:
        return 2 * y

    block = PlanBlock(steps=(PlanStep(operation=x, output="x"),))

    with pytest.raises(BiltLinearBlockError, match="BasicTransform is not a primitive equation"):
        extract_linear_block(block)


def test_opaque_subsystem_is_not_probed_or_treated_as_an_equation() -> None:
    @system
    class Pair(metaclass=Interface):
        x: float = field()
        y: float = field()

    @equation
    def link(x: float, y: float) -> bool:
        return x == y

    child = SubSystemConstituent(name="pair", implementation=Implementation(Pair, [link]))
    block = PlanBlock(steps=(PlanStep(operation=child, output="pair.x"),))

    with pytest.raises(BiltLinearBlockError, match="SubSystemConstituent is not a primitive equation"):
        extract_linear_block(block)


def test_inequalities_are_not_silently_converted_to_equalities() -> None:
    @equation
    def inequality(x: float, y: float) -> bool:
        return x < y

    block = PlanBlock(steps=(PlanStep(operation=inequality, output="x"),))

    with pytest.raises(BiltLinearBlockError, match="not a binary equality"):
        extract_linear_block(block)


def test_duplicate_output_assignments_are_rejected() -> None:
    block = PlanBlock(
        steps=(PlanStep(operation=first, output="x"), PlanStep(operation=second, output="x")),
        is_cyclic=True,
    )

    with pytest.raises(BiltLinearBlockError, match="duplicate output assignments"):
        extract_linear_block(block)


def test_one_equation_cannot_supply_two_independent_rows() -> None:
    block = PlanBlock(
        steps=(PlanStep(operation=first, output="x"), PlanStep(operation=first, output="y")),
        is_cyclic=True,
    )

    with pytest.raises(BiltLinearBlockError, match="more than once"):
        extract_linear_block(block)


def test_output_must_belong_to_its_matched_equation() -> None:
    block = PlanBlock(steps=(PlanStep(operation=first, output="unrelated"),))

    with pytest.raises(BiltLinearBlockError, match="does not declare block output 'unrelated'"):
        extract_linear_block(block)
