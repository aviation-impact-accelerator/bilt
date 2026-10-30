import pytest

from bilt import Implementation, Interface, Tensor, dim, equation, field, system, transform
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._structural_analysis._plan import PlanBlock, PlanStep
from bilt.core._structural_analysis._qualified_block import QualifiedBlock, QualifiedEquation
from bilt.core._structural_analysis._residual_block import BiltResidualBlockError, extract_residual_block
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._to_ast import tree_to_functiondef
from bilt.symbolic._atoms import ADD, NEG
from bilt.symbolic._tree import Tree


@equation
def product(x: float, y: float, p: float) -> bool:
    return x * y == p


@equation
def nonlinear(x: float, y: float, q: float) -> bool:
    return x**2 + 1 / y == q**2


@pytest.mark.parametrize("reverse", (False, True))
def test_local_residual_order_dependencies_and_original_sources(reverse: bool) -> None:
    steps = (PlanStep(product, "x"), PlanStep(nonlinear, "y"))
    block = PlanBlock(steps[::-1] if reverse else steps, is_cyclic=True)
    trees = (product.__bilt_equation_tree__, nonlinear.__bilt_equation_tree__)

    residual = extract_residual_block(block)

    assert residual.block is block
    assert residual.unknowns == ("x", "y")
    assert residual.inputs == frozenset({"p", "q"})
    assert tuple(row.source for row in residual.rows) == (product, nonlinear)
    assert tuple(row.label for row in residual.rows) == ("product", "nonlinear")
    for expression, tree in zip(residual.residuals, trees, strict=True):
        assert expression is Tree(ADD, (tree.left, Tree(NEG, (tree.right,))))
    # Evaluate only the constructed expressions to check subtraction direction.
    for expression, expected in zip(residual.residuals, (2, -4.75), strict=True):
        function = PYTHON_BACKEND.transpile(
            tree_to_functiondef(expression, name="residual", parameter_order=("x", "y", "p", "q"))
        )
        assert function.evaluate({"x": 2, "y": 4, "p": 6, "q": 3}) == expected
    assert product.__bilt_equation_tree__ is trees[0]
    assert nonlinear.__bilt_equation_tree__ is trees[1]


@pytest.mark.parametrize("reverse", (False, True))
def test_qualified_orders_and_instance_identity_are_retained(reverse: bool) -> None:
    rows = (QualifiedEquation(product, ("a",)), QualifiedEquation(product, ("b",)))
    rows = rows[::-1] if reverse else rows
    # Rectangular blocks are descriptions; no numerical solvability is claimed.
    unknowns = ("b.y", "a.x", "b.x")
    block = QualifiedBlock(rows, unknowns)

    residual = extract_residual_block(block)

    assert residual.block is block
    assert residual.rows is rows
    assert residual.unknowns == unknowns
    assert residual.inputs == frozenset({"a.y", "a.p", "b.p"})
    assert tuple(row.label for row in residual.rows) == tuple(row.label for row in rows)
    for expression, row in zip(residual.residuals, rows, strict=True):
        assert row.source is product
        assert expression.args[0] is row.tree.left
        assert expression.args[1].args[0] is row.tree.right


def test_cancelled_sides_keep_external_dependencies_and_original_expression() -> None:
    @equation
    def cancelled(x: float, p: float) -> bool:
        return x / p == x / p

    residual = extract_residual_block(PlanBlock((PlanStep(cancelled, "x"),)))

    assert residual.inputs == frozenset({"p"})
    expression = residual.residuals[0]
    assert expression.head is ADD
    assert expression.args[0] is cancelled.__bilt_equation_tree__.left
    assert expression.args[1].args[0] is cancelled.__bilt_equation_tree__.right


@pytest.mark.parametrize("qualified", (False, True))
@pytest.mark.parametrize("kind", ("inequality", "reduction", "external_selection"))
def test_unsupported_operations_are_rejected_even_in_external_inputs(*, qualified: bool, kind: str) -> None:
    time = dim("time")

    @equation
    def inequality(x: float, p: float) -> bool:
        return x < p

    @equation
    def reduction(x: Tensor, p: float) -> bool:
        return x.sum(time) == p

    @equation
    def external_selection(x: float, p: Tensor) -> bool:
        return x == p.isel(time, 0)

    source = {"inequality": inequality, "reduction": reduction, "external_selection": external_selection}[kind]
    block = (
        QualifiedBlock((QualifiedEquation(source, ("child",)),), ("child.x",))
        if qualified
        else PlanBlock((PlanStep(source, "x"),))
    )
    match = "not a binary equality" if kind == "inequality" else "unsupported residual operation"
    with pytest.raises(BiltResidualBlockError, match=match) as error:
        extract_residual_block(block)
    assert source.__bilt_name__ in str(error.value)
    if qualified:
        assert "child." in str(error.value)


def test_transform_and_opaque_subsystem_are_rejected() -> None:
    @transform
    def x(y: float) -> float:
        return y**2

    @system
    class Pair(metaclass=Interface):
        x: float = field()
        y: float = field()
        p: float = field()

    child = SubSystemConstituent(name="pair", implementation=Implementation(Pair, [product]))
    for operation, output in ((x, "x"), (child, "pair.x")):
        with pytest.raises(BiltResidualBlockError, match="is not a primitive equation"):
            extract_residual_block(PlanBlock((PlanStep(operation, output),)))


@pytest.mark.parametrize(
    ("steps", "match"),
    (
        ((PlanStep(product, "x"), PlanStep(nonlinear, "x")), "duplicate output assignments"),
        ((PlanStep(product, "x"), PlanStep(product, "y")), "more than once"),
        ((PlanStep(product, "missing"),), "does not declare block output"),
    ),
)
def test_invalid_local_matching_is_rejected(steps: tuple[PlanStep, ...], match: str) -> None:
    with pytest.raises(BiltResidualBlockError, match=match):
        extract_residual_block(PlanBlock(steps, is_cyclic=True))
