import itertools
import typing

import pytest

from bilt import Implementation, Interface, Tensor, dim, domain, equation, field, system, transform
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._errors import BiltAnalysisError
from bilt.core._structural_analysis._causalization_block_solver._expand_structure import StructuralExpansion
from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph
from bilt.core._structural_analysis._linear_block import BiltLinearBlockError
from bilt.core._structural_analysis._linear_domains import declared_linear_domains
from bilt.core._structural_analysis._plan import PlanBlock, PlanStep
from bilt.core._structural_analysis._qualified_block import QualifiedBlock, QualifiedEquation
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._to_ast import tree_to_functiondef
from bilt.symbolic._algebra._expr_ops import subtract
from bilt.symbolic._atoms import CoordinateType, DimensionType, QuantityType, SymbolType
from bilt.symbolic._domains import Domain
from bilt.symbolic._tree import symbol_tree
from bilt.symbolic._walk import subtrees
from bilt.units.dimensions import LENGTH
from bilt.units.units import meter

if typing.TYPE_CHECKING:
    from bilt.core._equation import BasicEquation
    from bilt.symbolic._tree import Tree

TIME = dim("time")
SITE = dim("site")
OverTime = typing.Annotated[Tensor, domain(TIME)]
OverSite = typing.Annotated[Tensor, domain(SITE)]
Grid = typing.Annotated[Tensor, domain(SITE, TIME)]
ReverseGrid = typing.Annotated[Tensor, domain(TIME, SITE)]
Shared = typing.Annotated[float, Domain.scalar()]
LengthArray = typing.Annotated[Tensor, LENGTH, domain(TIME)]


@system
class Pair(metaclass=Interface):
    x: Tensor = field()
    y: Tensor = field()
    p: Tensor = field()


@equation
def balance(x: Tensor, y: Tensor, p: Tensor) -> bool:
    return x + y == p


@equation
def feedback(inner: Pair, q: float) -> bool:
    return inner.x - inner.y == q


def _symbols(tree: Tree) -> set[str]:
    return {typing.cast("str", node.head.value) for node in subtrees(tree) if node.head.atom_type is SymbolType}


def _residual(row: QualifiedEquation, values: dict[str, float]) -> float:
    expression = subtract(row.tree.left, row.tree.right)
    function = PYTHON_BACKEND.transpile(
        tree_to_functiondef(expression, name="residual", parameter_order=row.parameters)
    )
    return typing.cast("float", function.evaluate(values))


@pytest.mark.parametrize("reverse_rows", (False, True))
@pytest.mark.parametrize("reverse_columns", (False, True))
def test_parent_and_child_equations_share_boundary_names_and_retain_explicit_order(
    *, reverse_rows: bool, reverse_columns: bool
) -> None:
    rows = (QualifiedEquation(feedback), QualifiedEquation(balance, ("inner",)))
    unknowns = ("inner.x", "inner.y")
    rows = rows[::-1] if reverse_rows else rows
    unknowns = unknowns[::-1] if reverse_columns else unknowns
    block = QualifiedBlock(rows, unknowns)

    assert block.rows == rows
    assert block.unknowns == unknowns
    assert block.inputs == frozenset({"inner.p", "q"})
    for row in rows:
        assert {"inner.x", "inner.y"} <= _symbols(row.tree)
        for x, y in itertools.product((-2.0, 0.0, 3.0), repeat=2):
            values = (
                {"inner.x": x, "inner.y": y, "inner.p": 7.0} if row.path else {"inner.x": x, "inner.y": y, "q": 2.0}
            )
            expected = x + y - 7 if row.path else x - y - 2
            assert _residual(row, values) == pytest.approx(expected)


def test_nested_names_are_relative_to_the_equation_owner() -> None:
    row = QualifiedEquation(feedback, ("rack", "module"))
    assert row.parameters == ("rack.module.inner.x", "rack.module.inner.y", "rack.module.q")
    assert _symbols(row.tree) == set(row.parameters)
    assert row.path == ("rack", "module")
    assert row.source is feedback
    assert row.label == "rack.module.feedback"


def test_root_qualification_reuses_the_original_tree() -> None:
    row = QualifiedEquation(balance)
    assert row.tree is balance.__bilt_equation_tree__
    assert row.parameters == balance.__bilt_parameters__
    assert row.label == balance.__bilt_name__


def test_shared_implementation_equations_remain_distinct_by_instance_path() -> None:
    left = QualifiedEquation(balance, ("left",))
    right = QualifiedEquation(balance, ("right",))
    block = QualifiedBlock((left, right), ("left.x", "right.x"))
    assert left.source is right.source
    assert left != right
    assert block.inputs == frozenset({"left.y", "left.p", "right.y", "right.p"})
    assert _symbols(left.tree).isdisjoint(_symbols(right.tree))


def test_distinct_equations_with_the_same_name_and_tree_are_not_deduplicated() -> None:
    def make_equation() -> BasicEquation:
        @equation
        def same(x: float, p: float) -> bool:
            return x == p

        return same

    first = QualifiedEquation(make_equation(), ("inner",))
    second = QualifiedEquation(make_equation(), ("inner",))
    assert first.label == second.label
    assert first.tree is second.tree
    block = QualifiedBlock((first, second), ("inner.x",))
    assert len(set(block.rows)) == len((first, second))


def test_qualification_preserves_shared_subexpressions_and_source_metadata() -> None:
    @equation
    def repeated(x: Tensor, p: Tensor) -> bool:
        return (x + p) * (x + p) == p

    original = repeated.__bilt_equation_tree__
    parameters = repeated.__bilt_parameters__
    row = QualifiedEquation(repeated, ("inner",))
    assert row.tree.left.args[0] is row.tree.left.args[1]
    assert repeated.__bilt_equation_tree__ is original
    assert repeated.__bilt_parameters__ is parameters
    assert _symbols(original) == {"x", "p"}
    assert _symbols(row.tree) == {"inner.x", "inner.p"}


def test_dimension_and_coordinate_names_are_not_variable_symbols() -> None:
    same_name = dim("x")

    @equation
    def selected(x: Tensor, p: float) -> bool:
        return x.sel(same_name, "x") == p

    row = QualifiedEquation(selected, ("inner",))
    original_leaves = {node for node in subtrees(selected.__bilt_equation_tree__) if node.is_leaf}
    retained = {node for node in original_leaves if node.head.atom_type in (DimensionType, CoordinateType)}
    assert retained
    assert retained <= subtrees(row.tree)
    assert _symbols(row.tree) == {"inner.x", "inner.p"}


def test_quantity_constants_and_unit_declarations_are_preserved() -> None:
    @equation
    def offset(x: LengthArray, p: LengthArray) -> bool:
        return x + 1.0 * meter == p

    row = QualifiedEquation(offset, ("inner",))
    quantities = {node for node in subtrees(offset.__bilt_equation_tree__) if node.head.atom_type is QuantityType}
    assert quantities
    assert quantities <= subtrees(row.tree)
    assert row.source.__bilt_port_declarations__["x"].dimensionality is LENGTH
    assert dict(row.domain_hints) == {"inner.x": domain(TIME), "inner.p": domain(TIME)}


@equation
def hinted(x: OverTime, y: Shared, p: Tensor) -> bool:
    return x + y == p


def test_domain_hints_are_qualified_without_inventing_missing_hints() -> None:
    row = QualifiedEquation(hinted, ("rack", "inner"))
    assert dict(row.domain_hints) == {"rack.inner.x": domain(TIME), "rack.inner.y": Domain.scalar()}
    assert QualifiedEquation(balance, ("inner",)).domain_hints == ()


def test_local_hint_reader_and_qualified_block_use_the_same_hint_rules() -> None:
    block = QualifiedBlock((QualifiedEquation(hinted),), ("x",))
    plan = PlanBlock((PlanStep(hinted, "x"),))
    assert dict(block.domain_hints) == declared_linear_domains(plan)


def test_sibling_domain_hints_do_not_conflict_on_local_names() -> None:
    @equation
    def other(x: OverSite, y: Shared, p: Tensor) -> bool:
        return x + y == p

    block = QualifiedBlock(
        (QualifiedEquation(hinted, ("left",)), QualifiedEquation(other, ("right",))), ("left.x", "right.x")
    )
    assert dict(block.domain_hints)["left.x"] == domain(TIME)
    assert dict(block.domain_hints)["right.x"] == domain(SITE)


def test_conflicting_hints_for_one_qualified_unknown_are_rejected() -> None:
    @equation
    def other(x: OverSite, p: Tensor) -> bool:
        return x == p

    with pytest.raises(BiltLinearBlockError, match=r"Conflicting.*'inner.x'"):
        QualifiedBlock((QualifiedEquation(hinted, ("inner",)), QualifiedEquation(other, ("inner",))), ("inner.x",))


def test_equivalent_axis_sets_with_different_order_keep_existing_hint_policy() -> None:
    @equation
    def first(x: Grid, p: Tensor) -> bool:
        return x == p

    @equation
    def second(x: ReverseGrid, q: Tensor) -> bool:
        return x == q

    rows = (QualifiedEquation(first, ("inner",)), QualifiedEquation(second, ("inner",)))
    block = QualifiedBlock(rows, ("inner.x",))
    assert dict(block.domain_hints)["inner.x"] == domain(TIME, SITE)


def test_multiple_domain_hints_on_one_parameter_remain_an_error() -> None:
    @equation
    def ambiguous(x: typing.Annotated[Tensor, domain(TIME), domain(SITE)], p: Tensor) -> bool:
        return x == p

    with pytest.raises(BiltLinearBlockError, match="Conflicting linear-block domain hints"):
        QualifiedEquation(ambiguous, ("inner",))


@pytest.mark.parametrize("path", (("",), ("outer.inner",)))
def test_ambiguous_instance_path_components_are_rejected(path: tuple[str, ...]) -> None:
    with pytest.raises(BiltAnalysisError, match="nonempty subsystem names without dots"):
        QualifiedEquation(balance, path)


@pytest.mark.parametrize(
    "invalid", ("empty_rows", "empty_columns", "duplicate_row", "duplicate_column", "missing_column")
)
def test_invalid_block_contracts_are_rejected(invalid: str) -> None:
    row = QualifiedEquation(balance, ("inner",))
    rows = () if invalid == "empty_rows" else (row, row) if invalid == "duplicate_row" else (row,)
    columns = {
        "empty_columns": (),
        "duplicate_column": ("inner.x", "inner.x"),
        "missing_column": ("unknown",),
    }.get(invalid, ("inner.x",))
    with pytest.raises(BiltAnalysisError):
        QualifiedBlock(rows, columns)


def test_transforms_are_not_reinterpreted_as_equation_rows() -> None:
    @transform
    def x(p: float) -> float:
        return p + 1

    with pytest.raises(BiltAnalysisError, match="primitive equation"):
        QualifiedEquation(typing.cast("BasicEquation", x), ("inner",))


def test_undeclared_symbol_is_not_silently_left_in_another_namespace() -> None:
    from bilt.symbolic._expressions import Expr  # noqa: PLC0415

    free = Expr.symbol("free")

    @equation
    def undeclared(x: float) -> bool:
        return x == free

    with pytest.raises(BiltAnalysisError, match="undeclared equation symbol 'free'"):
        QualifiedEquation(undeclared, ("inner",))


def test_rows_can_be_built_from_actual_parent_child_structural_expansion() -> None:
    child = SubSystemConstituent("inner", Implementation(Pair, [balance]))
    expansion = StructuralExpansion(FullIncidenceGraph([child, feedback]), {"inner.p", "q"}, "inner.x")
    expansion.collect()
    constraints = expansion.snapshot().constraints
    assert len(constraints) == len((balance, feedback))
    rows = tuple(QualifiedEquation(typing.cast("BasicEquation", row.operation), row.path) for row in constraints)
    block = QualifiedBlock(rows, ("inner.x", "inner.y"))
    assert block.inputs == frozenset({"inner.p", "q"})
    assert {row.path for row in rows} == {(), ("inner",)}
    assert {row.source for row in rows} == {balance, feedback}
    assert symbol_tree("inner.x") in subtrees(next(row.tree for row in rows if row.path))
