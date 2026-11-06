# ruff: noqa: SIM108
import pytest

from bilt.core._equation import equation
from bilt.symbolic._atoms import CONDITIONAL
from bilt.symbolic._expressions import Expr
from bilt.symbolic._tree import Tree, conditional_tree
from bilt.symbolic._walk import contains


def test_automatic_conditional_tracing_matches_symbolic_expression() -> None:
    @equation
    def relation(x: float, y: float) -> bool:
        if x > 0:
            value = x**2
        else:
            value = -x
        return y == value

    x, y = Expr.symbol("x"), Expr.symbol("y")
    expected = y == Expr(conditional_tree(x > 0, (x**2).tree, (-x).tree))
    assert relation.__bilt_equation_tree__ is expected
    assert relation.__bilt_equation_tree__.right.head is CONDITIONAL
    assert relation.__bilt_parameters__ == ("x", "y")


def test_dependencies_include_the_conditional_predicate() -> None:
    @equation
    def relation(x: float, y: float, threshold: float) -> bool:
        if x > threshold:
            selected = x
        else:
            selected = -x
        return y == selected

    tree = relation.__bilt_equation_tree__
    assert contains(tree, Expr.symbol("threshold").tree)
    assert contains(tree, Expr.symbol("x").tree)


def test_symbolic_tree_is_not_truthy() -> None:
    predicate = Expr.symbol("x") > 0
    assert isinstance(predicate, Tree)
    with pytest.raises(TypeError, match="Symbolic trees cannot be used as Python booleans"):
        bool(predicate)


def test_dynamic_symbolic_branch_without_source_fails_safely() -> None:
    namespace: dict[str, object] = {}
    exec(  # noqa: S102
        "def generated(x, y):\n"
        "    if x > 0:\n"
        "        value = x\n"
        "    else:\n"
        "        value = -x\n"
        "    return y == value\n",
        namespace,
    )
    generated = namespace["generated"]
    assert callable(generated)
    with pytest.raises(TypeError, match="Symbolic trees cannot be used as Python booleans"):
        equation(generated)
