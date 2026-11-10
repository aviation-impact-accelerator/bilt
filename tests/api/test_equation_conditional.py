# ruff: noqa: SIM108
import pytest

from bilt.core._equation import equation
from bilt.core._equation_ast import BiltEquationSyntaxError
from bilt.symbolic._atoms import CONDITIONAL
from bilt.symbolic._expressions import Expr
from bilt.symbolic._tree import Tree, conditional_tree
from bilt.symbolic._walk import contains

SHIFT = 2.0


def test_conditional_equation_traces_both_branches() -> None:
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


def test_conditional_equation_retains_original_numerical_call() -> None:
    @equation
    def relation(x: float, y: float) -> bool:
        if x > 0:
            value = x**2
        else:
            value = -x
        return y == value

    assert relation(2.0, 4.0)
    assert relation(-2.0, 2.0)
    assert not relation(2.0, -2.0)


def test_conditionals_can_use_local_bindings_and_module_globals() -> None:
    @equation
    def relation(x: float, y: float) -> bool:
        offset: float = x + SHIFT
        if offset > 0:
            value: float = offset**2
        else:
            value = -offset
        return y == value

    x = Expr.symbol("x")
    y = Expr.symbol("y")
    offset = x + SHIFT
    expected = y == Expr(conditional_tree(offset > 0, (offset**2).tree, (-offset).tree))
    assert relation.__bilt_equation_tree__ is expected
    assert relation(-3.0, 1.0)
    assert "_bilt_conditional" not in globals()


def test_regular_equations_preserve_existing_symbolic_tracing() -> None:
    @equation
    def relation(x: float, y: float) -> bool:
        return y == x + 1

    x, y = Expr.symbol("x"), Expr.symbol("y")
    assert relation.__bilt_equation_tree__ is (y == x + 1)


def test_symbolic_tree_cannot_be_used_as_python_boolean() -> None:
    x = Expr.symbol("x")
    predicate = x > 0
    assert isinstance(predicate, Tree)
    with pytest.raises(TypeError, match="Symbolic trees cannot be used as Python booleans"):
        bool(predicate)


def test_conditional_equation_rejects_closure_captures() -> None:
    offset = 3.0
    with pytest.raises(BiltEquationSyntaxError, match="captures closure variables"):

        @equation
        def relation(x: float, y: float) -> bool:
            if x > 0:
                value = x + offset
            else:
                value = -x
            return y == value


def test_conditional_equation_rejects_unsupported_return_flow() -> None:
    with pytest.raises(BiltEquationSyntaxError, match="exactly one return"):

        @equation
        def relation(x: float, y: float) -> bool:
            if x > 0:
                return y == x**2
            return y == -x


def test_equations_without_source_remain_supported() -> None:
    namespace: dict[str, object] = {}
    exec("def dynamically_defined(x, y):\n    return y == x + 1\n", namespace)  # noqa: S102
    function = namespace["dynamically_defined"]
    assert callable(function)
    normal = equation(function)
    assert normal.__bilt_equation_tree__.is_relation


def test_no_unnecessary_conditional_dependencies() -> None:
    @equation
    def relation(x: float, y: float, threshold: float) -> bool:
        if x > threshold:
            result = x
        else:
            result = -x
        return y == result

    tree = relation.__bilt_equation_tree__
    assert contains(tree, Expr.symbol("threshold").tree)
    assert contains(tree, Expr.symbol("x").tree)
