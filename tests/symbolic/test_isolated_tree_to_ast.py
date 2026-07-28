import ast
import typing

import pytest

from bilt.core._equation import equation
from bilt.symbolic._atoms import EQUALS
from bilt.symbolic._expressions import Expr
from bilt.symbolic._isolation import equation_to_ast
from bilt.symbolic._isolation._errors import TargetNotFoundError
from bilt.symbolic._isolation._to_ast import isolated_tree_to_functiondef
from bilt.symbolic._isolation._types import IsolatedTree


def _compile_and_execute_function(function_def: ast.FunctionDef) -> typing.Callable[..., object]:
    module = ast.Module(body=[function_def], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {"__builtins__": __builtins__}
    exec(compile(module, "<test_ast>", "exec"), namespace)  # noqa: S102
    return typing.cast("typing.Callable[..., object]", namespace[function_def.name])


def test_isolated_tree_to_functiondef_returns_rhs_value() -> None:
    x = Expr.symbol("x")
    a = Expr.symbol("a")
    b = Expr.symbol("b")
    isolated = IsolatedTree(relation_head=EQUALS, target=x, rhs=a + b)

    function_def = isolated_tree_to_functiondef(isolated, ("a", "b", "x"))
    compiled = _compile_and_execute_function(function_def)

    assert function_def.name == "x"
    assert [arg.arg for arg in function_def.args.args] == ["a", "b"]
    assert compiled(2.0, 3.0) == pytest.approx(5.0)


def test_equation_can_be_isolated_then_converted_and_executed() -> None:
    @equation
    def model(x: float, a: float, b: float, c: float) -> bool:
        return x == a + b * c

    function_def = equation_to_ast(model, "x")
    compiled = _compile_and_execute_function(function_def)

    assert [arg.arg for arg in function_def.args.args] == ["a", "b", "c"]
    assert compiled(2.0, 3.0, 4.0) == pytest.approx(14.0)


def test_equation_can_generate_from_non_isolated_target() -> None:
    @equation
    def model(a: float, b: float, x: float, y: float) -> bool:
        return a + b * x == y

    function_def = equation_to_ast(model, "x")
    compiled = _compile_and_execute_function(function_def)

    assert function_def.name == "x"
    assert [arg.arg for arg in function_def.args.args] == ["a", "b", "y"]
    assert compiled(2.0, 4.0, 3.0) == pytest.approx(0.25)


def test_equation_to_ast_rejects_unknown_target_name() -> None:
    @equation
    def model(x: float, a: float) -> bool:
        return x == a

    with pytest.raises(TargetNotFoundError, match="is not a parameter"):
        equation_to_ast(model, "missing")
