import ast
import typing

import pytest

from bilt.core._equation import equation
from bilt.core._tensors._expression_lowering import lower_expression
from bilt.core._to_ast import tree_to_functiondef
from bilt.symbolic._domains import Domain
from bilt.symbolic._expressions import Expr
from bilt.symbolic._isolation._isolate import isolate


def _compile_and_execute_function(function_def: ast.FunctionDef) -> typing.Callable[..., object]:
    module = ast.Module(body=[function_def], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {"__builtins__": __builtins__}
    exec(compile(module, "<test_ast>", "exec"), namespace)  # noqa: S102
    return typing.cast("typing.Callable[..., object]", namespace[function_def.name])


def test_tree_to_functiondef_returns_expression_value() -> None:
    a = Expr.symbol("a")
    b = Expr.symbol("b")

    function_def = tree_to_functiondef(
        (a + b).rep,
        name="x",
        parameter_order=("a", "b", "x"),
    )
    compiled = _compile_and_execute_function(function_def)

    assert function_def.name == "x"
    assert [arg.arg for arg in function_def.args.args] == ["a", "b"]
    assert compiled(2.0, 3.0) == pytest.approx(5.0)


def test_equation_can_be_isolated_lowered_converted_and_executed() -> None:
    @equation
    def model(x: float, a: float, b: float, c: float) -> bool:
        return x == a + b * c

    isolated = isolate(model.__bilt_equation_tree__, Expr.symbol("x"))
    lowered = lower_expression(
        isolated.rhs,
        symbol_domains=dict.fromkeys(("a", "b", "c"), Domain.scalar()),
        canonical_order=Domain.scalar(),
    )
    function_def = tree_to_functiondef(
        lowered.tree,
        name="x",
        parameter_order=model.__bilt_parameters__,
    )
    compiled = _compile_and_execute_function(function_def)

    assert [arg.arg for arg in function_def.args.args] == ["a", "b", "c"]
    assert compiled(2.0, 3.0, 4.0) == pytest.approx(14.0)


def test_equation_can_generate_from_non_isolated_target() -> None:
    @equation
    def model(a: float, b: float, x: float, y: float) -> bool:
        return a + b * x == y

    isolated = isolate(model.__bilt_equation_tree__, Expr.symbol("x"))
    lowered = lower_expression(
        isolated.rhs,
        symbol_domains=dict.fromkeys(("a", "b", "y"), Domain.scalar()),
        canonical_order=Domain.scalar(),
    )
    function_def = tree_to_functiondef(
        lowered.tree,
        name="x",
        parameter_order=model.__bilt_parameters__,
    )
    compiled = _compile_and_execute_function(function_def)

    assert function_def.name == "x"
    assert [arg.arg for arg in function_def.args.args] == ["a", "b", "y"]
    assert compiled(2.0, 4.0, 3.0) == pytest.approx(0.25)