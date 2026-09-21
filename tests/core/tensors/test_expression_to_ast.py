import ast

from bilt.core._tensors._expression_lowering import lower_expression
from bilt.core._tensors._expression_to_ast import (
    MEAN_INTRINSIC_NAME,
    lowered_expression_to_ast,
)
from bilt.math import mean
from bilt.symbolic._dimensions import Dim
from bilt.symbolic._domains import domain
from bilt.symbolic._expressions import Expr


def test_mean_lowers_to_backend_intrinsic() -> None:
    location = Dim("location")
    time = Dim("time")
    value = Expr.symbol("value")

    lowered = lower_expression(
        mean(value, time),
        symbol_domains={
            "value": domain(location, time),
        },
        canonical_order=domain(location, time),
    )

    actual = lowered_expression_to_ast(lowered)

    length_keywords = 2

    assert isinstance(actual, ast.Call)
    assert isinstance(actual.func, ast.Name)
    assert actual.func.id == MEAN_INTRINSIC_NAME

    assert len(actual.args) == 1
    assert isinstance(actual.args[0], ast.Name)
    assert actual.args[0].id == "value"

    assert len(actual.keywords) == length_keywords

    axis = actual.keywords[0]
    assert axis.arg == "axis"
    assert isinstance(axis.value, ast.Constant)
    assert axis.value.value == 1

    keepdims = actual.keywords[1]
    assert keepdims.arg == "keepdims"
    assert isinstance(keepdims.value, ast.Constant)
    assert keepdims.value.value is True


def test_mean_lowers_inside_arithmetic_expression() -> None:
    location = Dim("location")
    time = Dim("time")
    value = Expr.symbol("value")
    offset = Expr.symbol("offset")

    lowered = lower_expression(
        mean(value, time) + offset,
        symbol_domains={
            "value": domain(location, time),
            "offset": domain(),
        },
        canonical_order=domain(location, time),
    )

    actual = lowered_expression_to_ast(lowered)

    assert isinstance(actual, ast.BinOp)
    assert isinstance(actual.op, ast.Add)

    assert isinstance(actual.left, ast.Call)
    assert isinstance(actual.left.func, ast.Name)
    assert actual.left.func.id == MEAN_INTRINSIC_NAME

    assert isinstance(actual.right, ast.Name)
    assert actual.right.id == "offset"
