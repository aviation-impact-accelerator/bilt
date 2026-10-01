import ast

import pytest

from bilt.core._tensors._expression_lowering import lower_expression
from bilt.core._to_ast import (
    MAX_INTRINSIC_NAME,
    MEAN_INTRINSIC_NAME,
    MIN_INTRINSIC_NAME,
    PROD_INTRINSIC_NAME,
    SUM_INTRINSIC_NAME,
    _tree_to_ast_expr,
)
from bilt.symbolic._dimensions import Dim
from bilt.symbolic._domains import domain
from bilt.symbolic._expressions import Expr, tensor_symbol_expr


@pytest.mark.parametrize(
    ("method_name", "intrinsic_name"),
    (
        ("mean", MEAN_INTRINSIC_NAME),
        ("sum", SUM_INTRINSIC_NAME),
        ("min", MIN_INTRINSIC_NAME),
        ("max", MAX_INTRINSIC_NAME),
        ("prod", PROD_INTRINSIC_NAME),
    ),
)
def test_reduction_lowers_to_backend_intrinsic(
    method_name: str,
    intrinsic_name: str,
) -> None:
    location = Dim("location")
    time = Dim("time")
    value = tensor_symbol_expr("value")

    lowered = lower_expression(
        getattr(value, method_name)(time),
        symbol_domains={
            "value": domain(location, time),
        },
        canonical_order=domain(location, time),
    )

    actual = _tree_to_ast_expr(lowered.tree)

    assert isinstance(actual, ast.Call)
    assert isinstance(actual.func, ast.Name)
    assert actual.func.id == intrinsic_name

    assert len(actual.args) == 1
    assert isinstance(actual.args[0], ast.Name)
    assert actual.args[0].id == "value"

    axis, keepdims = actual.keywords

    assert axis.arg == "axis"
    assert isinstance(axis.value, ast.Constant)
    assert axis.value.value == 1

    assert keepdims.arg == "keepdims"
    assert isinstance(keepdims.value, ast.Constant)
    assert keepdims.value.value is True


def test_mean_lowers_inside_arithmetic_expression() -> None:
    location = Dim("location")
    time = Dim("time")
    value = tensor_symbol_expr("value")
    offset = Expr.symbol("offset")

    lowered = lower_expression(
        value.mean(time) + offset,
        symbol_domains={
            "value": domain(location, time),
            "offset": domain(),
        },
        canonical_order=domain(location, time),
    )

    actual = _tree_to_ast_expr(lowered.tree)

    assert isinstance(actual, ast.BinOp)
    assert isinstance(actual.op, ast.Add)

    assert isinstance(actual.left, ast.Call)
    assert isinstance(actual.left.func, ast.Name)
    assert actual.left.func.id == MEAN_INTRINSIC_NAME

    assert isinstance(actual.right, ast.Name)
    assert actual.right.id == "offset"
