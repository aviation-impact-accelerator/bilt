import typing

import pytest

from bilt.core._equation import equation
from bilt.core._tensors._tensors import Tensor  # noqa: TC001
from bilt.symbolic._dimensions import Dim
from bilt.symbolic._expressions import Expr, TensorExpr, tensor_symbol_expr
from bilt.symbolic._tree import (
    Tree,
    coordinate_tree,
    int_tree,
    isel_tree,
    max_tree,
    mean_tree,
    min_tree,
    prod_tree,
    sel_tree,
    sum_tree,
)

if typing.TYPE_CHECKING:
    import collections.abc


def test_tensor_parameter_uses_tensor_expression() -> None:
    @equation
    def relation(
        value: Tensor,
        result: Tensor,
    ) -> bool:
        return result == value

    assert isinstance(
        relation.__bilt_equation_symbols__["value"],
        TensorExpr,
    )
    assert isinstance(
        relation.__bilt_equation_symbols__["result"],
        TensorExpr,
    )


def test_non_tensor_parameter_uses_plain_expression() -> None:
    @equation
    def relation(
        value: float,
        result: float,
    ) -> bool:
        return result == value

    assert type(relation.__bilt_equation_symbols__["value"]) is Expr
    assert type(relation.__bilt_equation_symbols__["result"]) is Expr


@pytest.mark.parametrize(
    ("method_name", "tree_builder"),
    (
        ("mean", mean_tree),
        ("sum", sum_tree),
        ("min", min_tree),
        ("max", max_tree),
        ("prod", prod_tree),
    ),
)
def test_tensor_expression_reduction_preserves_tensor_expression(
    method_name: str,
    tree_builder: collections.abc.Callable[[Tree, Dim], Tree],
) -> None:
    time = Dim("time")
    value = tensor_symbol_expr("value")

    result = getattr(value, method_name)(time)

    assert type(result) is TensorExpr
    assert result.tree is tree_builder(value.tree, time)


def test_tensor_expression_arithmetic_preserves_tensor_expression() -> None:
    left = tensor_symbol_expr("left")
    right = tensor_symbol_expr("right")

    result = left + right

    assert type(result) is TensorExpr


def test_tensor_expression_isel_preserves_tensor_expression() -> None:
    time = Dim("time")
    value = tensor_symbol_expr("value")

    result = value.isel(time, 1)

    assert type(result) is TensorExpr
    assert result.tree is isel_tree(
        value.tree,
        time,
        int_tree(1),
    )


def test_tensor_expression_sel_preserves_tensor_expression() -> None:
    time = Dim("time")
    value = tensor_symbol_expr("value")

    result = value.sel(time, 2025)

    assert type(result) is TensorExpr
    assert result.tree is sel_tree(
        value.tree,
        time,
        coordinate_tree(2025),
    )
