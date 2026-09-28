from bilt.core._equation import equation
from bilt.core._tensors._tensors import Tensor  # noqa: TC001
from bilt.symbolic._dimensions import Dim
from bilt.symbolic._expressions import Expr, TensorExpr, tensor_symbol_expr
from bilt.symbolic._tree import mean_tree


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


def test_tensor_expression_mean_returns_plain_expression() -> None:
    time = Dim("time")
    value = tensor_symbol_expr("value")

    result = value.mean(time)

    assert type(result) is Expr
    assert result.tree is mean_tree(value.tree, time)


def test_tensor_expression_arithmetic_returns_plain_expression() -> None:
    left = tensor_symbol_expr("left")
    right = tensor_symbol_expr("right")

    result = left + right

    assert type(result) is Expr
