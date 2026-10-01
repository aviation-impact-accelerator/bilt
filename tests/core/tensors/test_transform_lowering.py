import ast
import inspect
import textwrap
import typing

import pytest

from bilt.core._evaluation_context import EvaluationContext
from bilt.core._tensors._errors import BiltTensorDomainError
from bilt.core._tensors._transform_lowering import lower_transform
from bilt.core._to_ast import (
    MAX_INTRINSIC_NAME,
    MEAN_INTRINSIC_NAME,
    MIN_INTRINSIC_NAME,
    PROD_INTRINSIC_NAME,
    SUM_INTRINSIC_NAME,
)
from bilt.symbolic._dimensions import Dim
from bilt.symbolic._domains import Domain, domain

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._tensors._tensors import Tensor


def _function_ast(
    function: collections.abc.Callable[..., object],
) -> ast.FunctionDef:
    return typing.cast(
        "ast.FunctionDef",
        ast.parse(
            textwrap.dedent(
                inspect.getsource(function),
            )
        ).body[0],
    )


def _lower_transform(
    function: collections.abc.Callable[..., object],
    *,
    symbol_domains: collections.abc.Mapping[str, Domain],
    canonical_order: Domain,
) -> tuple[ast.FunctionDef, Domain]:
    with EvaluationContext() as evaluation, pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            evaluation,
            "canonical_order",
            canonical_order,
        )
        return lower_transform(
            _function_ast(function),
            function=function,
            symbol_domains=symbol_domains,
        )


def test_lower_transform_lowers_mean_to_backend_intrinsic() -> None:
    location = Dim("location")
    time = Dim("time")

    def averaged(values: Tensor) -> Tensor:
        return values.mean(time)

    function_ast, output_domain = _lower_transform(
        averaged,
        symbol_domains={
            "values": domain(location, time),
        },
        canonical_order=domain(location, time),
    )

    assert output_domain == domain(location)

    return_statement = function_ast.body[0]
    assert isinstance(return_statement, ast.Return)

    expression = return_statement.value
    assert isinstance(expression, ast.Call)
    assert isinstance(expression.func, ast.Name)
    assert expression.func.id == MEAN_INTRINSIC_NAME

    assert len(expression.args) == 1
    assert isinstance(expression.args[0], ast.Name)
    assert expression.args[0].id == "values"

    axis_keyword, keepdims_keyword = expression.keywords

    assert axis_keyword.arg == "axis"
    assert isinstance(axis_keyword.value, ast.Constant)
    assert axis_keyword.value.value == 1

    assert keepdims_keyword.arg == "keepdims"
    assert isinstance(keepdims_keyword.value, ast.Constant)
    assert keepdims_keyword.value.value is True


def test_lower_transform_lowers_other_reductions_to_backend_intrinsics() -> None:
    location = Dim("location")
    time = Dim("time")

    def summed(values: Tensor) -> Tensor:
        return values.sum(time)

    def minimum(values: Tensor) -> Tensor:
        return values.min(time)

    def maximum(values: Tensor) -> Tensor:
        return values.max(time)

    def product(values: Tensor) -> Tensor:
        return values.prod(time)

    cases = (
        (summed, SUM_INTRINSIC_NAME),
        (minimum, MIN_INTRINSIC_NAME),
        (maximum, MAX_INTRINSIC_NAME),
        (product, PROD_INTRINSIC_NAME),
    )

    for function, intrinsic_name in cases:
        function_ast, output_domain = _lower_transform(
            function,
            symbol_domains={
                "values": domain(location, time),
            },
            canonical_order=domain(location, time),
        )

        assert output_domain == domain(location)

        return_statement = function_ast.body[0]
        assert isinstance(return_statement, ast.Return)

        expression = return_statement.value
        assert isinstance(expression, ast.Call)
        assert isinstance(expression.func, ast.Name)
        assert expression.func.id == intrinsic_name


def test_lower_transform_uses_canonical_axis() -> None:
    location = Dim("location")
    time = Dim("time")

    def averaged(values: Tensor) -> Tensor:
        return values.mean(time)

    function_ast, output_domain = _lower_transform(
        averaged,
        symbol_domains={
            "values": domain(location, time),
        },
        canonical_order=domain(time, location),
    )

    assert output_domain == domain(location)

    return_statement = function_ast.body[0]
    assert isinstance(return_statement, ast.Return)

    expression = return_statement.value
    assert isinstance(expression, ast.Call)

    axis_keyword = next(keyword for keyword in expression.keywords if keyword.arg == "axis")

    assert isinstance(axis_keyword.value, ast.Constant)
    assert axis_keyword.value.value == 0


def test_lower_transform_lowers_mean_inside_arithmetic() -> None:
    location = Dim("location")
    time = Dim("time")

    def result(
        values: Tensor,
        offset: Tensor,
    ) -> Tensor:
        return values.mean(time) + offset

    function_ast, output_domain = _lower_transform(
        result,
        symbol_domains={
            "values": domain(location, time),
            "offset": domain(),
        },
        canonical_order=domain(location, time),
    )

    assert output_domain == domain(location)

    return_statement = function_ast.body[0]
    assert isinstance(return_statement, ast.Return)

    expression = return_statement.value
    assert isinstance(expression, ast.BinOp)
    assert isinstance(expression.op, ast.Add)

    assert isinstance(expression.left, ast.Call)
    assert isinstance(expression.left.func, ast.Name)
    assert expression.left.func.id == MEAN_INTRINSIC_NAME

    assert isinstance(expression.right, ast.Name)
    assert expression.right.id == "offset"


def test_lower_transform_propagates_broadcast_domain() -> None:
    location = Dim("location")
    time = Dim("time")

    def result(
        values: Tensor,
        offset: Tensor,
    ) -> Tensor:
        return values.mean(time) + offset

    _, output_domain = _lower_transform(
        result,
        symbol_domains={
            "values": domain(time),
            "offset": domain(location, time),
        },
        canonical_order=domain(location, time),
    )

    assert output_domain == domain(location, time)


def test_lower_transform_strips_annotations() -> None:
    time = Dim("time")

    def averaged(values: Tensor) -> Tensor:
        return values.mean(time)

    function_ast, _ = _lower_transform(
        averaged,
        symbol_domains={
            "values": domain(time),
        },
        canonical_order=domain(time),
    )

    assert function_ast.decorator_list == []
    assert function_ast.returns is None
    assert all(argument.annotation is None for argument in function_ast.args.args)


def test_lower_transform_rejects_dimension_not_in_value_domain() -> None:
    location = Dim("location")
    time = Dim("time")

    def averaged(values: Tensor) -> Tensor:
        return values.mean(time)

    with pytest.raises(
        BiltTensorDomainError,
        match="Cannot reduce dimension",
    ):
        _lower_transform(
            averaged,
            symbol_domains={
                "values": domain(location),
            },
            canonical_order=domain(location, time),
        )


def test_lower_transform_preserves_plain_name_return() -> None:
    time = Dim("time")

    def result(values: Tensor) -> Tensor:
        return values

    function_ast, output_domain = _lower_transform(
        result,
        symbol_domains={
            "values": domain(time),
        },
        canonical_order=domain(time),
    )

    assert output_domain == domain(time)

    return_statement = function_ast.body[0]
    assert isinstance(return_statement, ast.Return)
    assert isinstance(return_statement.value, ast.Name)
    assert return_statement.value.id == "values"


def test_lower_transform_lowers_mean_inside_multiplication() -> None:
    location = Dim("location")
    time = Dim("time")
    expected_scale = 2

    def result(values: Tensor) -> Tensor:
        return 2 * values.mean(time)

    function_ast, output_domain = _lower_transform(
        result,
        symbol_domains={
            "values": domain(location, time),
        },
        canonical_order=domain(location, time),
    )

    assert output_domain == domain(location)

    return_statement = function_ast.body[0]
    assert isinstance(return_statement, ast.Return)

    expression = return_statement.value
    assert isinstance(expression, ast.BinOp)
    assert isinstance(expression.op, ast.Mult)

    assert isinstance(expression.left, ast.Constant)
    assert expression.left.value == expected_scale

    assert isinstance(expression.right, ast.Call)
    assert isinstance(expression.right.func, ast.Name)
    assert expression.right.func.id == MEAN_INTRINSIC_NAME
