import pytest

from bilt.core._tensors._errors import TensorDomainError
from bilt.core._tensors._expression_lowering import (
    LoweredExpression,
    lower_expression,
)
from bilt.math import mean
from bilt.symbolic._atoms import MEAN
from bilt.symbolic._dimensions import Dim
from bilt.symbolic._domains import domain
from bilt.symbolic._expressions import Expr
from bilt.symbolic._tree import axis_tree


def test_mean_lowers_dimension_to_canonical_axis() -> None:
    location = Dim("location")
    time = Dim("time")
    value = Expr.symbol("value")

    actual = lower_expression(
        mean(value, time),
        symbol_domains={
            "value": domain(location, time),
        },
        canonical_order=domain(location, time),
    )

    assert actual.tree.head is MEAN
    assert actual.tree.args[0] is value.rep
    assert actual.tree.args[1] is axis_tree(1)
    assert actual.domain == domain(location)


def test_mean_uses_canonical_axis_not_local_domain_position() -> None:
    location = Dim("location")
    time = Dim("time")
    value = Expr.symbol("value")

    actual = lower_expression(
        mean(value, time),
        symbol_domains={
            "value": domain(location, time),
        },
        canonical_order=domain(time, location),
    )

    assert actual.tree.args[1] is axis_tree(0)
    assert actual.domain == domain(location)


def test_mean_over_only_dimension_returns_scalar_domain() -> None:
    time = Dim("time")
    value = Expr.symbol("value")

    actual = lower_expression(
        mean(value, time),
        symbol_domains={
            "value": domain(time),
        },
        canonical_order=domain(time),
    )

    assert actual.tree.args[1] is axis_tree(0)
    assert actual.domain.is_scalar


def test_mean_rejects_dimension_not_in_value_domain() -> None:
    location = Dim("location")
    time = Dim("time")
    value = Expr.symbol("value")

    with pytest.raises(TensorDomainError):
        lower_expression(
            mean(value, time),
            symbol_domains={
                "value": domain(location),
            },
            canonical_order=domain(location, time),
        )


def test_mean_lowers_inside_arithmetic_expression() -> None:
    location = Dim("location")
    time = Dim("time")
    value = Expr.symbol("value")
    offset = Expr.symbol("offset")

    actual = lower_expression(
        mean(value, time) + offset,
        symbol_domains={
            "value": domain(location, time),
            "offset": domain(),
        },
        canonical_order=domain(location, time),
    )

    lowered_mean = actual.tree.args[0]

    assert lowered_mean.head is MEAN
    assert lowered_mean.args[0] is value.rep
    assert lowered_mean.args[1] is axis_tree(1)
    assert actual.domain == domain(location)


def test_lowered_expression_rejects_symbolic_dimension() -> None:
    time = Dim("time")
    value = Expr.symbol("value")

    with pytest.raises(ValueError, match="cannot contain symbolic dimensions"):
        LoweredExpression(
            tree=mean(value, time).rep,
            domain=domain(),
        )
