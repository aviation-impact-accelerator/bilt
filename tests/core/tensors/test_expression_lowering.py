import pytest

from bilt.core._tensors._expression_lowering import (
    LoweredExpression,
    lower_expression,
)
from bilt.errors import BiltSymbolicLoweringError, BiltTensorDomainError
from bilt.symbolic._atoms import ISEL, MAX, MEAN, MIN, PROD, SUM, Atom
from bilt.symbolic._dimensions import Dim
from bilt.symbolic._domains import domain
from bilt.symbolic._expressions import Expr, tensor_symbol_expr
from bilt.symbolic._operator_semantics import describe_expression
from bilt.symbolic._tree import axis_tree, int_tree


@pytest.mark.parametrize(
    ("method_name", "operator"),
    (
        ("mean", MEAN),
        ("sum", SUM),
        ("min", MIN),
        ("max", MAX),
        ("prod", PROD),
    ),
)
def test_reduction_lowers_dimension_to_canonical_axis(
    method_name: str,
    operator: Atom,
) -> None:
    location = Dim("location")
    time = Dim("time")
    value = tensor_symbol_expr("value")

    actual = lower_expression(
        getattr(value, method_name)(time),
        symbol_domains={
            "value": domain(location, time),
        },
        canonical_order=domain(location, time),
    )

    assert actual.tree.head is operator
    assert actual.tree.args[0] is value.rep
    assert actual.tree.args[1] is axis_tree(1)
    assert actual.domain == domain(location)


def test_mean_uses_canonical_axis_not_local_domain_position() -> None:
    location = Dim("location")
    time = Dim("time")
    value = tensor_symbol_expr("value")

    actual = lower_expression(
        value.mean(time),
        symbol_domains={
            "value": domain(location, time),
        },
        canonical_order=domain(time, location),
    )

    assert actual.tree.args[1] is axis_tree(0)
    assert actual.domain == domain(location)


def test_mean_over_only_dimension_returns_scalar_domain() -> None:
    time = Dim("time")
    value = tensor_symbol_expr("value")

    actual = lower_expression(
        value.mean(time),
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
    value = tensor_symbol_expr("value")

    with pytest.raises(BiltTensorDomainError):
        lower_expression(
            value.mean(time),
            symbol_domains={
                "value": domain(location),
            },
            canonical_order=domain(location, time),
        )


def test_mean_lowers_inside_arithmetic_expression() -> None:
    location = Dim("location")
    time = Dim("time")
    value = tensor_symbol_expr("value")
    offset = Expr.symbol("offset")

    actual = lower_expression(
        value.mean(time) + offset,
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
    value = tensor_symbol_expr("value")

    with pytest.raises(
        BiltSymbolicLoweringError,
        match="cannot contain symbolic dimensions",
    ):
        LoweredExpression(
            tree=value.mean(time).rep,
            domain=domain(),
        )


def test_isel_lowers_dimension_to_canonical_axis() -> None:
    location = Dim("location")
    time = Dim("time")
    value = tensor_symbol_expr("value")

    actual = lower_expression(
        value.isel(time, 1),
        symbol_domains={
            "value": domain(location, time),
        },
        canonical_order=domain(location, time),
    )

    assert actual.tree.head is ISEL
    assert actual.tree.args[0] is value.tree
    assert actual.tree.args[1] is axis_tree(1)
    assert actual.tree.args[2] is int_tree(1)
    assert actual.domain == domain(location)


def test_isel_uses_canonical_axis() -> None:
    location = Dim("location")
    time = Dim("time")
    value = tensor_symbol_expr("value")

    actual = lower_expression(
        value.isel(time, 1),
        symbol_domains={
            "value": domain(time, location),
        },
        canonical_order=domain(location, time),
    )

    assert actual.tree.args[1] is axis_tree(1)
    assert actual.domain == domain(location)


@pytest.mark.parametrize("method_name", ("mean", "sum", "min", "max", "prod"))
def test_reduction_uses_same_domain_as_shared_semantics(method_name: str) -> None:
    """Nested operator descriptions and executable lowering agree."""
    location = Dim("location")
    time = Dim("time")
    value = tensor_symbol_expr("value")
    expression = getattr(value + 1, method_name)(time)
    domains = {"value": domain(location, time)}
    description = describe_expression(expression.rep, domains)
    lowered = lower_expression(
        expression,
        symbol_domains=domains,
        canonical_order=domain(time, location),
    )
    assert lowered.domain == description.domain == domain(location)
    assert lowered.tree.args[1] is axis_tree(0)


def test_dynamic_index_preserves_existing_lowering_behaviour() -> None:
    """A symbolic integer index is resolved at execution, not structural analysis."""
    time = Dim("time")
    value = tensor_symbol_expr("value")
    expression = value.isel(time, Expr.symbol("index"))
    lowered = lower_expression(
        expression,
        symbol_domains={"value": domain(time), "index": domain()},
        canonical_order=domain(time),
    )
    assert lowered.domain.is_scalar
    assert lowered.tree.args[1] is axis_tree(0)
    assert lowered.tree.args[2] is expression.rep.args[2]


def test_domain_semantics_for_nested_broadcast_and_reduction() -> None:
    location = Dim("location")
    time = Dim("time")
    value = tensor_symbol_expr("value")
    expression = (value * Expr.symbol("factor") + Expr.symbol("offset")).sum(time)
    symbol_domains = {
        "value": domain(location, time),
        "factor": domain(time),
        "offset": domain(location),
    }
    lowered = lower_expression(
        expression,
        symbol_domains=symbol_domains,
        canonical_order=domain(time, location),
    )
    expected = describe_expression(expression.rep, symbol_domains)
    assert lowered.domain == expected.domain == domain(location)
