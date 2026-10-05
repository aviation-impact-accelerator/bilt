import pytest

from bilt.core._errors import BiltDomainBindingError
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._tensors._expression_lowering import lower_expression
from bilt.symbolic._atoms import ISEL
from bilt.symbolic._dimensions import Dim, coordinate_axis
from bilt.symbolic._domains import domain
from bilt.symbolic._expressions import tensor_symbol_expr
from bilt.symbolic._tree import axis_tree, int_tree


def test_sel_resolves_to_isel_when_coordinate_axis_is_bound() -> None:
    time = Dim("time")
    value = tensor_symbol_expr("value")

    with EvaluationContext() as evaluation:
        evaluation.domains.bind(
            coordinate_axis(
                time,
                (2024, 2025, 2026),
            )
        )

        lowered = lower_expression(
            value.sel(time, 2025),
            symbol_domains={
                "value": domain(time),
            },
            canonical_order=domain(time),
        )

    assert lowered.tree.head is ISEL
    assert lowered.tree.args[0] is value.tree
    assert lowered.tree.args[1] is axis_tree(0)
    assert lowered.tree.args[2] is int_tree(1)
    assert lowered.domain.is_scalar


def test_sel_requires_bound_coordinate_axis() -> None:
    time = Dim("time")
    value = tensor_symbol_expr("value")

    with (
        EvaluationContext(),
        pytest.raises(
            BiltDomainBindingError,
            match="Dimension time is not bound",
        ),
    ):
        lower_expression(
            value.sel(time, 2025),
            symbol_domains={
                "value": domain(time),
            },
            canonical_order=domain(time),
        )


def test_sel_rejects_coordinate_missing_from_bound_axis() -> None:
    time = Dim("time")
    value = tensor_symbol_expr("value")

    with EvaluationContext() as evaluation:
        evaluation.domains.bind(
            coordinate_axis(
                time,
                (2024, 2025, 2026),
            )
        )

        with pytest.raises(
            BiltDomainBindingError,
            match="not present",
        ):
            lower_expression(
                value.sel(time, 2030),
                symbol_domains={
                    "value": domain(time),
                },
                canonical_order=domain(time),
            )
