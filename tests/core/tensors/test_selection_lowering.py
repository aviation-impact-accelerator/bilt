import pytest

from bilt.core._errors import BiltDomainBindingError
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._tensors._expression_lowering import lower_expression
from bilt.core._tensors._selection_lowering import resolve_static_selections
from bilt.symbolic._atoms import ISEL, SEL
from bilt.symbolic._dimensions import Dim, coordinate_axis
from bilt.symbolic._domains import domain
from bilt.symbolic._expressions import tensor_symbol_expr
from bilt.symbolic._tree import axis_tree, int_tree


def test_sel_resolves_to_isel_when_coordinate_axis_is_bound() -> None:
    time = Dim("time")
    value = tensor_symbol_expr("value")

    lowered = lower_expression(
        value.sel(time, 2025),
        symbol_domains={
            "value": domain(time),
        },
        canonical_order=domain(time),
    )

    with EvaluationContext() as evaluation:
        evaluation.domains.bind(
            coordinate_axis(
                time,
                (2024, 2025, 2026),
            )
        )

        resolved = resolve_static_selections(lowered.tree)

    assert resolved.head is ISEL
    assert resolved.args[0] is value.tree
    assert resolved.args[1] is axis_tree(0)
    assert resolved.args[2] is int_tree(1)


def test_sel_remains_unresolved_without_coordinate_axis() -> None:
    time = Dim("time")
    value = tensor_symbol_expr("value")

    lowered = lower_expression(
        value.sel(time, 2025),
        symbol_domains={
            "value": domain(time),
        },
        canonical_order=domain(time),
    )

    with EvaluationContext():
        resolved = resolve_static_selections(lowered.tree)

    assert resolved.head is SEL
    assert resolved.args == lowered.tree.args


def test_sel_rejects_coordinate_missing_from_bound_axis() -> None:
    time = Dim("time")
    value = tensor_symbol_expr("value")

    lowered = lower_expression(
        value.sel(time, 2030),
        symbol_domains={
            "value": domain(time),
        },
        canonical_order=domain(time),
    )

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
            resolve_static_selections(lowered.tree)
