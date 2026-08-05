import pytest

from bilt.core._evaluation_context import (
    EvaluationContext,
    current_evaluation,
)


def test_current_evaluation_requires_active_context() -> None:
    with pytest.raises(
        RuntimeError,
        match="No Bilt evaluation is currently active",
    ):
        current_evaluation()


def test_nested_contexts_share_outer_evaluation() -> None:
    with EvaluationContext() as outer:
        assert current_evaluation() is outer

        with EvaluationContext() as nested:
            assert nested is outer
            assert current_evaluation() is outer
            assert nested.domains is outer.domains

        assert current_evaluation() is outer

    with pytest.raises(
        RuntimeError,
        match="No Bilt evaluation is currently active",
    ):
        current_evaluation()
