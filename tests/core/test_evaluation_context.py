import pytest

from bilt.core._evaluation_context import (
    EvaluationContext,
    current_evaluation,
)
from bilt.core._tensors._tensors import tensor
from bilt.errors import BiltDomainRegistryError, BiltNoActiveEvaluationError
from bilt.symbolic._dimensions import dim
from bilt.symbolic._domains import domain


def test_current_evaluation_requires_active_context() -> None:
    with pytest.raises(
        BiltNoActiveEvaluationError,
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
        BiltNoActiveEvaluationError,
        match="No Bilt evaluation is currently active",
    ):
        current_evaluation()


def test_canonical_order_for_cache_does_not_freeze_dimensions() -> None:

    before, after, extra = dim("before"), dim("after"), dim("extra")
    with EvaluationContext() as evaluation:
        evaluation.domains.bind(tensor([1.0], over=before))
        assert evaluation.canonical_order_for_cache().dims == (before,)

        # Cache inspection must not block adding a derived output dimension.
        evaluation.domains.bind(tensor([2.0], over=after))
        assert evaluation.canonical_order_for_cache().dims == (before, after)

        frozen = evaluation.canonical_order
        assert frozen.dims == (before, after)
        assert evaluation.canonical_order_for_cache() is frozen
        with pytest.raises(BiltDomainRegistryError):
            evaluation.domains.bind(tensor([3.0], over=extra))


def test_explicit_canonical_order_override_remains_supported() -> None:
    """The transform-lowering tests override this property with monkeypatch."""
    first, second = dim("first"), dim("second")
    with EvaluationContext() as evaluation:
        evaluation.domains.bind(tensor([[1.0]], over=(first, second)))
        assert evaluation.canonical_order_for_cache() == domain(first, second)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(evaluation, "canonical_order", domain(second, first))
            assert evaluation.canonical_order == domain(second, first)
            assert evaluation.canonical_order_for_cache() == domain(second, first)

        assert evaluation.canonical_order == domain(first, second)
