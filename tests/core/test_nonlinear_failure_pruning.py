import typing

import pytest

from bilt import PYTHON_BACKEND, Implementation, Interface, field, system, transform
from bilt._backend._errors import BiltUnsupportedNodeError
from bilt.core._structural_analysis._causalization_block_solver import _search as search
from bilt.core._structural_analysis._plan import PlanStep
from bilt.errors import BiltAnalysisError
from bilt.numerics import EvaluationOptions, RootOptions

if typing.TYPE_CHECKING:
    import ast
    from collections.abc import Iterator, Mapping

    import rustworkx as rx

    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.core._structural_analysis._realize import RealizationFailure


def test_new_transform_failure_reaches_the_active_nonlinear_search(monkeypatch: pytest.MonkeyPatch) -> None:
    @system
    class Value(metaclass=Interface):
        source: float = field()
        answer: float = field()

    @transform
    def answer(source: float) -> float:
        return source + 1

    original = search.candidate_assignments
    observed = []

    def observe(
        graph: rx.PyGraph, failures: Mapping[PlanStep, RealizationFailure], *, output: str | None
    ) -> Iterator[dict[str, BiltModelConstituent]]:
        for candidate in original(graph, failures, output=output):
            yield candidate
            # The real compiler has now failed, while this search is suspended.
            observed.append(PlanStep(answer, "answer") in failures)

    def refuse(_: ast.FunctionDef) -> typing.Never:
        message = "unsupported transform"
        raise BiltUnsupportedNodeError(message)

    monkeypatch.setattr(search, "candidate_assignments", observe)
    monkeypatch.setattr(PYTHON_BACKEND, "transpile", refuse)
    options = EvaluationOptions(nonlinear=RootOptions(variables={}, residual_scales={}))
    with (
        Implementation(Value, [answer]).with_options(options),
        pytest.raises(BiltAnalysisError, match="unsupported transform"),
    ):
        _ = Value(source=1).answer
    assert observed == [True]
