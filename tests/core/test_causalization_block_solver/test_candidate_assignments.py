import typing

import pytest
import rustworkx as rx

from bilt import equation, transform
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis._causalization_block_solver import _backend as solver
from bilt.core._structural_analysis._causalization_block_solver import _search as search
from bilt.core._structural_analysis._plan import PlanStep
from bilt.core._structural_analysis._realize import RealizationFailure
from bilt.symbolic._domains import Domain

if typing.TYPE_CHECKING:
    from bilt.core._model_constituent import BiltModelConstituent


@transform
def x(a: float) -> float:
    return a + 1


@transform
def y(a: float) -> float:
    return a * 2


def _graph() -> rx.PyGraph:
    graph = rx.PyGraph(multigraph=False)
    for output, operation in (("x", x), ("y", y)):
        attribute = graph.add_node(output)
        producer = graph.add_node(operation)
        graph.add_edge(attribute, producer, 1)
    return graph


def _unexpected_matching(_: rx.PyGraph) -> dict[str, BiltModelConstituent]:
    message = "A producerless target must be pruned before matching."
    raise AssertionError(message)


@pytest.mark.parametrize("present", (False, True))
def test_missing_or_isolated_target_skips_matching(monkeypatch: pytest.MonkeyPatch, *, present: bool) -> None:
    graph = _graph()
    if present:
        graph.add_node("missing")
    monkeypatch.setattr(search, "find_matching", _unexpected_matching)
    assert list(search.candidate_assignments(graph, {}, output="missing")) == []


@pytest.mark.parametrize("preexisting", (False, True))
def test_failed_target_producer_prunes_retries(monkeypatch: pytest.MonkeyPatch, *, preexisting: bool) -> None:
    graph = _graph()
    step = PlanStep(operation=x, output="x")
    failure = RealizationFailure(failed_plan_step=step)
    failures = {step: failure} if preexisting else {}
    candidates = search.candidate_assignments(graph, failures, output="x")
    if not preexisting:
        assert next(candidates)["x"] is x
        failures[step] = failure
    monkeypatch.setattr(search, "find_matching", _unexpected_matching)
    assert list(candidates) == []


def test_prunes_targetless_descendants_and_preserves_other_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = _graph()
    original_edges = graph.edge_list()
    matching = search.find_matching
    calls = []

    def checked_matching(candidate: rx.PyGraph) -> dict[str, BiltModelConstituent]:
        node = next(node for node in candidate.node_indices() if candidate[node] == "x")
        assert candidate.degree(node) > 0
        calls.append(candidate)
        return matching(candidate)

    monkeypatch.setattr(search, "find_matching", checked_matching)
    assignments = list(search.candidate_assignments(graph, {}, output="x"))
    assert {frozenset(assignment) for assignment in assignments} == {frozenset({"x", "y"}), frozenset({"x"})}
    expected_calls = 2
    assert len(calls) == expected_calls
    assert graph.edge_list() == original_edges


def test_matching_omitting_target_still_explores_alternatives() -> None:
    @equation
    def relation(x: float, y: float) -> bool:
        return x == y + 1

    graph = rx.PyGraph(multigraph=False)
    x_node = graph.add_node("x")
    y_node = graph.add_node("y")
    producer = graph.add_node(relation)
    graph.add_edge(x_node, producer, 0)
    graph.add_edge(y_node, producer, 1)
    assert search.find_matching(graph) == {"y": relation}
    assignments = list(search.candidate_assignments(graph, {}, output="x"))
    assert any(assignment.get("x") is relation for assignment in assignments)


def test_supplied_target_needs_no_producer() -> None:
    analyser = solver.CausalizationStructuralAnalyzer([y], PYTHON_BACKEND)
    assert analyser.required_inputs_for_target({"x"}, "x") == frozenset({"x"})
    with EvaluationContext():
        case = analyser.analyse(EvaluationRequest(output="x", input_domains={"x": Domain.scalar()}))
        expected = 4.0
        assert PYTHON_BACKEND.chain_case(case).callable(x=expected) == expected
