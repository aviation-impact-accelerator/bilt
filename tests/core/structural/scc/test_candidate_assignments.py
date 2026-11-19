import itertools
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
    assert {frozenset(assignment) for assignment in assignments} == {frozenset({"x", "y"})}
    expected_calls = 1
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


def test_unrelated_matching_edges_do_not_spawn_target_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = _graph()
    for index in range(12):
        attribute = graph.add_node(f"unused_{index}")
        operation = graph.add_node(y)
        graph.add_edge(attribute, operation, 0)

    original = search.find_matching
    calls = []

    def counted_matching(candidate: rx.PyGraph) -> dict[str, BiltModelConstituent]:
        calls.append(candidate)
        return original(candidate)

    monkeypatch.setattr(search, "find_matching", counted_matching)
    candidates = list(search.candidate_assignments(graph, {}, output="x"))
    assert len(candidates) == 1
    assert candidates[0]["x"] is x
    assert len(calls) == 1


def test_target_dependency_producer_alternatives_still_get_searched() -> None:
    @equation
    def target(x: float, y: float) -> bool:
        return x == y + 1

    @equation
    def source(y: float, z: float) -> bool:
        return y == z + 2

    graph = rx.PyGraph(multigraph=False)
    x_node = graph.add_node("x")
    y_node = graph.add_node("y")
    z_node = graph.add_node("z")
    target_node = graph.add_node(target)
    source_node = graph.add_node(source)
    graph.add_edge(x_node, target_node, 1)
    graph.add_edge(y_node, target_node, 0)
    graph.add_edge(y_node, source_node, 1)
    graph.add_edge(z_node, source_node, 0)

    assignments = list(search.candidate_assignments(graph, {}, output="x"))
    assert assignments[0]["x"] is target
    assert assignments[0]["y"] is source
    assert any(candidate.get("x") is target and "y" not in candidate for candidate in assignments)


def test_unmatched_target_can_reassign_competing_equation() -> None:
    @equation
    def relation(x: float, y: float) -> bool:
        return x == y + 1

    graph = rx.PyGraph(multigraph=False)
    x_node = graph.add_node("x")
    y_node = graph.add_node("y")
    relation_node = graph.add_node(relation)
    graph.add_edge(x_node, relation_node, 0)
    graph.add_edge(y_node, relation_node, 1)

    candidates = list(search.candidate_assignments(graph, {}, output="x"))
    assert candidates[0] == {"y": relation}
    assert any(candidate.get("x") is relation for candidate in candidates)


class _SmallEquation:
    def __init__(self, name: str, parameters: tuple[str, ...]) -> None:
        self.__bilt_name__ = name
        self.__bilt_parameters__ = parameters


def _exhaustive_candidates(graph: rx.PyGraph, *, target: str) -> list[dict[str, BiltModelConstituent]]:
    """Reference search that tries every matched edge, including unrelated ones."""
    target_node = next(node for node in graph.node_indices() if graph[node] == target)
    pending = [graph]
    visited: set[frozenset[tuple[int, int]]] = set()
    found = []
    while pending:
        candidate = pending.pop()
        if candidate.degree(target_node) == 0:
            continue
        edges = list(candidate.edge_list())
        signature = frozenset((min(a, b), max(a, b)) for a, b in edges)
        if signature in visited:
            continue
        visited.add(signature)
        assignments = search.find_matching(candidate)
        found.append(assignments)
        for first, second in edges:
            name, operation = (
                (candidate[first], candidate[second])
                if isinstance(candidate[first], str)
                else (candidate[second], candidate[first])
            )
            if assignments.get(name) is operation:
                alternative = candidate.copy()
                alternative.remove_edge(first, second)
                pending.append(alternative)
    return found


def _target_route(assignments: dict[str, BiltModelConstituent], target: str) -> tuple[tuple[str, str | None], ...]:
    route: dict[str, str | None] = {}
    pending = [target]
    while pending:
        attribute = pending.pop()
        if attribute in route:
            continue
        operation = assignments.get(attribute)
        route[attribute] = None if operation is None else operation.__bilt_name__
        if operation is not None:
            pending.extend(parameter for parameter in operation.__bilt_parameters__ if parameter != attribute)
    return tuple(sorted(route.items()))


def test_exhaustive_small_matching_routes_match_unpruned_search() -> None:
    """Retain target routes across all small incidence patterns, including cycles."""
    names = ("x", "y", "z")
    port_sets = [ports for size in range(1, len(names) + 1) for ports in itertools.combinations(names, size)]
    for relations in itertools.product(port_sets, repeat=3):
        if not any("x" in ports for ports in relations):
            continue
        graph = rx.PyGraph(multigraph=False)
        attributes = {name: graph.add_node(name) for name in names}
        for index, ports in enumerate(relations):
            operation = graph.add_node(typing.cast("BiltModelConstituent", _SmallEquation(f"relation_{index}", ports)))
            for name in ports:
                graph.add_edge(attributes[name], operation, 0)

        exhaustive = {_target_route(assignment, "x") for assignment in _exhaustive_candidates(graph, target="x")}
        pruned = {_target_route(assignment, "x") for assignment in search.candidate_assignments(graph, {}, output="x")}
        assert pruned == exhaustive, f"Target route missing for {relations!r}"


def test_failed_preferred_producer_keeps_other_route_available() -> None:
    @equation
    def alternative(x: float, a: float) -> bool:
        return x == a * 2

    graph = rx.PyGraph(multigraph=False)
    target = graph.add_node("x")
    graph.add_edge(target, graph.add_node(x), 1)
    graph.add_edge(target, graph.add_node(alternative), 1)

    failures: dict[PlanStep, RealizationFailure] = {}
    candidates = search.candidate_assignments(graph, failures, output="x")
    first = next(candidates)
    failures[PlanStep(operation=first["x"], output="x")] = RealizationFailure(
        failed_plan_step=PlanStep(operation=first["x"], output="x")
    )
    assert any(candidate.get("x") is not first["x"] for candidate in candidates)


def test_dedup_scope_preserves_all_small_target_routes() -> None:
    """Only suppress repeated yields; never lose distinct possible routes."""
    names = ("x", "y", "z")
    choices = [ports for count in (1, 2) for ports in itertools.combinations(names, count)]
    for relations in itertools.product(choices, repeat=2):
        graph = rx.PyGraph(multigraph=False)
        ports = {name: graph.add_node(name) for name in names}
        for index, attached in enumerate(relations):
            operation = graph.add_node(typing.cast("BiltModelConstituent", _SmallEquation(f"eq_{index}", attached)))
            for name in attached:
                graph.add_edge(ports[name], operation, 0)
        baseline = {
            _target_route(assignment, "x") for assignment in search.candidate_assignments(graph, {}, output="x")
        }
        with search.matching_dedup_scope():
            optimized = {
                _target_route(assignment, "x") for assignment in search.candidate_assignments(graph, {}, output="x")
            }
        assert optimized == baseline


def test_dedup_scope_restores_previous_state_after_exception() -> None:
    graph = _graph()
    ordinary = list(search.candidate_assignments(graph, {}, output=None))
    with pytest.raises(ValueError, match="aborted"):
        with search.matching_dedup_scope():
            assert list(search.candidate_assignments(graph, {}, output=None))
            raise ValueError("aborted")
    assert list(search.candidate_assignments(graph, {}, output=None)) == ordinary
