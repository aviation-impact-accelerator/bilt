import dataclasses
import typing

import pytest

from bilt import Implementation, Interface, equation, field, system, transform
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._errors import BiltAnalysisError
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis._causalization_block_solver import _backend as solver
from bilt.core._structural_analysis._causalization_block_solver import _matching as matching
from bilt.core._structural_analysis._causalization_block_solver._expand_structure import (
    ExpandedConstraint,
    StructuralExpansion,
)
from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph
from bilt.core._system import _implementation as implementation_module
from bilt.symbolic._domains import Domain

if typing.TYPE_CHECKING:
    import collections.abc

    import rustworkx as rx

    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.core._structural_analysis._plan import PlanStep
    from bilt.core._structural_analysis._realize import RealizationFailure


@system
class Ports(metaclass=Interface):
    inlet: float = typing.cast("float", field())
    outlet: float = typing.cast("float", field())
    heat: float = typing.cast("float", field())
    flow: float = typing.cast("float", field())
    capacity: float = typing.cast("float", field())


@equation
def exchange(inlet: float, outlet: float, heat: float, flow: float, capacity: float) -> bool:
    return heat == flow * capacity * (outlet - inlet)


@equation
def supply(inlet: float, left: Ports) -> bool:
    return left.inlet == inlet


@equation
def connection(left: Ports, right: Ports) -> bool:
    return right.inlet == left.outlet


@equation
def discharge(outlet: float, right: Ports) -> bool:
    return outlet == right.outlet


@equation
def left_heat(heat: float, left: Ports) -> bool:
    return left.heat == heat / 2


@equation
def right_heat(heat: float, right: Ports) -> bool:
    return right.heat == heat / 2


@equation
def left_capacity(capacity: float, left: Ports) -> bool:
    return left.capacity == capacity


@equation
def right_capacity(capacity: float, right: Ports) -> bool:
    return right.capacity == capacity


@equation
def left_flow(flow: float, left: Ports) -> bool:
    return left.flow == flow


@equation
def right_flow(flow: float, right: Ports) -> bool:
    return right.flow == flow


def _nested(depth: int, *, reverse: bool = False) -> Implementation:
    implementation = Implementation(Ports, [exchange])
    for _ in range(depth):
        operations = [
            supply,
            left_heat,
            right_heat,
            left_capacity,
            right_capacity,
            connection,
            left_flow,
            right_flow,
            discharge,
        ]
        if reverse:
            operations.reverse()
        implementation = Implementation(
            Ports, operations, internal_implementations=dict.fromkeys(("left", "right"), implementation)
        )
    return implementation


@pytest.mark.parametrize("reverse", (False, True))
def test_forward_boundary_producers_are_preferred_without_removing_edges(*, reverse: bool) -> None:
    incidence = _nested(1, reverse=reverse).structural_analysis()._incidence  # noqa: SLF001
    available = {"inlet", "heat", "flow", "capacity"}
    graph = matching.build_matching_graph(incidence, available)
    original = list(graph.edge_list())
    expansion = StructuralExpansion(incidence, available, "outlet")
    expansion.collect()
    matching.prefer_structural_producers(graph, expansion.snapshot())
    assignments = matching.find_matching(graph)
    assert assignments["left.outlet"].__bilt_name__ == "left"
    assert assignments["right.inlet"] is connection
    assert assignments["right.outlet"].__bilt_name__ == "right"
    assert assignments["outlet"] is discharge
    assert list(graph.edge_list()) == original


@pytest.mark.parametrize("depth", (1, 2, 3))
@pytest.mark.parametrize("output", ("inlet", "outlet"))
def test_nested_series_uses_first_candidates_in_each_query(
    monkeypatch: pytest.MonkeyPatch, depth: int, output: str
) -> None:
    monkeypatch.setattr(implementation_module, "ExhaustiveStructuralAnalyser", solver.CausalizationStructuralAnalyzer)
    original = solver._candidate_assignments  # noqa: SLF001

    def first_only(
        graph: rx.PyGraph,
        failures: collections.abc.Mapping[PlanStep, RealizationFailure],
        *,
        output: str | None,
    ) -> collections.abc.Iterator[dict[str, BiltModelConstituent]]:
        candidates = original(graph, failures, output=output)
        first = next(candidates, None)
        if first is not None:
            yield first
        msg = "Reachable series requests should not enumerate alternative matchings."
        raise AssertionError(msg)

    monkeypatch.setattr(solver, "_candidate_assignments", first_only)
    implementation = _nested(depth)
    values = {"heat": 12_000.0, "flow": 2.0, "capacity": 4_000.0}
    values["inlet" if output == "outlet" else "outlet"] = 10.0 if output == "outlet" else 11.5
    request = EvaluationRequest(output=output, input_domains=dict.fromkeys(values, Domain.scalar()))
    with EvaluationContext():
        case = implementation.analyser_for(PYTHON_BACKEND).analyse(request)
        expected = 11.5 if output == "outlet" else 10.0
        assert PYTHON_BACKEND.chain_case(case).callable(**values) == expected


def test_unresolved_expansion_does_not_supply_hints() -> None:
    incidence = _nested(1).structural_analysis()._incidence  # noqa: SLF001
    graph = matching.build_matching_graph(incidence, {"inlet"})
    original = list(graph.weighted_edge_list())
    expansion = StructuralExpansion(incidence, {"inlet"}, "outlet")
    assert not expansion.snapshot().complete
    matching.prefer_structural_producers(graph, expansion.snapshot())
    assert list(graph.weighted_edge_list()) == original


@pytest.mark.parametrize("depth", (1, 2))
def test_preferences_follow_the_current_request_in_both_directions(monkeypatch: pytest.MonkeyPatch, depth: int) -> None:
    monkeypatch.setattr(implementation_module, "ExhaustiveStructuralAnalyser", solver.CausalizationStructuralAnalyzer)
    analyser = _nested(depth).analyser_for(PYTHON_BACKEND)
    shared = {"heat": 12_000.0, "flow": 2.0, "capacity": 4_000.0}
    requests = (("outlet", {"inlet": 10.0}, 11.5), ("inlet", {"outlet": 11.5}, 10.0))
    with EvaluationContext():
        for output, temperature, expected in requests:
            values = shared | temperature
            request = EvaluationRequest(output=output, input_domains=dict.fromkeys(values, Domain.scalar()))
            case = analyser.analyse(request)
            assert PYTHON_BACKEND.chain_case(case).callable(**values) == expected


def test_transform_inputs_are_not_propagated_backwards() -> None:
    @transform
    def supplied(unknown: float) -> float:
        return unknown + 1

    @equation
    def downstream(unknown: float, result: float) -> bool:
        return result == unknown + 1

    incidence = FullIncidenceGraph([supplied, downstream])
    graph = matching.build_matching_graph(incidence, {"supplied"})
    original = list(graph.weighted_edge_list())
    expansion = StructuralExpansion(incidence, {"supplied"}, "result")
    expansion.collect()
    snapshot = expansion.snapshot()
    snapshot = dataclasses.replace(
        snapshot,
        constraints=(ExpandedConstraint((), supplied, frozenset({"supplied", "unknown"})), *snapshot.constraints),
    )
    matching.prefer_structural_producers(graph, snapshot)
    assert list(graph.weighted_edge_list()) == original


def test_hint_discovery_failure_preserves_existing_route_query(monkeypatch: pytest.MonkeyPatch) -> None:
    def refused_hint(*_: object, **__: object) -> typing.Never:
        msg = "Cannot discover a preference."
        raise BiltAnalysisError(msg)

    monkeypatch.setattr(implementation_module, "ExhaustiveStructuralAnalyser", solver.CausalizationStructuralAnalyzer)
    implementation = _nested(1)
    monkeypatch.setattr(solver, "StructuralExpansion", refused_hint)
    analyser = implementation.analyser_for(PYTHON_BACKEND)
    available = {"left.outlet", "heat", "flow", "capacity"}
    assert analyser.required_inputs_for_target(available, "outlet") == frozenset(available)


def test_self_dependent_transform_is_not_a_forward_source() -> None:
    @transform
    def unknown(unknown: float) -> float:
        return unknown + 1

    @equation
    def downstream(unknown: float, result: float) -> bool:
        return result == unknown + 1

    incidence = FullIncidenceGraph([unknown, downstream])
    graph = matching.build_matching_graph(incidence, set())
    original = list(graph.weighted_edge_list())
    expansion = StructuralExpansion(incidence, set(), "result")
    expansion.collect()
    matching.prefer_structural_producers(graph, expansion.snapshot())
    assert list(graph.weighted_edge_list()) == original


def test_single_child_query_does_not_expand_an_ordering_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_expansion(*_: object, **__: object) -> typing.Never:
        msg = "A single child must keep the existing lightweight ordering."
        raise AssertionError(msg)

    monkeypatch.setattr(implementation_module, "ExhaustiveStructuralAnalyser", solver.CausalizationStructuralAnalyzer)
    child = Implementation(Ports, [exchange])
    parent = Implementation(Ports, [supply], internal_implementations={"left": child})
    monkeypatch.setattr(solver, "StructuralExpansion", unexpected_expansion)
    available = {"left.inlet", "left.heat", "left.flow", "left.capacity"}
    assert parent.analyser_for(PYTHON_BACKEND).required_inputs_for_target(available, "left.outlet") == frozenset(
        available
    )


def test_failed_preferred_direction_still_leaves_alternative_matching() -> None:
    from bilt.core._structural_analysis._plan import PlanStep  # noqa: PLC0415
    from bilt.core._structural_analysis._realize import RealizationFailure  # noqa: PLC0415

    incidence = _nested(1).structural_analysis()._incidence  # noqa: SLF001
    available = {"inlet", "heat", "flow", "capacity"}
    graph = matching.build_matching_graph(incidence, available)
    expansion = StructuralExpansion(incidence, available, "outlet")
    expansion.collect()
    matching.prefer_structural_producers(graph, expansion.snapshot())
    failures: dict[PlanStep, RealizationFailure] = {}
    candidates = solver._candidate_assignments(graph, failures, output="outlet")  # noqa: SLF001
    assert next(candidates)["right.inlet"] is connection
    failed = PlanStep(operation=connection, output="right.inlet")
    failures[failed] = RealizationFailure(failed_plan_step=failed)
    assert next(candidates).get("right.inlet") is not connection
