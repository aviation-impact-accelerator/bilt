# ruff: noqa: PLR2004

"""Experimental alias matching must preserve native plans and values."""

import pytest

from bilt import Implementation, Interface, equation, field, system
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis._causalization_block_solver._alias_matching import (
    alias_matching_scope,
    build_alias_matching,
)
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph
from bilt.core._structural_analysis._causalization_block_solver._matching import build_matching_graph
from bilt.symbolic._domains import Domain


@equation
def origin(input_value: float, a: float) -> bool:
    return a == 2 * input_value


@equation
def connect(a: float, b: float) -> bool:
    return a == b


@equation
def sink(b: float, answer: float) -> bool:
    return answer == 3 * b


@equation
def offset(a: float, b: float) -> bool:
    return a == b + 1


def test_alias_matching_removes_participants_before_search() -> None:
    incidence = FullIncidenceGraph([origin, connect, sink])
    reduced = build_alias_matching(incidence, {"input_value"})
    assert reduced is not None
    assert reduced.canonical("a") == reduced.canonical("b")
    assert reduced.removed_nodes == 2
    assert reduced.removed_edges > 0


@pytest.mark.parametrize(("output", "expected"), (("a", 8.0), ("b", 8.0), ("answer", 24.0)))
def test_alias_matching_preserves_native_outputs(output: str, expected: float) -> None:
    analyser = CausalizationStructuralAnalyzer([origin, connect, sink], PYTHON_BACKEND)
    request = EvaluationRequest(output=output, input_domains={"input_value": Domain.scalar()})
    with EvaluationContext():
        baseline = analyser.analyse(request)
        with alias_matching_scope():
            aliased = analyser.analyse(request)
        baseline_value = PYTHON_BACKEND.chain_case(baseline).callable(input_value=4.0)
        alias_value = PYTHON_BACKEND.chain_case(aliased).callable(input_value=4.0)
        assert alias_value == baseline_value == expected


def test_supplied_input_connection_is_not_contracted() -> None:
    incidence = FullIncidenceGraph([origin, connect, sink])
    assert build_alias_matching(incidence, {"a", "input_value"}) is None


def test_arithmetic_connection_is_not_contracted() -> None:
    incidence = FullIncidenceGraph([origin, offset, sink])
    assert build_alias_matching(incidence, {"input_value"}) is None


def test_alias_projection_preserves_existing_producer_preferences() -> None:
    incidence = FullIncidenceGraph([origin, connect, sink])
    original = build_matching_graph(incidence, {"input_value"})
    for first, second in original.edge_list():
        if original[first] is origin or original[second] is origin:
            original.update_edge(first, second, 7)
    reduced = build_alias_matching(incidence, {"input_value"}, matching_graph=original)
    assert reduced is not None
    assert 7 in {weight for _, _, weight in reduced.graph.weighted_edge_list()}


@system
class NestedPorts(metaclass=Interface):
    inlet: float = field()
    outlet: float = field()


@equation
def double(inlet: float, outlet: float) -> bool:
    return outlet == 2 * inlet


@equation
def supply(inlet: float, left: NestedPorts) -> bool:
    return left.inlet == inlet


@equation
def bridge(left: NestedPorts, right: NestedPorts) -> bool:
    return right.inlet == left.outlet


@equation
def discharge(outlet: float, right: NestedPorts) -> bool:
    return outlet == right.outlet


@pytest.mark.timeout(10)
def test_alias_matching_preserves_nested_child_routes() -> None:
    child = Implementation(NestedPorts, [double])
    parent = Implementation(
        NestedPorts,
        [supply, bridge, discharge],
        internal_implementations={"left": child, "right": child},
    )
    analyser = parent.analyser_for(PYTHON_BACKEND)
    request = EvaluationRequest(output="outlet", input_domains={"inlet": Domain.scalar()})
    with EvaluationContext():
        baseline = analyser.analyse(request)
        with alias_matching_scope():
            experimental = analyser.analyse(request)
        value = {"inlet": 3.0}
        assert PYTHON_BACKEND.chain_case(baseline).callable(**value) == 12.0
        assert PYTHON_BACKEND.chain_case(experimental).callable(**value) == 12.0
