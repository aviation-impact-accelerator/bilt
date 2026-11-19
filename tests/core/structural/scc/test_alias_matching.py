# ruff: noqa: PLR2004

"""Experimental alias matching must preserve native plans and values."""

import pytest

from bilt import equation
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis._causalization_block_solver._alias_matching import (
    alias_matching_scope,
    build_alias_matching,
)
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph
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
