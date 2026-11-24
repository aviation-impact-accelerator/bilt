"""Ordinary property requests can reuse a compatible previous structural matching."""

import typing

import pytest

from bilt import transform
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis._causalization_block_solver import _backend as solver
from bilt.core._structural_analysis._causalization_block_solver import _native_prototype as native
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.symbolic._domains import Domain

if typing.TYPE_CHECKING:
    import collections.abc

    import rustworkx as rx

    from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph


@pytest.fixture(autouse=True)
def _reference_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the Python-specific graph and cache implementation."""
    monkeypatch.setattr(native, "MODE", "python")


@transform
def base(a: float) -> float:
    return a + 1


@transform
def left(base: float) -> float:
    return base * 2


@transform
def right(base: float) -> float:
    return base * 3


def _evaluate(analyser: CausalizationStructuralAnalyzer, output: str, inputs: dict[str, float]) -> float:
    request = EvaluationRequest(output=output, input_domains=dict.fromkeys(inputs, Domain.scalar()))
    with EvaluationContext():
        plan = analyser.analyse(request)
        compiled = PYTHON_BACKEND.chain_case(plan)
        used = {name: value for name, value in inputs.items() if name in compiled.parameters}
        return typing.cast("float", compiled.callable(**used))


def test_second_output_reuses_prior_matching_without_graph_build(monkeypatch: pytest.MonkeyPatch) -> None:
    analyser = CausalizationStructuralAnalyzer([base, left, right], PYTHON_BACKEND)
    original = solver.build_matching_graph
    builds: list[object] = []

    def counted(incidence: FullIncidenceGraph, available: collections.abc.Set[str]) -> rx.PyGraph:
        builds.append((incidence, frozenset(available)))
        return original(incidence, available)

    monkeypatch.setattr(solver, "build_matching_graph", counted)

    assert _evaluate(analyser, "left", {"a": 3.0}) == pytest.approx(8.0)
    assert _evaluate(analyser, "right", {"a": 3.0}) == pytest.approx(12.0)
    assert len(builds) == 1


def test_seed_failure_falls_back_to_current_target_search(monkeypatch: pytest.MonkeyPatch) -> None:
    analyser = CausalizationStructuralAnalyzer([base, left, right], PYTHON_BACKEND)
    original = solver.build_matching_graph
    builds: list[object] = []

    def counted(incidence: FullIncidenceGraph, available: collections.abc.Set[str]) -> rx.PyGraph:
        builds.append((incidence, frozenset(available)))
        return original(incidence, available)

    monkeypatch.setattr(solver, "build_matching_graph", counted)
    assert _evaluate(analyser, "left", {"a": 3.0}) == pytest.approx(8.0)

    # Simulate a successful previous route lacking a producer for the next output.
    # Normal matching search must recover rather than trusting that seed.
    signature = frozenset({("a", Domain.scalar())})
    analyser._last_matching = (signature, {"base": base, "left": left})  # noqa: SLF001
    assert _evaluate(analyser, "right", {"a": 3.0}) == pytest.approx(12.0)
    expected_builds = 2
    assert len(builds) == expected_builds


def test_changed_supplied_boundary_does_not_reuse_matching(monkeypatch: pytest.MonkeyPatch) -> None:
    analyser = CausalizationStructuralAnalyzer([base, left, right], PYTHON_BACKEND)
    original = solver.build_matching_graph
    builds: list[object] = []

    def counted(incidence: FullIncidenceGraph, available: collections.abc.Set[str]) -> rx.PyGraph:
        builds.append((incidence, frozenset(available)))
        return original(incidence, available)

    monkeypatch.setattr(solver, "build_matching_graph", counted)

    assert _evaluate(analyser, "left", {"a": 3.0}) == pytest.approx(8.0)
    assert _evaluate(analyser, "right", {"a": 3.0, "unused": 9.0}) == pytest.approx(12.0)
    expected_builds = 2
    assert len(builds) == expected_builds
