import typing

import pytest

from bilt import Implementation, Interface, equation, field, system, transform
from bilt._backend._errors import BiltUnsupportedNodeError
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis._causalization_block_solver import _backend as solver
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.core._structural_analysis._causalization_block_solver._errors import BiltExecutionPlanError
from bilt.core._structural_analysis._causalization_block_solver._route_cache import route_cache_scope
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._system import _implementation as implementation_module
from bilt.symbolic._domains import Domain

if typing.TYPE_CHECKING:
    import collections.abc

    import rustworkx as rx

    from bilt._backend import Backend, BackendCase
    from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph
    from bilt.core._structural_analysis._plan import AnalysisPlan
    from bilt.core._structural_analysis._realize import RealizationFailure


@transform
def b(a: float) -> float:
    return 2 * a


@pytest.fixture
def graph_builds(monkeypatch: pytest.MonkeyPatch) -> list[FullIncidenceGraph]:
    original = solver.build_matching_graph
    builds = []

    def build(incidence: FullIncidenceGraph, available: collections.abc.Set[str]) -> rx.PyGraph:
        builds.append(incidence)
        return original(incidence, available)

    monkeypatch.setattr(solver, "build_matching_graph", build)
    return builds


@pytest.mark.parametrize("available", ({"a"}, set()))
def test_reuses_successful_and_unsuccessful_queries(
    available: set[str], graph_builds: list[FullIncidenceGraph]
) -> None:
    analyser = CausalizationStructuralAnalyzer([b], PYTHON_BACKEND)
    expected = frozenset({"a"}) if available else None
    with route_cache_scope():
        assert analyser.required_inputs_for_target(available, "b") == expected
        assert analyser.required_inputs_for_target(available, "b") == expected
    assert len(graph_builds) == 1


def test_keys_distinguish_analyser_and_supplied_inputs(graph_builds: list[FullIncidenceGraph]) -> None:
    dependent = CausalizationStructuralAnalyzer([b], PYTHON_BACKEND)

    @equation
    def constant(b: float) -> bool:
        return b == 1.0

    constant_analyser = CausalizationStructuralAnalyzer([constant], PYTHON_BACKEND)
    available = {"a"}
    with route_cache_scope():
        assert dependent.required_inputs_for_target(available, "b") == frozenset({"a"})
        available.clear()
        assert dependent.required_inputs_for_target(available, "b") is None
        assert constant_analyser.required_inputs_for_target(available, "b") == frozenset()
        assert constant_analyser.required_inputs_for_target(available, "missing") is None
        available.add("a")
        assert dependent.required_inputs_for_target(available, "b") == frozenset({"a"})
    expected_builds = 4
    assert len(graph_builds) == expected_builds


def test_standalone_queries_do_not_retain_cache_between_calls(graph_builds: list[FullIncidenceGraph]) -> None:
    analyser = CausalizationStructuralAnalyzer([b], PYTHON_BACKEND)
    with EvaluationContext():
        for _ in range(2):
            assert analyser.required_inputs_for_target({"a"}, "b") == frozenset({"a"})
    expected_builds = 2
    assert len(graph_builds) == expected_builds


def test_scope_is_discarded_after_exception(graph_builds: list[FullIncidenceGraph]) -> None:
    analyser = CausalizationStructuralAnalyzer([b], PYTHON_BACKEND)

    def abort() -> typing.Never:
        with route_cache_scope():
            analyser.required_inputs_for_target({"a"}, "b")
            message = "abort analysis"
            raise RuntimeError(message)

    with pytest.raises(RuntimeError, match="abort"):
        abort()
    with route_cache_scope():
        assert analyser.required_inputs_for_target({"a"}, "b") == frozenset({"a"})
    expected_builds = 2
    assert len(graph_builds) == expected_builds


def test_analysis_shares_queries_but_discards_them_before_next_request(
    monkeypatch: pytest.MonkeyPatch, graph_builds: list[FullIncidenceGraph]
) -> None:
    child = CausalizationStructuralAnalyzer([b], PYTHON_BACKEND)
    parent = CausalizationStructuralAnalyzer([b], PYTHON_BACKEND)
    original = solver.realize_plan

    def realise(plan: AnalysisPlan, backend: Backend) -> BackendCase | RealizationFailure:
        assert child.required_inputs_for_target({"a"}, "b") == frozenset({"a"})
        assert child.required_inputs_for_target({"a"}, "b") == frozenset({"a"})
        return original(plan, backend)

    monkeypatch.setattr(solver, "realize_plan", realise)
    request = EvaluationRequest(output="b", input_domains={"a": Domain.scalar()})
    with EvaluationContext():
        for _ in range(2):
            case = parent.analyse(request)
            expected = 6.0
            assert PYTHON_BACKEND.chain_case(case).callable(a=3.0) == expected
    expected_queries = 2
    assert graph_builds.count(child.full_incidence_graph) == expected_queries


def _refuse(*_: object, **__: object) -> typing.Never:
    message = "Compilation refused for cache invalidation test."
    raise BiltUnsupportedNodeError(message)


def test_realisation_failure_invalidates_positive_and_negative_answers(
    monkeypatch: pytest.MonkeyPatch, graph_builds: list[FullIncidenceGraph]
) -> None:
    analyser = CausalizationStructuralAnalyzer([b], PYTHON_BACKEND)
    request = EvaluationRequest(output="b", input_domains={"a": Domain.scalar()})
    monkeypatch.setattr(PYTHON_BACKEND, "transpile", _refuse)
    with route_cache_scope(), EvaluationContext():
        assert analyser.required_inputs_for_target({"a"}, "b") == frozenset({"a"})
        assert analyser.required_inputs_for_target(set(), "b") is None
        with pytest.raises(BiltExecutionPlanError, match="Compilation refused"):
            analyser.analyse(request)
        before = len(graph_builds)
        for _ in range(2):
            assert analyser.required_inputs_for_target({"a"}, "b") is None
            assert analyser.required_inputs_for_target(set(), "b") is None
        expected_builds = 2
        assert len(graph_builds) - before == expected_builds


def test_child_failure_invalidates_cached_parent_route(monkeypatch: pytest.MonkeyPatch) -> None:
    @system
    class Ports(metaclass=Interface):
        a: float = field()
        b: float = field()

    monkeypatch.setattr(implementation_module, "ExhaustiveStructuralAnalyser", CausalizationStructuralAnalyzer)
    implementation = Implementation(Ports, [b])
    child = SubSystemConstituent("child", implementation)
    parent = CausalizationStructuralAnalyzer([child], PYTHON_BACKEND)
    request = EvaluationRequest(output="b", input_domains={"a": Domain.scalar()})
    monkeypatch.setattr(PYTHON_BACKEND, "transpile", _refuse)
    with route_cache_scope(), EvaluationContext():
        assert parent.required_inputs_for_target({"child.a"}, "child.b") == frozenset({"child.a"})
        with pytest.raises(BiltExecutionPlanError):
            implementation.analyser_for(PYTHON_BACKEND).analyse(request)
        assert parent.required_inputs_for_target({"child.a"}, "child.b") is None
