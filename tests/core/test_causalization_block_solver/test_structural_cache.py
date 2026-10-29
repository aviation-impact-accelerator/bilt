import typing

import pytest

from bilt import Implementation, Interface, equation, field, system, transform
from bilt._backend._errors import BiltUnsupportedNodeError
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis._causalization_block_solver import _structural_scope as scope_module
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.core._structural_analysis._causalization_block_solver._errors import (
    BiltExecutionPlanError,
    BiltUnderconstrainedError,
)
from bilt.core._structural_analysis._causalization_block_solver._expand_structure import StructuralExpansion
from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph
from bilt.core._structural_analysis._causalization_block_solver._route_cache import route_cache_scope
from bilt.core._structural_analysis._causalization_block_solver._structural_cache import (
    get_structural_cache,
    structural_cache_scope,
)
from bilt.core._structural_analysis._causalization_block_solver._structural_scope import inspect_structure
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._system import _implementation as implementation_module
from bilt.symbolic._domains import Domain

type Query = tuple[FullIncidenceGraph, frozenset[str], frozenset[str]]


@system
class Ports(metaclass=Interface):
    a: float = field()
    b: float = field()


@equation
def relation(a: float, b: float) -> bool:
    return b == 2 * a


@transform
def b(a: float) -> float:
    return 2 * a


@pytest.fixture
def walks(monkeypatch: pytest.MonkeyPatch) -> list[Query]:
    original = scope_module._scope_nodes  # noqa: SLF001
    queries: list[Query] = []

    def walk(
        incidence: FullIncidenceGraph, supplied: frozenset[str], targets: frozenset[str]
    ) -> tuple[set[int], dict[int, set[str]], set[str]]:
        queries.append((incidence, supplied, targets))
        return original(incidence, supplied, targets)

    monkeypatch.setattr(scope_module, "_scope_nodes", walk)
    return queries


@pytest.mark.parametrize("target", ("b", "missing", "a"))
def test_reuses_populated_empty_and_supplied_target_scopes(target: str, walks: list[Query]) -> None:
    incidence = FullIncidenceGraph([relation])
    with structural_cache_scope():
        first = inspect_structure(incidence, {"a"}, {target})
        assert inspect_structure(incidence, {"a"}, {target}) is first
        assert first.required == frozenset({target})
        assert first.available == frozenset({"a"})
    assert len(walks) == 1
    assert get_structural_cache() is None


def test_keys_preserve_owner_inputs_targets_and_mutable_argument_changes(walks: list[Query]) -> None:
    incidence = FullIncidenceGraph([relation])
    other = FullIncidenceGraph([relation])
    available = {"a"}
    required = {"b"}
    with structural_cache_scope():
        first = inspect_structure(incidence, available, required)
        available.add("outside")
        extra = inspect_structure(incidence, available, required)
        assert extra.available == frozenset({"a", "outside"})
        assert extra is not first
        available.clear()
        inspect_structure(incidence, available, required)
        required.add("missing")
        inspect_structure(incidence, available, required)
        inspect_structure(other, {"a"}, {"b"})
        assert inspect_structure(incidence, {"a"}, {"b"}) is first
    expected_walks = 5
    assert len(walks) == expected_walks


def test_inspections_outside_a_request_do_not_retain_results(walks: list[Query]) -> None:
    incidence = FullIncidenceGraph([relation])
    first = inspect_structure(incidence, {"a"}, {"b"})
    assert inspect_structure(incidence, {"a"}, {"b"}) is not first
    expected_walks = 2
    assert len(walks) == expected_walks
    assert get_structural_cache() is None


def test_nested_scope_and_exception_restore_the_outer_cache(walks: list[Query]) -> None:
    incidence = FullIncidenceGraph([relation])
    with structural_cache_scope() as outer:
        first = inspect_structure(incidence, {"a"}, {"b"})

        def abort() -> typing.Never:
            with structural_cache_scope() as inner:
                assert inner is outer
                assert inspect_structure(incidence, {"a"}, {"b"}) is first
                message = "abort nested analysis"
                raise RuntimeError(message)

        with pytest.raises(RuntimeError, match="abort"):
            abort()
        assert get_structural_cache() is outer
        assert inspect_structure(incidence, {"a"}, {"b"}) is first
    assert get_structural_cache() is None
    with structural_cache_scope():
        assert inspect_structure(incidence, {"a"}, {"b"}) is not first
    expected_walks = 2
    assert len(walks) == expected_walks


def test_inspection_exception_is_not_cached(monkeypatch: pytest.MonkeyPatch, walks: list[Query]) -> None:
    incidence = FullIncidenceGraph([relation])
    original = scope_module._scope_nodes  # noqa: SLF001

    def refuse(*_: object) -> typing.Never:
        message = "inspection failed"
        raise RuntimeError(message)

    with structural_cache_scope():
        monkeypatch.setattr(scope_module, "_scope_nodes", refuse)
        with pytest.raises(RuntimeError, match="inspection failed"):
            inspect_structure(incidence, {"a"}, {"b"})
        monkeypatch.setattr(scope_module, "_scope_nodes", original)
        first = inspect_structure(incidence, {"a"}, {"b"})
        assert inspect_structure(incidence, {"a"}, {"b"}) is first
    assert len(walks) == 1


def _shared_parent() -> tuple[CausalizationStructuralAnalyzer, Implementation]:
    implementation = Implementation(Ports, [relation])
    left = SubSystemConstituent("left", implementation)
    right = SubSystemConstituent("right", implementation)

    @equation
    def combined(answer: float, left: Ports, right: Ports) -> bool:
        return answer == left.b + right.b

    return CausalizationStructuralAnalyzer([left, right, combined], PYTHON_BACKEND), implementation


def test_shared_child_query_reuse_preserves_distinct_instance_rows(walks: list[Query]) -> None:
    parent, child = _shared_parent()
    with structural_cache_scope():
        expansion = StructuralExpansion(parent.full_incidence_graph, {"left.a", "right.a"}, "answer")
        expansion.collect()
    rows = [row for row in expansion.snapshot().constraints if row.operation is relation]
    assert {row.path for row in rows} == {("left",), ("right",)}
    assert {row.attributes for row in rows} == {frozenset({"left.a", "left.b"}), frozenset({"right.a", "right.b"})}
    child_incidence = child.structural_analysis()._incidence  # noqa: SLF001
    assert sum(owner is child_incidence for owner, _, _ in walks) == 1


@pytest.mark.parametrize("query_only", (False, True))
def test_nested_requests_share_queries_and_next_request_starts_fresh(
    monkeypatch: pytest.MonkeyPatch, walks: list[Query], *, query_only: bool
) -> None:
    monkeypatch.setattr(implementation_module, "CausalizationStructuralAnalyzer", CausalizationStructuralAnalyzer)
    parent, child = _shared_parent()
    child_incidence = child.structural_analysis()._incidence  # noqa: SLF001
    request = EvaluationRequest(output="answer", input_domains=dict.fromkeys(("left.a", "right.a"), Domain.scalar()))
    previous: set[Query] | None = None
    with EvaluationContext():
        for _ in range(2):
            walks.clear()
            if query_only:
                assert parent.required_inputs_for_target(set(request.input_domains), "answer") == frozenset(
                    {"left.a", "right.a"}
                )
            else:
                case = parent.analyse(request)
                expected = 14.0
                assert PYTHON_BACKEND.chain_case(case).callable(left__a=3.0, right__a=4.0) == expected
            assert get_structural_cache() is None
            assert len(walks) == len(set(walks))
            # Compiled child evaluations may survive within EvaluationContext;
            # parent expansion must still inspect the child's structure afresh.
            current = {query for query in walks if query[0] is child_incidence}
            assert current
            if previous is not None:
                assert current == previous
            previous = current
    assert sum(owner is child_incidence for owner, _, _ in walks) == 1


def test_realisation_failure_invalidates_routes_but_preserves_structural_evidence(
    monkeypatch: pytest.MonkeyPatch, walks: list[Query]
) -> None:
    analyser = CausalizationStructuralAnalyzer([b], PYTHON_BACKEND)
    request = EvaluationRequest(output="b", input_domains={"a": Domain.scalar()})

    def refuse(*_: object, **__: object) -> typing.Never:
        message = "Compilation refused"
        raise BiltUnsupportedNodeError(message)

    monkeypatch.setattr(PYTHON_BACKEND, "transpile", refuse)
    with structural_cache_scope(), route_cache_scope(), EvaluationContext():
        first = inspect_structure(analyser.full_incidence_graph, {"a"}, {"b"})
        assert analyser.required_inputs_for_target({"a"}, "b") == frozenset({"a"})
        with pytest.raises(BiltExecutionPlanError, match="Compilation refused"):
            analyser.analyse(request)
        assert inspect_structure(analyser.full_incidence_graph, {"a"}, {"b"}) is first
        assert analyser.required_inputs_for_target({"a"}, "b") is None
        assert len(walks) == 1


def test_failed_analysis_discards_its_structural_cache(walks: list[Query]) -> None:
    analyser = CausalizationStructuralAnalyzer([b], PYTHON_BACKEND)
    request = EvaluationRequest(output="b", input_domains={})
    with pytest.raises(BiltUnderconstrainedError):
        analyser.analyse(request)
    assert get_structural_cache() is None
    assert walks
    walks.clear()
    # A later, successful request starts without evidence retained by failure.
    with EvaluationContext():
        case = analyser.analyse(EvaluationRequest(output="b", input_domains={"a": Domain.scalar()}))
        expected = 6.0
        assert PYTHON_BACKEND.chain_case(case).callable(a=3.0) == expected
    assert len(walks) == 1
    assert get_structural_cache() is None
