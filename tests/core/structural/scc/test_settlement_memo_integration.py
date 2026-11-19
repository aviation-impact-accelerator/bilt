"""Exercise settlement reuse across child systems, cyclic blocks and failure retries."""

# ruff: noqa: SLF001 - verify private settlement lifecycle and cache behavior

import typing

lazy import pytest

from bilt import Implementation, Interface, equation, field, system, transform
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt._backend._errors import BiltUnsupportedNodeError
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis._causalization_block_solver import _planning, _settle_assignments
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.core._structural_analysis._causalization_block_solver._settlement_cache import (
    _cache,
    settlement_memo_scope,
)
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._system import _implementation as implementation_module
from bilt.symbolic._domains import Domain


@equation
def total(x: float, y: float, p: float) -> bool:
    return x + y == p


@equation
def difference(x: float, y: float, q: float) -> bool:
    return x - y == q


@system
class Child(metaclass=Interface):
    a: float = field()
    b: float = field()
    c: float = field()


@transform
def b(a: float) -> float:
    return a * 2


@equation
def c(b: float, c: float) -> bool:
    return c == b + 1


@equation
def combined(answer: float, child: Child) -> bool:
    return answer == child.b + child.c


def test_coupled_linear_block_settlement_reused_without_changing_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _settle_assignments._TargetSettlement.settle
    calls: list[str] = []

    def counted(self: _settle_assignments._TargetSettlement, output: str):
        calls.append(output)
        return original(self, output)

    monkeypatch.setattr(_settle_assignments._TargetSettlement, "settle", counted)
    assigned = {"x": total, "y": difference}
    with settlement_memo_scope():
        first = _planning.plan_blocks(assigned, {"p", "q"}, "x", backend=PYTHON_BACKEND)
        second = _planning.plan_blocks(dict(reversed(tuple(assigned.items()))), {"p", "q"}, "x", backend=PYTHON_BACKEND)

    assert first == second
    assert any(block.is_cyclic for block in first)
    assert calls == ["x"]


def test_nested_child_sibling_inputs_preserved_on_cache_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(implementation_module, "CausalizationStructuralAnalyzer", CausalizationStructuralAnalyzer)
    child = SubSystemConstituent("child", Implementation(Child, [b, c]))
    assignments = {"answer": combined, "child.b": child, "child.c": child}
    with settlement_memo_scope():
        first = _settle_assignments.settle_assignments(
            assignments, {"child.a"}, PYTHON_BACKEND, target="answer"
        )
        second = _settle_assignments.settle_assignments(
            dict(reversed(tuple(assignments.items()))), {"child.a"}, PYTHON_BACKEND, target="answer"
        )

    def routes(result):
        return {step.output: step.child_inputs for step in result[0]}

    assert first[1] == second[1]
    assert routes(first) == routes(second)
    assert routes(first)["child.b"] == frozenset({"a"})
    assert routes(first)["child.c"] == frozenset({"b"})


def test_different_backend_identities_never_share_settlements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _settle_assignments._TargetSettlement.settle
    called: list[str] = []

    def counted(self: _settle_assignments._TargetSettlement, output: str):
        called.append(output)
        return original(self, output)

    monkeypatch.setattr(_settle_assignments._TargetSettlement, "settle", counted)
    assignments = {"x": total, "y": difference}
    with settlement_memo_scope():
        for backend in (PYTHON_BACKEND, JAX_BACKEND, PYTHON_BACKEND, JAX_BACKEND):
            _settle_assignments.settle_assignments(assignments, {"p", "q"}, backend, target="x")
    assert called == ["x", "x"]


def test_failed_coupled_block_invalidates_memo_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = PYTHON_BACKEND.transpile_linear_solve
    attempts = []

    def compile_once_rejected(function):
        attempts.append(function)
        if len(attempts) == 1:
            raise BiltUnsupportedNodeError("first block route refused")
        return original(function)

    monkeypatch.setattr(PYTHON_BACKEND, "transpile_linear_solve", compile_once_rejected)
    analyser = CausalizationStructuralAnalyzer([total, difference], PYTHON_BACKEND)
    request = EvaluationRequest(output="x", input_domains={"p": Domain.scalar(), "q": Domain.scalar()})
    with settlement_memo_scope(), EvaluationContext():
        case = analyser.analyse(request)
    assert PYTHON_BACKEND.chain_case(case).callable(p=10.0, q=2.0) == pytest.approx(6.0)
    assert len(attempts) == 2
    assert not analyser._failures


def test_cache_is_bounded_to_256_structural_entries() -> None:
    with settlement_memo_scope():
        for index in range(260):
            _settle_assignments.settle_assignments({"x": total}, {"p"}, PYTHON_BACKEND, target=f"x_{index}")
        cache = _cache.get()
        assert cache is not None
        assert len(cache) == 256
    assert _cache.get() is None


def test_nested_opt_in_scope_preserves_owning_request_cache() -> None:
    with settlement_memo_scope():
        outer = _cache.get()
        assert outer is not None
        with settlement_memo_scope():
            assert _cache.get() is outer
        assert _cache.get() is outer
    assert _cache.get() is None
