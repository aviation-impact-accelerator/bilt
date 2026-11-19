"""Exact settlement memoization never changes default structural planning."""

import pytest

from bilt import equation, transform
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._structural_analysis._causalization_block_solver import _settle_assignments as settlement
from bilt.core._structural_analysis._causalization_block_solver._route_cache import invalidate_route_cache
from bilt.core._structural_analysis._causalization_block_solver._settlement_cache import (
    settlement_cache_scope,
    settlement_memo_scope,
)


@equation
def relation(a: float, b: float) -> bool:
    return b == a + 1


@transform
def result(b: float) -> float:
    return b * 2


def test_exact_settlement_reuses_original_result(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    original = settlement._TargetSettlement.settle

    def counted(self: settlement._TargetSettlement, output: str):
        calls.append(output)
        return original(self, output)

    monkeypatch.setattr(settlement._TargetSettlement, "settle", counted)
    assignments = {"b": relation, "result": result}
    with settlement_memo_scope(), settlement_cache_scope():
        first = settlement.settle_assignments(assignments, {"a"}, PYTHON_BACKEND, target="result")
        second = settlement.settle_assignments(dict(reversed(tuple(assignments.items()))), {"a"}, PYTHON_BACKEND, target="result")
        assert first == second
        assert len(calls) == 1
        settlement.settle_assignments(assignments, set(), PYTHON_BACKEND, target="result")
        settlement.settle_assignments(assignments, {"a"}, PYTHON_BACKEND, target="b")
        assert len(calls) == 3


def test_failure_invalidation_clears_settlement_memo(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    original = settlement._TargetSettlement.settle

    def counted(self: settlement._TargetSettlement, output: str):
        calls.append(output)
        return original(self, output)

    monkeypatch.setattr(settlement._TargetSettlement, "settle", counted)
    assignments = {"b": relation, "result": result}
    with settlement_memo_scope(), settlement_cache_scope():
        first = settlement.settle_assignments(assignments, {"a"}, PYTHON_BACKEND, target="result")
        assert settlement.settle_assignments(assignments, {"a"}, PYTHON_BACKEND, target="result") == first
        assert len(calls) == 1
        invalidate_route_cache()
        assert settlement.settle_assignments(assignments, {"a"}, PYTHON_BACKEND, target="result") == first
        assert len(calls) == 2


def test_different_assignment_identity_is_not_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    original = settlement._TargetSettlement.settle

    def counted(self: settlement._TargetSettlement, output: str):
        calls.append(output)
        return original(self, output)

    monkeypatch.setattr(settlement._TargetSettlement, "settle", counted)

    @equation
    def alternative(a: float, b: float) -> bool:
        return b == 3 * a

    with settlement_memo_scope(), settlement_cache_scope():
        one = settlement.settle_assignments({"b": relation}, {"a"}, PYTHON_BACKEND, target="b")
        two = settlement.settle_assignments({"b": alternative}, {"a"}, PYTHON_BACKEND, target="b")
        assert one != two
        assert len(calls) == 2


def test_no_reuse_outside_opt_in_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    original = settlement._TargetSettlement.settle

    def counted(self: settlement._TargetSettlement, output: str):
        calls.append(output)
        return original(self, output)

    monkeypatch.setattr(settlement._TargetSettlement, "settle", counted)
    with settlement_cache_scope():
        for _ in range(2):
            settlement.settle_assignments({"b": relation}, {"a"}, PYTHON_BACKEND, target="b")
    assert len(calls) == 2


def test_cache_does_not_survive_another_request(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    original = settlement._TargetSettlement.settle

    def counted(self: settlement._TargetSettlement, output: str):
        calls.append(output)
        return original(self, output)

    monkeypatch.setattr(settlement._TargetSettlement, "settle", counted)
    with settlement_memo_scope():
        for _ in range(2):
            with settlement_cache_scope():
                settlement.settle_assignments({"b": relation}, {"a"}, PYTHON_BACKEND, target="b")
    assert len(calls) == 2
