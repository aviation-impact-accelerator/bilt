# ruff: noqa: PLR2004

"""Check structural preparation reuse without sharing execution decisions."""

import pytest
import rustworkx as rx

from bilt import Implementation, Interface, equation, field, system, transform
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis._causalization_block_solver import _structural_validation
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.core._structural_analysis._causalization_block_solver._expand_structure import StructuralExpansion
from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph
from bilt.core._structural_analysis._causalization_block_solver._structural_validation import (
    shares_equation_scope,
    validate_request_structure,
)
from bilt.core._sub_system import SubSystemConstituent
from bilt.symbolic._domains import Domain


@system
class Ports(metaclass=Interface):
    x: float = field()
    y: float = field()
    z: float = field()
    w: float = field()


@equation
def relation_y(x: float, y: float) -> bool:
    return y == x + 2.0


@equation
def relation_z(y: float, z: float) -> bool:
    return z == 3.0 * y


@transform
def w(z: float) -> float:
    return z + 1.0


def test_equation_equivalence_does_not_reverse_transforms() -> None:
    graph = FullIncidenceGraph([relation_y, relation_z, w])
    structure = validate_request_structure(graph, {"x"}, "z")
    assert structure is not None
    assert shares_equation_scope(structure.snapshot, "z", "y")
    assert not shares_equation_scope(structure.snapshot, "z", "w")

    reverse = validate_request_structure(graph, {"x"}, "w")
    assert reverse is not None
    assert not shares_equation_scope(reverse.snapshot, "w", "z")

    direct = validate_request_structure(graph, {"x"}, "y")
    reused = validate_request_structure(graph, {"x"}, "y", prepared=structure.snapshot)
    assert direct is not None
    assert reused is not None
    assert reused.well_determined == direct.well_determined
    assert {row.operation for row in reused.snapshot.constraints} == {
        row.operation for row in direct.snapshot.constraints
    }


def test_equivalent_outputs_reuse_expansion_but_validate_each_output(monkeypatch: pytest.MonkeyPatch) -> None:
    analyser = CausalizationStructuralAnalyzer([relation_y, relation_z, w], PYTHON_BACKEND)
    original = StructuralExpansion.collect
    calls = []

    def collect(self: StructuralExpansion) -> tuple[rx.PyGraph, dict[str, int]]:
        calls.append(self)
        return original(self)

    monkeypatch.setattr(StructuralExpansion, "collect", collect)
    with EvaluationContext():
        for output, expected in (("z", 15.0), ("y", 5.0)):
            request = EvaluationRequest(output=output, input_domains={"x": Domain.scalar()})
            case = analyser.analyse(request)
            assert PYTHON_BACKEND.chain_case(case).callable(x=3.0) == expected
    assert len(calls) == 1

    with EvaluationContext():
        request = EvaluationRequest(output="w", input_domains={"x": Domain.scalar()})
        case = analyser.analyse(request)
        assert PYTHON_BACKEND.chain_case(case).callable(x=3.0) == 16.0
    assert len(calls) == 2


def test_different_supplied_inputs_remain_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    analyser = CausalizationStructuralAnalyzer([relation_y, relation_z], PYTHON_BACKEND)
    original = _structural_validation.StructuralExpansion.collect
    calls = []

    def collect(self: StructuralExpansion):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(StructuralExpansion, "collect", collect)
    with EvaluationContext():
        for available in ({"x"}, {"y"}):
            request = EvaluationRequest(output="z", input_domains={key: Domain.scalar() for key in available})
            analyser.analyse(request)
    assert len(calls) == 2


def test_shared_nested_children_retain_instance_rows() -> None:
    child = Implementation(Ports, [relation_y, relation_z])
    left = SubSystemConstituent("left", child)
    right = SubSystemConstituent("right", child)

    @equation
    def combine(answer: float, left: Ports, right: Ports) -> bool:
        return answer == left.z + right.z

    graph = FullIncidenceGraph([left, right, combine])
    structure = validate_request_structure(graph, {"left.x", "right.x"}, "answer")
    assert structure is not None
    paths = {row.path for row in structure.snapshot.constraints if row.operation is relation_y}
    assert paths == {("left",), ("right",)}
    assert not shares_equation_scope(structure.snapshot, "answer", "left.x")


@pytest.mark.parametrize("output", ("y", "z", "w"))
def test_cached_and_uncached_validation_agree_for_output(output: str) -> None:
    graph = FullIncidenceGraph([relation_y, relation_z, w])
    cached = validate_request_structure(graph, {"x"}, "z")
    assert cached is not None
    original = validate_request_structure(graph, {"x"}, output)
    prepared = cached.snapshot if shares_equation_scope(cached.snapshot, "z", output) else None
    obtained = validate_request_structure(graph, {"x"}, output, prepared=prepared)
    assert obtained == original
