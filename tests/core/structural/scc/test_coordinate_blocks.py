"""Test request-scoped coordinate block discovery independently of execution."""

# ruff: noqa: PLR2004 - Fixed panel cardinalities are the subject of these tests.

import typing

import pytest

from bilt import Tensor, dim, domain, equation
from bilt.core._domain_registry import DomainRegistry
from bilt.core._structural_analysis._causalization_block_solver._coordinate_blocks import discover_coordinate_blocks
from bilt.core._structural_analysis._causalization_block_solver._errors import BiltOverconstrainedError
from bilt.core._structural_analysis._causalization_block_solver._expand_structure import (
    ExpandedConstraint,
    ExpansionSnapshot,
)
from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph
from bilt.core._structural_analysis._causalization_block_solver._structural_validation import validate_request_structure
from bilt.symbolic._dimensions import coordinate_axis

SOURCE = dim("source")
TARGET = dim("target")
SourceTensor = typing.Annotated[Tensor, domain(SOURCE)]
TargetTensor = typing.Annotated[Tensor, domain(TARGET)]
Influence = typing.Annotated[Tensor, domain(TARGET, SOURCE)]


@equation
def boundary(matrix: Influence, circulation: SourceTensor, alpha: float, rhs: TargetTensor) -> bool:
    return (matrix * circulation).sum(SOURCE) + alpha == rhs


@equation
def total(circulation: SourceTensor, total_value: float) -> bool:
    return circulation.sum(SOURCE) == total_value


@equation
def first(circulation: SourceTensor, first_value: float) -> bool:
    return circulation.isel(SOURCE, 0) == first_value


@equation
def second(circulation: SourceTensor, second_value: float) -> bool:
    return circulation.isel(SOURCE, 1) == second_value


def _snapshot(
    rows: tuple[object, ...],
    unknowns: set[str],
    known: set[str],
) -> ExpansionSnapshot:
    constraints = tuple(
        ExpandedConstraint((), typing.cast("typing.Any", row), frozenset(row.__bilt_parameters__))
        for row in rows
    )
    return ExpansionSnapshot(
        constraints=constraints,
        attributes=frozenset(unknowns | known),
        available=frozenset(known),
        unresolved=(),
        expanded_paths=frozenset(),
        query_count=0,
    )


def _registry() -> DomainRegistry:
    result = DomainRegistry()
    result.bind_axis(coordinate_axis(SOURCE, (0, 1)))
    result.bind_axis(coordinate_axis(TARGET, (0, 1)))
    return result


@pytest.mark.parametrize("output", ("circulation", "alpha"))
def test_mixed_scalar_tensor_block_is_discovered(output: str) -> None:
    snapshot = _snapshot(
        (boundary, total),
        {"circulation", "alpha"},
        {"matrix", "rhs", "total_value"},
    )
    result = discover_coordinate_blocks(
        snapshot,
        {"matrix": domain(TARGET, SOURCE), "rhs": domain(TARGET), "total_value": domain()},
        _registry(),
        output,
    )
    assert result is not None
    assert len(result.blocks) == 1
    block = result.blocks[0]
    assert set(block.unknowns) == {"circulation", "alpha"}
    assert {row.source for row in block.rows} == {boundary, total}


def test_multiple_scalar_equations_can_determine_one_tensor_port() -> None:
    snapshot = _snapshot(
        (first, second),
        {"circulation"},
        {"first_value", "second_value"},
    )
    result = discover_coordinate_blocks(
        snapshot,
        {"first_value": domain(), "second_value": domain()},
        _registry(),
        "circulation",
    )
    assert result is not None
    assert len(result.blocks) == 1
    assert result.blocks[0].unknowns == ("circulation",)
    assert len(result.blocks[0].rows) == 2


def test_missing_unknown_domain_returns_no_refinement() -> None:
    @equation
    def unconstrained_hint(x: Tensor, y: float) -> bool:
        return x.sum(SOURCE) == y

    snapshot = _snapshot((unconstrained_hint,), {"x"}, {"y"})
    assert discover_coordinate_blocks(snapshot, {"y": domain()}, _registry(), "x") is None


def test_no_operator_retains_existing_port_planner() -> None:
    @equation
    def direct(x: float, y: float) -> bool:
        return x == y

    snapshot = _snapshot((direct,), {"x"}, {"y"})
    assert discover_coordinate_blocks(snapshot, {"y": domain()}, _registry(), "x") is None


def test_request_validation_uses_coordinate_matching_before_port_dm() -> None:
    """Two scalar constraints determine both coordinates of one tensor unknown."""
    result = validate_request_structure(
        FullIncidenceGraph((first, second)),
        {"first_value", "second_value"},
        "circulation",
        input_domains={"first_value": domain(), "second_value": domain()},
        bindings=_registry(),
    )
    assert result is not None
    assert result.coordinate_blocks is not None
    assert result.coordinate_blocks.blocks[0].unknowns == ("circulation",)


def test_request_validation_keeps_existing_path_without_domain_data() -> None:
    """Earlier callers without domain bindings preserve port-level semantics."""
    with pytest.raises(BiltOverconstrainedError):
        validate_request_structure(
            FullIncidenceGraph((first, second)),
            {"first_value", "second_value"},
            "circulation",
        )
