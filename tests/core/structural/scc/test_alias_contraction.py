# ruff: noqa: PLR2004, FBT003

"""Differential checks for experimental direct port alias contraction."""

from dataclasses import replace
from typing import Annotated

import pytest

from bilt import Implementation, Interface, QuantityTensor, equation, field, system, transform
from bilt.core._structural_analysis._causalization_block_solver._alias_contraction import contract_aliases
from bilt.core._structural_analysis._causalization_block_solver._errors import (
    BiltOverconstrainedError,
    BiltUnderconstrainedError,
)
from bilt.core._structural_analysis._causalization_block_solver._expand_structure import (
    ExpandedConstraint,
    ExpansionSnapshot,
    StructuralExpansion,
    UnresolvedBoundary,
)
from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph
from bilt.core._structural_analysis._causalization_block_solver._structural_scope import StructuralBoundary
from bilt.core._structural_analysis._causalization_block_solver._structural_validation import validate_structure
from bilt.core._sub_system import SubSystemConstituent
from bilt.symbolic._dimensions import dim
from bilt.symbolic._domains import domain
from bilt.units.dimensions import DIMENSIONLESS


@system
class Ports(metaclass=Interface):
    x: float = field()
    y: float = field()


@equation
def connect(a: float, b: float) -> bool:
    return a == b


@equation
def relation(b: float, c: float) -> bool:
    return c == 2 * b


@equation
def offset(a: float, b: float) -> bool:
    return a == b + 1


@transform
def c(b: float) -> float:
    return 2 * b


@equation
def bridge(left: Ports, right: Ports) -> bool:
    return left.y == right.x


@equation
def expose(output: float, right: Ports) -> bool:
    return output == right.y


@equation
def child_relation(x: float, y: float) -> bool:
    return y == 3 * x


def _snapshot(rows: tuple[ExpandedConstraint, ...], *, supplied: frozenset[str] = frozenset()) -> ExpansionSnapshot:
    return ExpansionSnapshot(
        constraints=rows,
        attributes=frozenset().union(*(row.attributes for row in rows)),
        available=supplied,
        unresolved=(),
        expanded_paths=frozenset(),
        query_count=0,
    )


def _classification(snapshot: ExpansionSnapshot, contracted: bool, output: str) -> str:
    if contracted:
        incidence = contract_aliases(snapshot)
        graph = incidence.graph
        names = incidence.attribute_nodes
    else:
        graph, names = snapshot.to_incidence()
    try:
        validate_structure(
            graph,
            frozenset(names.values()),
            frozenset(),
            required_nodes=frozenset({names[output]}),
        )
    except (BiltOverconstrainedError, BiltUnderconstrainedError) as error:
        return type(error).__name__
    return "well_determined"


def test_direct_connection_contracts_a_variable_and_a_constraint() -> None:
    snapshot = _snapshot(
        (
            ExpandedConstraint((), connect, frozenset({"a", "b"})),
            ExpandedConstraint((), relation, frozenset({"b", "c"})),
        )
    )
    result = contract_aliases(snapshot)
    assert len(result.eliminated) == 1
    assert result.representatives["a"] == result.representatives["b"]
    assert result.distinct_attributes == 2
    assert result.graph.num_nodes() == 3
    assert _classification(snapshot, False, "c") == _classification(snapshot, True, "c")


def test_arithmetic_and_transforms_are_never_aliases() -> None:
    snapshot = _snapshot(
        (
            ExpandedConstraint((), offset, frozenset({"a", "b"})),
            ExpandedConstraint((), c, frozenset({"b", "c"})),
        )
    )
    result = contract_aliases(snapshot)
    assert not result.eliminated
    assert result.distinct_attributes == 3


@pytest.mark.parametrize("supplied", (frozenset({"a"}), frozenset({"b"}), frozenset({"a", "b"})))
def test_supplied_ports_are_not_contracted_without_value_consistency_checks(supplied: frozenset[str]) -> None:
    snapshot = _snapshot((ExpandedConstraint((), connect, frozenset({"a", "b"})),), supplied=supplied)
    assert not contract_aliases(snapshot).eliminated


def test_nested_child_paths_keep_distinct_instance_identifiers() -> None:
    child = Implementation(Ports, [child_relation])
    left = SubSystemConstituent("left", child)
    right = SubSystemConstituent("right", child)
    expansion = StructuralExpansion(FullIncidenceGraph([left, right, bridge, expose]), {"left.x"}, "output")
    expansion.collect()
    snapshot = expansion.snapshot()
    assert snapshot.complete
    result = contract_aliases(snapshot)
    assert result.representatives["left.y"] == result.representatives["right.x"]
    assert result.representatives["left.x"] != result.representatives["right.y"]
    assert len(result.eliminated) == 2
    assert _classification(snapshot, False, "output") == _classification(snapshot, True, "output")


def test_cycle_remains_a_source_constraint() -> None:
    row = ExpandedConstraint((), connect, frozenset({"a", "b"}))
    # The same original operation can appear in different source paths.
    snapshot = _snapshot(
        (
            row,
            ExpandedConstraint((), connect, frozenset({"a", "b"})),
        )
    )
    result = contract_aliases(snapshot)
    assert len(result.eliminated) == 1
    assert result.graph.num_nodes() == 2



def test_repeated_same_source_object_is_not_removed_twice() -> None:
    row = ExpandedConstraint((), connect, frozenset({"a", "b"}))
    snapshot = _snapshot((row, row))
    result = contract_aliases(snapshot)
    assert len(result.eliminated) == 1
    assert len(result.retained) == 1
    assert result.retained[0] is row
    assert result.graph.num_nodes() == 2
    assert _classification(snapshot, False, "a") == _classification(snapshot, True, "a")


def test_multi_hop_alias_chain_keeps_one_representative() -> None:
    snapshot = _snapshot(
        (
            ExpandedConstraint((), connect, frozenset({"a", "b"})),
            ExpandedConstraint((), connect, frozenset({"b", "c"})),
            ExpandedConstraint((), relation, frozenset({"c", "d"})),
        )
    )
    result = contract_aliases(snapshot)
    assert len(result.eliminated) == 2
    assert result.representatives["a"] == result.representatives["b"] == result.representatives["c"]
    assert result.distinct_attributes == 2
    assert _classification(snapshot, False, "d") == _classification(snapshot, True, "d")


@pytest.mark.parametrize(
    ("rows", "output", "expected"),
    (
        (
            (ExpandedConstraint((), connect, frozenset({"a", "b"})),),
            "a",
            "BiltUnderconstrainedError",
        ),
        (
            (
                ExpandedConstraint((), connect, frozenset({"a", "b"})),
                ExpandedConstraint((), connect, frozenset({"a", "b"})),
                ExpandedConstraint((), connect, frozenset({"a", "b"})),
            ),
            "a",
            "BiltOverconstrainedError",
        ),
    ),
)
def test_invalid_structural_classification_is_preserved(
    rows: tuple[ExpandedConstraint, ...], output: str, expected: str
) -> None:
    snapshot = _snapshot(rows)
    assert _classification(snapshot, False, output) == expected
    assert _classification(snapshot, True, output) == expected



year = dim("year")
scenario = dim("scenario")


@equation
def same_tensor_domain(
    first: Annotated[QuantityTensor, DIMENSIONLESS, domain(year)],
    second: Annotated[QuantityTensor, DIMENSIONLESS, domain(year)],
) -> bool:
    return first == second


@equation
def different_tensor_domain(
    first: Annotated[QuantityTensor, DIMENSIONLESS, domain(year)],
    second: Annotated[QuantityTensor, DIMENSIONLESS, domain(scenario)],
) -> bool:
    return first == second


def test_named_tensor_domains_must_match_before_aliasing() -> None:
    matching = _snapshot((ExpandedConstraint((), same_tensor_domain, frozenset({"first", "second"})),))
    mismatch = _snapshot((ExpandedConstraint((), different_tensor_domain, frozenset({"first", "second"})),))
    assert len(contract_aliases(matching).eliminated) == 1
    assert not contract_aliases(mismatch).eliminated


def test_incomplete_snapshot_is_not_contractible() -> None:
    snapshot = _snapshot(())
    child = SubSystemConstituent("child", Implementation(Ports, [child_relation]))
    incomplete = replace(
        snapshot,
        unresolved=(UnresolvedBoundary(("child",), StructuralBoundary(child, frozenset(), frozenset({"x"}))),),
    )
    with pytest.raises(Exception, match="before all relevant child boundaries"):
        contract_aliases(incomplete)
