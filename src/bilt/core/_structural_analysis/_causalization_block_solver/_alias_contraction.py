"""Experimental contraction of direct equality ports in a complete structural scope.

This representation is for differential validation and profiling. Execution plans
and source equations retain their original identities until an alias-aware planner
can be validated separately.
"""

import dataclasses
import typing

import rustworkx as rx

from bilt.core._equation import BasicEquation
from bilt.core._errors import BiltAnalysisError
from bilt.symbolic._atoms import SymbolType

if typing.TYPE_CHECKING:
    from bilt.core._port_declaration import PortDeclaration
    from bilt.core._structural_analysis._causalization_block_solver._expand_structure import (
        ExpandedConstraint,
        ExpansionSnapshot,
    )


@dataclasses.dataclass(frozen=True)
class AliasIncidence:
    graph: rx.PyGraph
    attribute_nodes: dict[str, int]
    representatives: dict[str, str]
    eliminated: tuple[ExpandedConstraint, ...]
    retained: tuple[ExpandedConstraint, ...]

    @property
    def distinct_attributes(self) -> int:
        return len(set(self.attribute_nodes.values()))


def _compatible(left: PortDeclaration | None, right: PortDeclaration | None) -> bool:
    if left is None or right is None:
        return left is right
    return (
        left.dimensionality == right.dimensionality
        and left.absolute == right.absolute
        and left.over_a_domain == right.over_a_domain
        and left.scalar_only == right.scalar_only
        and left.required_domain == right.required_domain
    )


def _alias_pair(row: ExpandedConstraint) -> tuple[str, str] | None:
    operation = row.operation
    if not isinstance(operation, BasicEquation):
        return None
    tree = operation.__bilt_equation_tree__
    if not tree.is_equality or not tree.left.is_leaf or not tree.right.is_leaf:
        return None
    if tree.left.head.atom_type is not SymbolType or tree.right.head.atom_type is not SymbolType:
        return None
    left = typing.cast("str", tree.left.head.value)
    right = typing.cast("str", tree.right.head.value)
    if left == right or not _compatible(
        operation.__bilt_port_declarations__.get(left),
        operation.__bilt_port_declarations__.get(right),
    ):
        return None
    prefix = ".".join((*row.path, "")) if row.path else ""
    qualified = (prefix + left, prefix + right)
    return qualified if row.attributes == frozenset(qualified) else None


def contract_aliases(snapshot: ExpansionSnapshot) -> AliasIncidence:
    """Contract independent, unsupplied, metadata-compatible equality ports.

    This is not an execution representation: it retains original name mappings
    and original source equations for future validation of a full alias pipeline.
    """
    if not snapshot.complete:
        msg = "Cannot contract aliases before all relevant child boundaries are expanded."
        raise BiltAnalysisError(msg)

    parent = {name: name for name in snapshot.attributes}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    eliminated: list[ExpandedConstraint] = []
    eliminated_indices: set[int] = set()
    for index, row in enumerate(snapshot.constraints):
        pair = _alias_pair(row)
        if pair is None or any(name in snapshot.available or name not in parent for name in pair):
            continue
        left, right = find(pair[0]), find(pair[1])
        if left == right:
            # Retain a redundant/cyclic connection as a constraint.
            continue
        parent[max(left, right)] = min(left, right)
        eliminated.append(row)
        eliminated_indices.add(index)

    representatives = {name: find(name) for name in sorted(snapshot.attributes)}
    unknowns = sorted({representatives[name] for name in snapshot.attributes - snapshot.available})
    graph = rx.PyGraph(multigraph=False)
    canonical_nodes = dict(zip(unknowns, graph.add_nodes_from(unknowns), strict=True))
    attribute_nodes = {
        name: canonical_nodes[representatives[name]] for name in sorted(snapshot.attributes - snapshot.available)
    }
    rows = [row for index, row in enumerate(snapshot.constraints) if index not in eliminated_indices]
    constraint_nodes = graph.add_nodes_from([row.label for row in rows])
    edges = [
        (node, canonical_nodes[representatives[name]], None)
        for row, node in zip(rows, constraint_nodes, strict=True)
        for name in sorted({representatives[name] for name in row.attributes - snapshot.available})
    ]
    graph.add_edges_from(edges)
    return AliasIncidence(graph, attribute_nodes, representatives, tuple(eliminated), tuple(rows))
