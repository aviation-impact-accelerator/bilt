"""Inspect local model structure without selecting an execution route."""

__all__ = ["StructuralAnalysis", "StructuralBoundary", "StructuralScope", "inspect_structure"]

import dataclasses
import typing

from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._transform import BasicTransform

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._equation import BasicEquation
    from bilt.core._model_constituent import BiltModelConstituent


@dataclasses.dataclass(frozen=True)
class StructuralBoundary:
    """An unresolved child instance and the child-local ports reached so far.

    ``required`` records touched unknown ports, not a promise that the child
    can produce them. Parent constraints may also constrain these ports.
    ``available`` contains only supplied boundary values, not candidate outputs.
    """

    operation: SubSystemConstituent
    available: frozenset[str]
    required: frozenset[str]


@dataclasses.dataclass(frozen=True)
class StructuralScope:
    """Original local constraints and unresolved child references for a request.

    Names remain local to this implementation. A composing caller must qualify
    them by instance path, key constraints by path and source identity, and
    revisit parent scope when expansion exposes additional boundary connections.
    This is structural evidence, not a validity classification or execution plan.
    """

    constraints: tuple[BasicEquation | BasicTransform, ...]
    boundaries: tuple[StructuralBoundary, ...]
    attributes: frozenset[str]
    available: frozenset[str]
    required: frozenset[str]


class StructuralAnalysis:
    """Own the local incidence graph independently of execution analysers.

    Constituents are supplied at construction. Child analysis objects are
    requested only when a boundary is explicitly refined.
    """

    def __init__(self, constituents: collections.abc.Iterable[BiltModelConstituent]) -> None:
        self._incidence = FullIncidenceGraph(constituents)

    def inspect(
        self,
        available: collections.abc.Set[str],
        required: collections.abc.Set[str],
    ) -> StructuralScope:
        """Inspect this implementation level without selecting execution routes."""
        return inspect_structure(self._incidence, available, required)

    @staticmethod
    def inspect_boundary(boundary: StructuralBoundary) -> StructuralScope:
        """Query the child's analysis object, leaving nested children unresolved."""
        analysis = boundary.operation.implementation.structural_analysis()
        return analysis.inspect(boundary.available, boundary.required)


def inspect_structure(
    incidence: FullIncidenceGraph,
    available: collections.abc.Set[str],
    required: collections.abc.Set[str],
) -> StructuralScope:
    """Collect request-scoped primitives, stopping at supplied values and children.

    Equations contribute all their attributes. Transforms enter only through
    their declared output. No matching, isolation or backend operation is used;
    cycles and competing constraints are preserved. Unknown names remain visible.
    """
    supplied = frozenset(available)
    targets = frozenset(required)
    selected, child_ports, attributes = _scope_nodes(incidence, supplied, targets)

    constraints: dict[int, BasicEquation | BasicTransform] = {}
    boundaries: list[StructuralBoundary] = []
    for node in incidence.constituent_nodes:
        constituent = incidence.graph[node]
        if node in selected:
            constraints[id(constituent)] = typing.cast("BasicEquation | BasicTransform", constituent)
        if node in child_ports:
            child = typing.cast("SubSystemConstituent", constituent)
            boundaries.append(
                StructuralBoundary(
                    operation=child,
                    available=frozenset(
                        child.child_port(port) for port in child.__bilt_parameters__ if port in supplied
                    ),
                    required=frozenset(child_ports[node]),
                )
            )

    return StructuralScope(
        constraints=tuple(constraints.values()),
        boundaries=tuple(boundaries),
        attributes=frozenset(attributes),
        available=supplied,
        required=targets,
    )


def _scope_nodes(
    incidence: FullIncidenceGraph, supplied: frozenset[str], targets: frozenset[str]
) -> tuple[set[int], dict[int, set[str]], set[str]]:
    """Follow local incidence while leaving each encountered child opaque."""
    selected: set[int] = set()
    visited: set[str] = set()
    attributes = set(targets)
    pending = sorted(targets, reverse=True)
    child_ports: dict[int, set[str]] = {}

    while pending:
        attribute = pending.pop()
        if attribute in supplied or attribute in visited:
            continue
        visited.add(attribute)
        node = incidence.attribute_nodes.get(attribute)
        if node is None:
            continue

        for constituent_node in incidence.graph.neighbors(node):
            constituent = incidence.graph[constituent_node]
            if isinstance(constituent, SubSystemConstituent):
                child_ports.setdefault(constituent_node, set()).add(constituent.child_port(attribute))
                continue
            if isinstance(constituent, BasicTransform) and attribute != constituent.__bilt_name__:
                continue
            if constituent_node in selected:
                continue
            selected.add(constituent_node)
            connected = {incidence.graph[neighbor] for neighbor in incidence.graph.neighbors(constituent_node)}
            attributes.update(connected)
            pending.extend(sorted(connected - supplied, reverse=True))

    return selected, child_ports, attributes
