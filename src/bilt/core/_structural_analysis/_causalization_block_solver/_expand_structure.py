"""Expand relevant subsystem structure without selecting execution routes."""

__all__ = ["ExpansionSnapshot", "StructuralExpansion", "expand_structure"]

import dataclasses
import functools
import typing

import rustworkx as rx

from bilt.core._errors import BiltAnalysisError
from bilt.core._structural_analysis._causalization_block_solver._structural_scope import inspect_structure
from bilt.core._transform import BasicTransform

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._equation import BasicEquation
    from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph
    from bilt.core._structural_analysis._causalization_block_solver._structural_scope import (
        StructuralBoundary,
        StructuralScope,
    )


@dataclasses.dataclass(frozen=True)
class ExpandedConstraint:
    """An original primitive constraint qualified by its subsystem instance path."""

    path: tuple[str, ...]
    operation: BasicEquation | BasicTransform
    attributes: frozenset[str]

    @property
    def label(self) -> str:
        """Return the qualified source name for diagnostics."""
        return _prefix(self.path) + self.operation.__bilt_name__


@dataclasses.dataclass(frozen=True)
class UnresolvedBoundary:
    """A child instance that has been encountered but not structurally inspected.

    Ports in boundary.required are touched unknowns, not promised outputs.
    The path distinguishes instances that share one implementation.
    """

    path: tuple[str, ...]
    boundary: StructuralBoundary


@dataclasses.dataclass(frozen=True)
class ExpansionSnapshot:
    """Known primitives and explicit uncertainty; this is not a DM classification."""

    constraints: tuple[ExpandedConstraint, ...]
    attributes: frozenset[str]
    available: frozenset[str]
    unresolved: tuple[UnresolvedBoundary, ...]
    expanded_paths: frozenset[tuple[str, ...]]
    query_count: int

    @property
    def complete(self) -> bool:
        """Whether every boundary encountered in this request scope is expanded."""
        return not self.unresolved

    def to_incidence(self) -> tuple[rx.PyGraph, dict[str, int]]:
        """Build the definitive graph only when no relevant boundary is unresolved."""
        if not self.complete:
            message = "Cannot build definitive structural incidence with unresolved subsystem boundaries."
            raise BiltAnalysisError(message)
        graph = rx.PyGraph(multigraph=False)
        attributes = {name: graph.add_node(name) for name in sorted(self.attributes - self.available)}
        for constraint in self.constraints:
            node = graph.add_node(constraint.label)
            for attribute in sorted(constraint.attributes - self.available):
                graph.add_edge(node, attributes[attribute], None)
        return graph, attributes


@dataclasses.dataclass(frozen=True)
class _Context:
    query: collections.abc.Callable[
        [collections.abc.Set[str], collections.abc.Set[str]],
        StructuralScope,
    ]
    ancestors: frozenset[int]


class StructuralExpansion:
    """Resume structural discovery by explicitly selecting unresolved child paths.

    Each action drains local and already-open ancestor queries to a fixed point.
    Newly encountered children stay opaque until expand() is called for them.
    This state is request-local and never consults matching or execution failures.
    """

    def __init__(self, incidence: FullIncidenceGraph, available: collections.abc.Set[str], output: str) -> None:
        self.available = frozenset(available)
        self.contexts: dict[tuple[str, ...], _Context] = {
            (): _Context(functools.partial(inspect_structure, incidence), frozenset())
        }
        self.attributes: set[str] = set()
        self.rows: dict[tuple[tuple[str, ...], int], ExpandedConstraint] = {}
        self.boundaries: dict[tuple[str, ...], StructuralBoundary] = {}
        self.pending: list[tuple[tuple[str, ...], str]] = []
        self.scheduled: set[tuple[tuple[str, ...], str]] = set()
        self.query_count = 0

        self._add_attributes({output})
        self._drain()

    def snapshot(self) -> ExpansionSnapshot:
        """Capture discovered constraints and unresolved boundaries without opening children."""
        return ExpansionSnapshot(
            constraints=tuple(self.rows.values()),
            attributes=frozenset(self.attributes),
            available=self.available,
            expanded_paths=frozenset(self.contexts) - {()},
            query_count=self.query_count,
            unresolved=tuple(UnresolvedBoundary(path, self.boundaries[path]) for path in sorted(self.boundaries)),
        )

    def expand(self, path: tuple[str, ...]) -> ExpansionSnapshot:
        """Open one encountered child; nested children remain explicitly unresolved.

        Reopening an expanded path is a no-op. Unknown paths raise KeyError.
        Newly exposed attributes are also interrogated in open ancestor contexts.
        """
        self._expand(path)
        return self.snapshot()

    def _expand(self, path: tuple[str, ...]) -> None:
        if path not in self.contexts:
            boundary = self.boundaries[path]
            self._open(path[:-1], boundary)
            del self.boundaries[path]
            self._drain()

    def collect(self) -> tuple[rx.PyGraph, dict[str, int]]:
        """Expand every encountered boundary for existing definitive validation."""
        while self.boundaries:
            self._expand(min(self.boundaries))
        return self.snapshot().to_incidence()

    def _drain(self) -> None:
        while self.pending:
            namespace, attribute = self.pending.pop()
            prefix = _prefix(namespace)
            supplied = {name[len(prefix) :] for name in self.available if name.startswith(prefix)}
            scope = self.contexts[namespace].query(supplied, {attribute})
            self.query_count += 1
            self._merge(namespace, scope)

    def _merge(self, namespace: tuple[str, ...], scope: StructuralScope) -> None:
        prefix = _prefix(namespace)
        self._add_attributes({prefix + attribute for attribute in scope.attributes})
        for constraint in scope.constraints:
            connected = set(constraint.__bilt_parameters__)
            if isinstance(constraint, BasicTransform):
                connected.add(constraint.__bilt_name__)
            self.rows[namespace, id(constraint)] = ExpandedConstraint(
                namespace, constraint, frozenset(prefix + attribute for attribute in connected)
            )
        for boundary in scope.boundaries:
            path = (*namespace, boundary.operation.__bilt_name__)
            if path in self.contexts:
                continue
            previous = self.boundaries.get(path)
            self.boundaries[path] = (
                boundary
                if previous is None
                else dataclasses.replace(boundary, required=previous.required | boundary.required)
            )

    def _open(self, namespace: tuple[str, ...], boundary: StructuralBoundary) -> None:
        path = (*namespace, boundary.operation.__bilt_name__)
        if path in self.contexts:
            return
        implementation = boundary.operation.implementation
        ancestors = self.contexts[namespace].ancestors
        if id(implementation) in ancestors:
            message = f"Recursive subsystem containment at {'.'.join(path)!r} is not supported."
            raise BiltAnalysisError(message)
        self.contexts[path] = _Context(implementation.structural_analysis().inspect, ancestors | {id(implementation)})
        # Expansion can expose ports referenced by parent or sibling constraints.
        # Replay existing attributes into the new child context, too.
        for attribute in sorted(self.attributes):
            self._schedule(path, attribute)

    def _add_attributes(self, attributes: collections.abc.Set[str]) -> None:
        for attribute in sorted(attributes - self.attributes):
            self.attributes.add(attribute)
            for namespace in self.contexts:
                self._schedule(namespace, attribute)

    def _schedule(self, namespace: tuple[str, ...], attribute: str) -> None:
        prefix = _prefix(namespace)
        if attribute in self.available or not attribute.startswith(prefix):
            return
        job = namespace, attribute[len(prefix) :]
        if job not in self.scheduled:
            self.scheduled.add(job)
            self.pending.append(job)


def _prefix(namespace: tuple[str, ...]) -> str:
    return ".".join((*namespace, "")) if namespace else ""


def expand_structure(
    incidence: FullIncidenceGraph,
    available: collections.abc.Set[str],
    output: str,
) -> tuple[rx.PyGraph, dict[str, int]]:
    """Resolve relevant boundaries and return primitive incidence plus attributes.

    Child queries are lazy and backend-independent. Each newly exposed attribute
    is also inspected in its ancestor contexts, discovering parent connections.
    Each context/attribute pair is queried once. Original source identity plus
    instance path prevents duplicate constraints without merging distinct children.
    Supplied attributes are fixed; cycles among unknowns remain in the graph.
    """
    return StructuralExpansion(incidence, available, output).collect()
