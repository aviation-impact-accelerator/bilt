"""Expand relevant subsystem structure without selecting execution routes."""

__all__ = ["expand_structure"]

import dataclasses
import functools
import typing

import rustworkx as rx

from bilt.core._errors import BiltAnalysisError
from bilt.core._structural_analysis._causalization_block_solver._structural_scope import inspect_structure
from bilt.core._transform import BasicTransform

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph
    from bilt.core._structural_analysis._causalization_block_solver._structural_scope import (
        StructuralBoundary,
        StructuralScope,
    )


@dataclasses.dataclass(frozen=True)
class _Context:
    query: collections.abc.Callable[
        [collections.abc.Set[str], collections.abc.Set[str]],
        StructuralScope,
    ]
    ancestors: frozenset[int]


class _Expansion:
    def __init__(self, incidence: FullIncidenceGraph, available: collections.abc.Set[str]) -> None:
        self.available = frozenset(available)
        self.contexts: dict[tuple[str, ...], _Context] = {
            (): _Context(functools.partial(inspect_structure, incidence), frozenset())
        }
        self.attributes: set[str] = set()
        self.rows: dict[tuple[tuple[str, ...], int], tuple[str, frozenset[str]]] = {}
        self.pending: list[tuple[tuple[str, ...], str]] = []
        self.scheduled: set[tuple[tuple[str, ...], str]] = set()

    def collect(self, output: str) -> tuple[rx.PyGraph, dict[str, int]]:
        self._add_attributes({output})
        while self.pending:
            namespace, attribute = self.pending.pop()
            prefix = _prefix(namespace)
            supplied = {name[len(prefix) :] for name in self.available if name.startswith(prefix)}
            scope = self.contexts[namespace].query(supplied, {attribute})
            self._merge(namespace, scope)

        graph = rx.PyGraph(multigraph=False)
        attributes = {name: graph.add_node(name) for name in sorted(self.attributes - self.available)}
        for label, connected in self.rows.values():
            node = graph.add_node(label)
            for attribute in sorted(connected - self.available):
                graph.add_edge(node, attributes[attribute], None)
        return graph, attributes

    def _merge(self, namespace: tuple[str, ...], scope: StructuralScope) -> None:
        prefix = _prefix(namespace)
        self._add_attributes({prefix + attribute for attribute in scope.attributes})
        for constraint in scope.constraints:
            connected = set(constraint.__bilt_parameters__)
            if isinstance(constraint, BasicTransform):
                connected.add(constraint.__bilt_name__)
            self.rows[namespace, id(constraint)] = (
                prefix + constraint.__bilt_name__,
                frozenset(prefix + attribute for attribute in connected),
            )
        for boundary in scope.boundaries:
            self._open(namespace, boundary)

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
    return _Expansion(incidence, available).collect(output)
