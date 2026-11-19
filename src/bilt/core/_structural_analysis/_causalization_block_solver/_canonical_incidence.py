"""A model-owned local incidence index shared by structural discovery and matching.

Equation identities and full incidence remain authoritative for validation.
Only compatible direct equalities are candidates for structural aliasing;
request-supplied ports split the candidate classes without changing the model.
"""

import dataclasses
import typing

from bilt.core._equation import BasicEquation
from bilt.core._structural_analysis._causalization_block_solver._alias_contraction import _direct_equal_ports

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph


@dataclasses.dataclass(frozen=True)
class AliasLink:
    node: int
    operation: BasicEquation
    left: str
    right: str


@dataclasses.dataclass(frozen=True)
class CanonicalView:
    """Request-safe canonical names with incidence to original source operations."""

    representatives: dict[str, str]
    members: dict[str, tuple[str, ...]]
    adjacency: dict[str, tuple[tuple[int, str], ...]]
    removed: frozenset[int]
    spanning: tuple[AliasLink, ...]

    def canonical(self, name: str) -> str:
        return self.representatives.get(name, name)


class CanonicalIncidenceIndex:
    """Precompute stable local incidence and candidate aliases once per model.

    View caching is keyed only by supplied ports participating in candidate
    equalities. Ordinary non-alias supplied ports affect traversal/matching but
    cannot change alias partitions, so they do not invalidate this cache.
    """

    def __init__(self, incidence: FullIncidenceGraph) -> None:
        graph = incidence.graph
        self.rows: dict[int, BiltModelConstituent] = {node: graph[node] for node in incidence.constituent_nodes}
        self.row_attributes: dict[int, tuple[str, ...]] = {
            node: tuple(sorted(typing.cast("str", graph[neighbor]) for neighbor in graph.neighbors(node)))
            for node in incidence.constituent_nodes
        }
        self.attribute_rows: dict[str, tuple[int, ...]] = {
            name: tuple(graph.neighbors(node)) for name, node in incidence.attribute_nodes.items()
        }
        candidates: list[AliasLink] = []
        for node in incidence.constituent_nodes:
            operation = self.rows[node]
            if not isinstance(operation, BasicEquation):
                continue
            pair = _direct_equal_ports(operation)
            if pair is None or any(name not in self.attribute_rows for name in pair):
                continue
            candidates.append(AliasLink(node, operation, *pair))
        self.links = tuple(candidates)
        self.alias_endpoints = frozenset(name for link in self.links for name in (link.left, link.right))
        self._views: dict[frozenset[str], CanonicalView] = {}
        # Precompute the common unsupplied partition while constructing the model.
        self._views[frozenset()] = self._build_view(frozenset())

    def view(self, available: collections.abc.Collection[str]) -> CanonicalView:
        """Return the cached partition, splitting at supplied alias endpoints."""
        blocked = frozenset(available) & self.alias_endpoints
        if cached := self._views.get(blocked):
            return cached
        view = self._build_view(blocked)
        if len(self._views) >= 32:  # noqa: PLR2004 - bounded model-local cache
            # Preserve the construction-time common case, evicting an older variant.
            oldest = next(key for key in self._views if key)
            del self._views[oldest]
        self._views[blocked] = view
        return view

    def _build_view(self, blocked: frozenset[str]) -> CanonicalView:
        parent = {name: name for name in self.attribute_rows}

        def find(name: str) -> str:
            while parent[name] != name:
                parent[name] = parent[parent[name]]
                name = parent[name]
            return name

        spanning: list[AliasLink] = []
        for link in self.links:
            if link.left in blocked or link.right in blocked:
                continue
            left, right = find(link.left), find(link.right)
            if left == right:
                continue  # Keep redundant/cyclic equations as structural evidence.
            parent[max(left, right)] = min(left, right)
            spanning.append(link)

        representatives = {name: find(name) for name in sorted(parent)}
        groups: dict[str, list[str]] = {}
        neighbors: dict[str, list[tuple[int, str]]] = {}
        for name, representative in representatives.items():
            groups.setdefault(representative, []).append(name)
            neighbors.setdefault(representative, []).extend((node, name) for node in self.attribute_rows[name])

        return CanonicalView(
            representatives,
            {name: tuple(members) for name, members in groups.items()},
            {name: tuple(entries) for name, entries in neighbors.items()},
            frozenset(link.node for link in spanning),
            tuple(spanning),
        )
