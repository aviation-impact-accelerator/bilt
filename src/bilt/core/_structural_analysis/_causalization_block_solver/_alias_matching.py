"""Experimental local matching with canonical aliases and original-name restoration.

The definitive structural validation is unchanged. This module only substitutes
a smaller request-local candidate matching graph, then restores ordinary Bilt
assignments before the existing planner sees them. Execution is still performed
by original operations, including the direct equality connections.
"""

import contextlib
import contextvars
import dataclasses
import typing
from collections import deque
from itertools import islice

import rustworkx as rx

from bilt.core._structural_analysis._causalization_block_solver import _search as search
from bilt.core._structural_analysis._causalization_block_solver._matching import build_matching_graph
from bilt.core._structural_analysis._plan import PlanStep

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._equation import BasicEquation
    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph
    from bilt.core._structural_analysis._realize import RealizationFailure


_enabled: contextvars.ContextVar[bool] = contextvars.ContextVar("bilt_alias_matching", default=False)


@contextlib.contextmanager
def alias_matching_scope() -> collections.abc.Generator[None]:
    """Opt in to alias-aware matching for one evaluation and its child calls."""
    token = _enabled.set(True)
    try:
        yield
    finally:
        _enabled.reset(token)


def alias_matching_enabled() -> bool:
    return _enabled.get()


@dataclasses.dataclass(frozen=True, eq=False)
class _Candidate:
    """A matching-only proxy retaining the original output of each candidate edge."""

    original: BiltModelConstituent
    outputs: dict[str, str]
    __bilt_parameters__: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class AliasMatching:
    """A reduced matching graph and the evidence needed to restore Bilt names."""

    graph: rx.PyGraph
    representatives: dict[str, str]
    alias_edges: tuple[tuple[BasicEquation, str, str], ...]
    original_nodes: int
    original_edges: int

    def canonical(self, name: str) -> str:
        return self.representatives.get(name, name)

    def restore(self, selected: collections.abc.Mapping[str, _Candidate]) -> dict[str, BiltModelConstituent]:
        """Restore original production/connection assignments from a matching."""
        assignments: dict[str, BiltModelConstituent] = {}
        anchors: dict[str, str] = {}
        for representative, candidate in selected.items():
            native = candidate.outputs[representative]
            assignments[native] = candidate.original
            anchors[representative] = native

        adjacency: dict[str, list[tuple[str, BasicEquation]]] = {}
        for operation, left, right in self.alias_edges:
            adjacency.setdefault(left, []).append((right, operation))
            adjacency.setdefault(right, []).append((left, operation))

        for anchor in anchors.values():
            visited = {anchor}
            pending = deque([anchor])
            while pending:
                name = pending.popleft()
                for other, operation in adjacency.get(name, ()):
                    if other in visited:
                        continue
                    visited.add(other)
                    assignments[other] = operation
                    pending.append(other)
        return assignments

    @property
    def removed_nodes(self) -> int:
        return self.original_nodes - self.graph.num_nodes()

    @property
    def removed_edges(self) -> int:
        return self.original_edges - self.graph.num_edges()


def build_alias_matching(
    incidence: FullIncidenceGraph,
    available: collections.abc.Set[str],
    *,
    matching_graph: rx.PyGraph | None = None,
) -> AliasMatching | None:
    """Contract independent direct-equality ports for one matching request.

    An alias class is never joined across a supplied input. Return None when no
    safe alias exists, leaving normal matching and failure handling untouched.
    """
    supplied = frozenset(available)
    view = incidence.canonical_index.view(supplied)
    if not view.spanning:
        return None
    representatives = view.representatives
    removed = view.removed
    alias_edges = tuple((link.operation, link.left, link.right) for link in view.spanning)

    original = matching_graph if matching_graph is not None else build_matching_graph(incidence, supplied)
    graph = rx.PyGraph(multigraph=False)
    attributes = {
        name: graph.add_node(name)
        for name in sorted({representatives[name] for name in representatives if name not in supplied})
    }
    for node in original.node_indices():
        source = original[node]
        if isinstance(source, str):
            continue
        operation = source.operation if isinstance(source, PlanStep) else typing.cast("BiltModelConstituent", source)
        # Direct aliases already have a retained spanning forest for recovery.
        if node in removed:
            continue
        natives: dict[str, str] = {}
        for neighbor in original.neighbors(node):
            name = original[neighbor]
            if not isinstance(name, str):
                continue
            canonical = representatives[name]
            if canonical not in natives or name < natives[canonical]:
                natives[canonical] = name
        if not natives:
            continue
        candidate = _Candidate(
            original=operation,
            outputs=natives,
            __bilt_parameters__=tuple(sorted({representatives[name] for name in operation.__bilt_parameters__})),
        )
        candidate_node = graph.add_node(candidate)
        for canonical in sorted(natives):
            # Preserve the source matching edge weight where possible.
            source_node = next(
                neighbor
                for neighbor in original.neighbors(node)
                if isinstance(original[neighbor], str) and representatives[original[neighbor]] == canonical
            )
            graph.add_edge(candidate_node, attributes[canonical], original.get_edge_data(node, source_node))

    return AliasMatching(
        graph=graph,
        representatives=representatives,
        alias_edges=tuple(alias_edges),
        original_nodes=original.num_nodes(),
        original_edges=original.num_edges(),
    )


def candidate_assignments(
    incidence: FullIncidenceGraph,
    original_graph: rx.PyGraph,
    available: collections.abc.Set[str],
    failures: collections.abc.Mapping[PlanStep, RealizationFailure],
    output: str,
) -> collections.abc.Iterator[dict[str, BiltModelConstituent]]:
    """Try reduced matching first, then exhaust the unchanged production search.

    The ordinary fallback is important: the alias view only proves that a
    reduced structural matching exists, not that its native equation directions
    can be realised with the requested domains or chosen child routes.
    """
    contracted = (
        build_alias_matching(incidence, available, matching_graph=original_graph) if alias_matching_enabled() else None
    )
    if contracted is not None:
        target = None if output in available else contracted.canonical(output)
        # Try only the first reduced matching; exploring alternatives can expand exponentially.
        for assignments in islice(search.candidate_assignments(contracted.graph, {}, output=target), 1):
            restored = contracted.restore(typing.cast("dict[str, _Candidate]", assignments))
            if output in available or output in restored:
                # Do not recursively enable experimental matching in child-route
                # searches triggered by this candidate's settlement or realisation.
                token = _enabled.set(False)
                try:
                    yield restored
                finally:
                    _enabled.reset(token)

    for assignments in search.candidate_assignments(
        original_graph,
        failures,
        output=None if output in available else output,
    ):
        # The fallback is entirely native, including nested child searches.
        token = _enabled.set(False)
        try:
            yield assignments
        finally:
            _enabled.reset(token)
