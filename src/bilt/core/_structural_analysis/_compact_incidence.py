"""Compact structural incidence for equations with named tensor dimensions.

This describes *possible* coordinate dependencies without building one graph
vertex per scalar tensor element or one edge per possible coefficient. It is
independent of numerical values, linearity, and which variable is requested.
A bounded graph conversion provides a compatibility path for small problems.
"""

__all__ = [
    "BiltCompactIncidenceError",
    "CompactIncidence",
    "CompactResidual",
    "DependencyPath",
    "ResidualCoordinate",
    "UnknownCoordinate",
    "build_compact_incidence",
]

import dataclasses
import itertools
import math
import typing

import rustworkx as rx

from bilt.symbolic._atoms import SymbolType
from bilt.symbolic._domains import Domain, union
from bilt.symbolic._operator_semantics import CoordinateRelation, ExpressionSemantics, describe_expression

if typing.TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from bilt.core._domain_registry import DomainRegistry
    from bilt.symbolic._tree import Tree


class BiltCompactIncidenceError(ValueError):
    """A compact structural description cannot be built or safely expanded."""


@dataclasses.dataclass(frozen=True, slots=True)
class UnknownCoordinate:
    """One coordinate of an unknown tensor or scalar."""

    name: str
    indices: tuple[int, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class ResidualCoordinate:
    """One scalar residual, preserving its original equation identity."""

    equation: int
    indices: tuple[int, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class DependencyPath:
    """A sequence of coordinate maps from a residual to an unknown.

    A dense contraction is represented by a handful of relations regardless
    of tensor sizes. Querying an output coordinate follows this path lazily.
    Different paths to the same unknown are retained; graph conversion
    deduplicates their resulting coordinate edges.
    """

    unknown: str
    relations: tuple[CoordinateRelation, ...]

    def inputs(self, output_indices: tuple[int, ...], bindings: DomainRegistry) -> Iterator[UnknownCoordinate]:
        """Yield coordinates of this unknown that may affect one residual."""

        def follow(level: int, indices: tuple[int, ...]) -> Iterator[UnknownCoordinate]:
            if level == len(self.relations):
                yield UnknownCoordinate(self.unknown, indices)
                return
            for child_indices in self.relations[level].inputs(indices, bindings):
                yield from follow(level + 1, child_indices)

        yield from follow(0, output_indices)


@dataclasses.dataclass(frozen=True, slots=True)
class CompactResidual:
    """One original equality, its scalar residual domain and unknown paths."""

    equation: int
    domain: Domain
    paths: tuple[DependencyPath, ...]

    def dependencies(self, output_indices: tuple[int, ...], bindings: DomainRegistry) -> frozenset[UnknownCoordinate]:
        """Return the distinct unknown coordinates affecting one residual."""
        return frozenset(dependency for path in self.paths for dependency in path.inputs(output_indices, bindings))


@dataclasses.dataclass(frozen=True, slots=True)
class CompactIncidence:
    """Potential bipartite incidence without materialised scalar vertices."""

    unknown_domains: tuple[tuple[str, Domain], ...]
    residuals: tuple[CompactResidual, ...]

    @property
    def pattern_count(self) -> int:
        """Number of symbolic unknown dependency paths, independent of axis sizes."""
        return sum(len(residual.paths) for residual in self.residuals)

    def materialize(
        self,
        bindings: DomainRegistry,
        *,
        max_nodes: int = 4096,
        max_edges: int = 100_000,
    ) -> tuple[rx.PyGraph, frozenset[int]]:
        """Expand to an ordinary bipartite graph when safely bounded.

        The output is compatible with existing rustworkx matching/DM code.
        This is a deliberate fallback, not how large models should be
        analysed. Neither graph size nor numerical rank is inferred from
        the number of compact patterns.
        """
        if max_nodes < 1 or max_edges < 1:
            message = "Structural graph limits must be positive."
            raise BiltCompactIncidenceError(message)
        count = sum(_size(domain, bindings) for _, domain in self.unknown_domains)
        count += sum(_size(residual.domain, bindings) for residual in self.residuals)
        if count > max_nodes:
            message = f"Expanding structural incidence requires {count} nodes (limit {max_nodes})."
            raise BiltCompactIncidenceError(message)

        graph = rx.PyGraph(multigraph=False)
        unknown_nodes = {
            UnknownCoordinate(name, indices): graph.add_node(UnknownCoordinate(name, indices))
            for name, domain in self.unknown_domains
            for indices in _coordinates(domain, bindings)
        }
        residual_nodes = {
            ResidualCoordinate(residual.equation, indices): graph.add_node(
                ResidualCoordinate(residual.equation, indices)
            )
            for residual in self.residuals
            for indices in _coordinates(residual.domain, bindings)
        }
        for residual in self.residuals:
            for indices in _coordinates(residual.domain, bindings):
                row = ResidualCoordinate(residual.equation, indices)
                for unknown in residual.dependencies(indices, bindings):
                    graph.add_edge(residual_nodes[row], unknown_nodes[unknown], None)
                    if graph.num_edges() > max_edges:
                        message = f"Expanding structural incidence exceeds the {max_edges} edge limit."
                        raise BiltCompactIncidenceError(message)
        return graph, frozenset(unknown_nodes.values())


def build_compact_incidence(
    equations: Sequence[Tree],
    unknown_domains: Mapping[str, Domain],
    input_domains: Mapping[str, Domain],
) -> CompactIncidence:
    """Describe equalities using exact named domains, without numeric extents.

    All unknown and input domains must already be resolved. Domain inference,
    structural matching and solver selection remain separate responsibilities.
    """
    if set(unknown_domains) & set(input_domains):
        message = "An attribute cannot be both an unknown and a supplied input."
        raise BiltCompactIncidenceError(message)
    domains = dict(input_domains) | dict(unknown_domains)
    residuals = []
    for index, equation in enumerate(equations):
        if not equation.is_equality:
            message = "Compact structural incidence requires binary equality constraints."
            raise BiltCompactIncidenceError(message)
        left = describe_expression(equation.left, domains)
        right = describe_expression(equation.right, domains)
        result_domain = union(left.domain, right.domain)
        paths = tuple(
            itertools.chain(
                _collect_paths(left, result_domain, unknown_domains),
                _collect_paths(right, result_domain, unknown_domains),
            )
        )
        residuals.append(CompactResidual(index, result_domain, paths))
    return CompactIncidence(tuple(unknown_domains.items()), tuple(residuals))


def _collect_paths(
    root: ExpressionSemantics,
    residual_domain: Domain,
    unknown_domains: Mapping[str, Domain],
) -> Iterator[DependencyPath]:
    """Trace only the symbolic expression tree, never tensor coordinates."""

    def walk(node: ExpressionSemantics, relations: tuple[CoordinateRelation, ...]) -> Iterator[DependencyPath]:
        if node.tree.is_leaf:
            if node.tree.head.atom_type is not SymbolType:
                return
            name = typing.cast("str", node.tree.head.value)
            if name in unknown_domains and node.domain == unknown_domains[name]:
                yield DependencyPath(name, relations)
            return
        for child, relation in zip(node.children, node.operand_relations, strict=True):
            if relation is not None:
                yield from walk(child, (*relations, relation))

    yield from walk(root, (CoordinateRelation(root.domain, residual_domain),))


def _size(domain: Domain, bindings: DomainRegistry) -> int:
    return math.prod(bindings.size(dimension) for dimension in domain.dims)


def _coordinates(domain: Domain, bindings: DomainRegistry) -> Iterator[tuple[int, ...]]:
    return itertools.product(*(range(bindings.size(dimension)) for dimension in domain.dims))
