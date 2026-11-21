"""Scalar-coordinate incidence for supported symbolic linear tensor equations.

This refines the existing port-level incidence without changing causalisation.
It does not evaluate model coefficients, choose equation directions, or imply
numerical nonsingularity. A later planner can apply the existing matching and
Dulmage-Mendelsohn algorithms to the returned graph.
"""

__all__ = [
    "CoordinateIncidence",
    "ResidualCoordinate",
    "UnknownCoordinate",
    "build_coordinate_incidence",
]

import dataclasses
import itertools
import math
import typing

import rustworkx as rx

from bilt.core._structural_analysis._linear_block import BiltLinearBlockError
from bilt.core._structural_analysis._linear_operator import _validate_linear
from bilt.symbolic._atoms import ADD, ISEL, MEAN, MUL, NEG, SEL, SUM, DimensionType, IntType, SymbolType
from bilt.symbolic._domains import Domain, union

if typing.TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from bilt.core._domain_registry import DomainRegistry
    from bilt.symbolic._dimensions import Dim
    from bilt.symbolic._tree import Tree


@dataclasses.dataclass(frozen=True, slots=True)
class UnknownCoordinate:
    """One scalar coordinate of an unknown, in its declared domain order."""

    name: str
    indices: tuple[int, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class ResidualCoordinate:
    """One scalar equation row, identified by equation position and coordinates."""

    equation: int
    indices: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class CoordinateIncidence:
    """A bounded, symbolic equation-coordinate to unknown-coordinate incidence.

    Each edge is a *possible* nonzero coefficient, not a numerical rank claim.
    Residuals remain separate even when two equations have identical expressions.
    """

    unknowns: tuple[UnknownCoordinate, ...]
    residuals: tuple[ResidualCoordinate, ...]
    residual_domains: tuple[Domain, ...]
    edges: frozenset[tuple[ResidualCoordinate, UnknownCoordinate]]

    def to_graph(self) -> tuple[rx.PyGraph, frozenset[int]]:
        """Create a bipartite graph compatible with the existing DM analyser."""
        graph = rx.PyGraph(multigraph=False)
        unknown_nodes = {coordinate: graph.add_node(coordinate) for coordinate in self.unknowns}
        residual_nodes = {coordinate: graph.add_node(coordinate) for coordinate in self.residuals}
        for residual, unknown in sorted(
            self.edges,
            key=lambda pair: (residual_nodes[pair[0]], unknown_nodes[pair[1]]),
        ):
            graph.add_edge(residual_nodes[residual], unknown_nodes[unknown], None)
        return graph, frozenset(unknown_nodes.values())


def build_coordinate_incidence(  # noqa: C901, PLR0913 - explicit constraints and safety limits for structural tracing.
    equations: Sequence[Tree],
    unknown_domains: Mapping[str, Domain],
    input_domains: Mapping[str, Domain],
    bindings: DomainRegistry,
    *,
    max_coordinates: int = 4096,
    max_edges: int = 100_000,
) -> CoordinateIncidence:
    """Trace supported linear expressions to scalar-coordinate dependencies.

    All unknown domains must already be resolved (or explicitly hinted). This
    is a standalone structural refinement, not a replacement for domain
    inference. The registry provides extents and coordinate labels already
    bound from request inputs; no numerical values or basis matrices are used.

    Coordinate limits prevent accidental expansion of very large tensors.
    Unsupported operators and missing dimension bindings fail explicitly.
    """
    if max_coordinates < 1 or max_edges < 1:
        message = "Coordinate incidence limits must be positive."
        raise ValueError(message)
    if set(unknown_domains) & set(input_domains):
        message = "An attribute cannot be both an unknown and a supplied input."
        raise BiltLinearBlockError(message)
    if not unknown_domains or not equations:
        message = "Coordinate incidence requires equations and unknowns."
        raise BiltLinearBlockError(message)

    symbols = frozenset(unknown_domains)
    domains = dict(input_domains) | dict(unknown_domains)
    tracer = _CoordinateTracer(domains, symbols, bindings)
    residual_domains = []
    for equation in equations:
        if not equation.is_equality:
            message = "Coordinate incidence supports binary equality constraints only."
            raise BiltLinearBlockError(message)
        _validate_linear(equation.left, symbols)
        _validate_linear(equation.right, symbols)
        residual_domains.append(union(tracer.domain(equation.left), tracer.domain(equation.right)))

    count = sum(_extent(domain, bindings) for domain in (*unknown_domains.values(), *residual_domains))
    if count > max_coordinates:
        message = f"Coordinate incidence exceeds the {max_coordinates} coordinate limit ({count} required)."
        raise BiltLinearBlockError(message)

    unknowns = tuple(
        UnknownCoordinate(name, indices)
        for name, domain in unknown_domains.items()
        for indices in _coordinates(domain, bindings)
    )
    residuals = tuple(
        ResidualCoordinate(row, indices)
        for row, domain in enumerate(residual_domains)
        for indices in _coordinates(domain, bindings)
    )

    edges: set[tuple[ResidualCoordinate, UnknownCoordinate]] = set()
    for residual in residuals:
        expression = equations[residual.equation]
        positions = dict(zip(residual_domains[residual.equation].dims, residual.indices, strict=True))
        for side in (expression.left, expression.right):
            for coordinate in tracer.dependencies(side, positions):
                edges.add((residual, coordinate))
                if len(edges) > max_edges:
                    message = f"Coordinate incidence exceeds the {max_edges} edge limit."
                    raise BiltLinearBlockError(message)

    return CoordinateIncidence(unknowns, residuals, tuple(residual_domains), frozenset(edges))


def _extent(domain: Domain, bindings: DomainRegistry) -> int:
    return math.prod(bindings.size(dim) for dim in domain.dims)


def _coordinates(domain: Domain, bindings: DomainRegistry) -> Iterator[tuple[int, ...]]:
    return itertools.product(*(range(bindings.size(dim)) for dim in domain.dims))


@dataclasses.dataclass
class _CoordinateTracer:
    """Infer expression domains and trace unknown coordinates backwards."""

    domains: Mapping[str, Domain]
    unknowns: frozenset[str]
    bindings: DomainRegistry
    _domains: dict[Tree, Domain] = dataclasses.field(default_factory=dict)

    def domain(self, tree: Tree) -> Domain:
        if tree not in self._domains:
            self._domains[tree] = self._domain(tree)
        return self._domains[tree]

    def _domain(self, tree: Tree) -> Domain:
        if tree.is_leaf:
            if tree.head.atom_type is SymbolType:
                name = typing.cast("str", tree.head.value)
                try:
                    return self.domains[name]
                except KeyError:
                    message = f"No domain supplied for symbol {name!r} in coordinate incidence."
                    raise BiltLinearBlockError(message) from None
            return Domain.scalar()
        if tree.head in (SUM, MEAN, ISEL, SEL):
            source = self.domain(tree.args[0])
            dimension = tree.args[1]
            if dimension.head.atom_type is not DimensionType or dimension.head.value not in source.dims:
                message = f"Operator {tree.head} requires an axis in its input domain."
                raise BiltLinearBlockError(message)
            return Domain(tuple(dim for dim in source.dims if dim != dimension.head.value))
        return union(*(self.domain(arg) for arg in tree.args))

    def dependencies(  # noqa: C901, PLR0911, PLR0912 - explicit operator dispatch makes coordinate semantics auditable.
        self,
        tree: Tree,
        positions: Mapping[Dim, int],
    ) -> set[UnknownCoordinate]:
        """Collect possible unknown coordinates contributing to one result."""
        if tree.is_leaf:
            if tree.head.atom_type is not SymbolType:
                return set()
            name = typing.cast("str", tree.head.value)
            if name not in self.unknowns:
                return set()
            try:
                indices = tuple(positions[dim] for dim in self.domains[name].dims)
            except KeyError as error:
                message = f"Unknown coordinate {name!r} is missing an axis position: {error}."
                raise BiltLinearBlockError(message) from error
            return {UnknownCoordinate(name, indices)}

        if tree.head in (SUM, MEAN):
            dim = typing.cast("Dim", tree.args[1].head.value)
            result: set[UnknownCoordinate] = set()
            for index in range(self.bindings.size(dim)):
                result.update(self.dependencies(tree.args[0], dict(positions) | {dim: index}))
            return result

        if tree.head in (ISEL, SEL):
            dim = typing.cast("Dim", tree.args[1].head.value)
            selector = tree.args[2].head
            if tree.head is ISEL:
                if selector.atom_type is not IntType:
                    message = "Linear index selection needs a literal integer."
                    raise BiltLinearBlockError(message)
                size = self.bindings.size(dim)
                index = typing.cast("int", selector.value)
                if not -size <= index < size:
                    message = f"Index {index} is outside dimension {dim!s} of size {size}."
                    raise BiltLinearBlockError(message)
                index %= size
            else:
                index = self.bindings.position(dim, selector.value)
            return self.dependencies(tree.args[0], dict(positions) | {dim: index})

        if tree.head in (ADD, NEG, MUL):
            result = set()
            for child in tree.args:
                result.update(self.dependencies(child, positions))
            return result

        if any(self._contains_unknown(child) for child in tree.args):
            message = f"Unsupported unknown-dependent coordinate operator: {tree.head}."
            raise BiltLinearBlockError(message)
        return set()

    def _contains_unknown(self, tree: Tree) -> bool:
        if tree.is_leaf:
            return tree.head.atom_type is SymbolType and tree.head.value in self.unknowns
        return any(self._contains_unknown(child) for child in tree.args)
