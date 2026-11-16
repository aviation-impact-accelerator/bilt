"""Pure named-domain and coordinate-dependency semantics of symbolic operators.

These rules describe possible structural dependencies; they do not decide
linearity, evaluate coefficients, or construct a numerical or matching graph.
Relations can be queried lazily for specific output coordinates.
"""

__all__ = [
    "AxisSelector",
    "BiltOperatorSemanticsError",
    "CoordinateRelation",
    "ExpressionSemantics",
    "OperatorSemantics",
    "SymbolCoordinate",
    "describe_expression",
    "describe_operator",
]

import dataclasses
import itertools
import typing

from bilt.symbolic._atoms import (
    ADD,
    CONDITIONAL,
    INV,
    ISEL,
    MAX,
    MEAN,
    MIN,
    MUL,
    NEG,
    POW,
    PROD,
    SEL,
    SUM,
    CoordinateType,
    DimensionType,
    FloatType,
    IntType,
    QuantityType,
    SymbolType,
)
from bilt.symbolic._domains import Domain, union

if typing.TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from bilt.core._domain_registry import DomainRegistry
    from bilt.symbolic._dimensions import Dim
    from bilt.symbolic._tree import Tree


_CONDITIONAL_ARITY = 3
_BINARY_ARITY = 2


class BiltOperatorSemanticsError(ValueError):
    """An operator's domain or coordinate mapping cannot be described."""


@dataclasses.dataclass(frozen=True, slots=True)
class AxisSelector:
    """A literal position or coordinate label for one named axis."""

    dimension: Dim
    kind: typing.Literal["position", "label"]
    value: object

    def position(self, bindings: DomainRegistry) -> int:
        """Resolve an axis selection without evaluating tensor values."""
        if self.kind == "label":
            return bindings.position(self.dimension, self.value)
        if not isinstance(self.value, int) or isinstance(self.value, bool):
            message = "A positional selector must be a literal integer."
            raise BiltOperatorSemanticsError(message)
        length = bindings.size(self.dimension)
        if not -length <= self.value < length:
            message = f"Index {self.value} is outside dimension {self.dimension!s} of size {length}."
            raise BiltOperatorSemanticsError(message)
        return self.value % length


@dataclasses.dataclass(frozen=True, slots=True)
class CoordinateRelation:
    """Map one output coordinate to possible coordinates of one operand.

    Unchanged axes retain their positions; a reduction ranges over an axis;
    selection fixes one axis position. A scalar operand has no axes and can
    therefore contribute to every coordinate of a broadcast result.
    """

    operand_domain: Domain
    output_domain: Domain
    reduced_axes: tuple[Dim, ...] = ()
    selector: AxisSelector | None = None

    def inputs(
        self,
        output_indices: tuple[int, ...],
        bindings: DomainRegistry,
    ) -> Iterator[tuple[int, ...]]:
        """Generate contributing input coordinates, without materialising edges."""
        if len(output_indices) != len(self.output_domain.dims):
            message = "Output coordinate rank does not match the expression domain."
            raise BiltOperatorSemanticsError(message)
        positions = dict(zip(self.output_domain.dims, output_indices, strict=True))
        for dim, position in positions.items():
            if not 0 <= position < bindings.size(dim):
                message = f"Output position {position} is outside dimension {dim!s}."
                raise BiltOperatorSemanticsError(message)

        axes: list[tuple[int, ...] | range] = []
        for dim in self.operand_domain.dims:
            if self.selector is not None and dim == self.selector.dimension:
                selected = self.selector.position(bindings)
                axes.append((selected,))
            elif dim in positions:
                axes.append((positions[dim],))
            elif dim in self.reduced_axes:
                axes.append(range(bindings.size(dim)))
            else:
                message = f"No coordinate relation for operand dimension {dim!s}."
                raise BiltOperatorSemanticsError(message)
        yield from itertools.product(*axes)


@dataclasses.dataclass(frozen=True, slots=True)
class OperatorSemantics:
    """One operator's output domain and relations for its syntactic operands.

    Metadata operands (named dimensions and literal selectors) have no
    coordinate dependency relation and are represented by None.
    """

    output_domain: Domain
    operand_relations: tuple[CoordinateRelation | None, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class SymbolCoordinate:
    """A symbolic leaf and one of the coordinates that can affect the result."""

    name: str
    indices: tuple[int, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class ExpressionSemantics:
    """An immutable expression tree decorated with operator semantics."""

    tree: Tree
    domain: Domain
    children: tuple[ExpressionSemantics, ...] = ()
    operand_relations: tuple[CoordinateRelation | None, ...] = ()

    def dependencies(
        self,
        output_indices: tuple[int, ...],
        bindings: DomainRegistry,
    ) -> Iterator[SymbolCoordinate]:
        """Lazily trace structural dependencies through nested operations."""
        if self.tree.is_leaf:
            if self.tree.head.atom_type is SymbolType:
                yield SymbolCoordinate(typing.cast("str", self.tree.head.value), output_indices)
            return
        for child, relation in zip(self.children, self.operand_relations, strict=True):
            if relation is None:
                continue
            for indices in relation.inputs(output_indices, bindings):
                yield from child.dependencies(indices, bindings)


def describe_expression(
    tree: Tree,
    symbol_domains: Mapping[str, Domain],
) -> ExpressionSemantics:
    """Describe an expression using only symbolic operators and supplied domains.

    Axis extents are not required until dependencies are enumerated. This
    permits an analyser to retain a compact relation over very large tensors.
    """
    if tree.is_leaf:
        if tree.head.atom_type is SymbolType:
            name = typing.cast("str", tree.head.value)
            try:
                layout = symbol_domains[name]
            except KeyError:
                message = f"No domain supplied for symbol {name!r}."
                raise BiltOperatorSemanticsError(message) from None
            return ExpressionSemantics(tree, layout)
        if tree.head.atom_type in (IntType, FloatType, QuantityType, CoordinateType, DimensionType):
            return ExpressionSemantics(tree, Domain.scalar())
        message = f"Unsupported symbolic leaf {tree.head!s}."
        raise BiltOperatorSemanticsError(message)
    children = tuple(describe_expression(child, symbol_domains) for child in tree.args)
    described = describe_operator(tree, tuple(child.domain for child in children))
    return ExpressionSemantics(tree, described.output_domain, children, described.operand_relations)


def describe_operator(
    tree: Tree,
    operand_domains: tuple[Domain, ...],
) -> OperatorSemantics:
    """Return one shared domain and dependency rule for a symbolic operation."""
    if len(tree.args) != len(operand_domains):
        message = "An operator needs one domain for each operand."
        raise BiltOperatorSemanticsError(message)

    if tree.head in (ADD, MUL, POW, NEG, INV):
        return _arithmetic_operator(tree, operand_domains)

    if tree.head in (SUM, MEAN, MIN, MAX, PROD, ISEL, SEL):
        return _axis_operator(tree, operand_domains)

    if tree.head is CONDITIONAL:
        if len(operand_domains) != _CONDITIONAL_ARITY or any(not item.is_scalar for item in operand_domains):
            message = "Conditional predicates and branches must have scalar domains."
            raise BiltOperatorSemanticsError(message)
        result = Domain.scalar()
        return OperatorSemantics(result, tuple(CoordinateRelation(item, result) for item in operand_domains))

    if tree.is_relation:
        if len(operand_domains) != _BINARY_ARITY:
            message = "A comparison requires two operands."
            raise BiltOperatorSemanticsError(message)
        result = union(*operand_domains)
        return OperatorSemantics(result, tuple(CoordinateRelation(item, result) for item in operand_domains))

    message = f"Unsupported symbolic operator semantics for {tree.head!s}."
    raise BiltOperatorSemanticsError(message)


def _arithmetic_operator(tree: Tree, operand_domains: tuple[Domain, ...]) -> OperatorSemantics:
    if tree.head in (NEG, INV) and len(operand_domains) != 1:
        message = f"{tree.head!s} must have one operand."
        raise BiltOperatorSemanticsError(message)
    if tree.head is POW and len(operand_domains) != _BINARY_ARITY:
        message = "Power must have two operands."
        raise BiltOperatorSemanticsError(message)
    if tree.head in (ADD, MUL) and not operand_domains:
        message = f"{tree.head!s} needs at least one operand."
        raise BiltOperatorSemanticsError(message)
    result = union(*operand_domains)
    return OperatorSemantics(result, tuple(CoordinateRelation(item, result) for item in operand_domains))


def _axis_operator(tree: Tree, domains: tuple[Domain, ...]) -> OperatorSemantics:
    count = 3 if tree.head in (ISEL, SEL) else 2
    if len(domains) != count or len(tree.args) != count:
        message = f"Operator {tree.head!s} requires {count} operands."
        raise BiltOperatorSemanticsError(message)

    axis_tree = tree.args[1]
    if not axis_tree.is_leaf or axis_tree.head.atom_type is not DimensionType:
        message = f"Operator {tree.head!s} requires a literal named dimension."
        raise BiltOperatorSemanticsError(message)
    dimension = typing.cast("Dim", axis_tree.head.value)
    source = domains[0]
    if dimension not in source.dims:
        message = f"Cannot apply {tree.head!s} to absent dimension {dimension!s}."
        raise BiltOperatorSemanticsError(message)
    output = Domain(tuple(dim for dim in source.dims if dim != dimension))

    if tree.head in (ISEL, SEL):
        selector = tree.args[2]
        atom_type = IntType if tree.head is ISEL else CoordinateType
        if not selector.is_leaf or selector.head.atom_type is not atom_type:
            message = f"{tree.head!s} requires a literal selector."
            raise BiltOperatorSemanticsError(message)
        selection = AxisSelector(dimension, "position" if tree.head is ISEL else "label", selector.head.value)
        relation = CoordinateRelation(source, output, selector=selection)
    else:
        relation = CoordinateRelation(source, output, reduced_axes=(dimension,))
    return OperatorSemantics(output, (relation, *(None for _ in range(count - 1))))
