__all__ = [
    "LoweredExpression",
    "lower_expression",
    "reduce_dimension",
]

import dataclasses
import typing

from bilt.core._canonical_ordering import canonicalized
from bilt.core._evaluation_context import current_evaluation
from bilt.core._tensors._errors import BiltSymbolicLoweringError, BiltTensorDomainError
from bilt.symbolic._atoms import (
    ADD,
    CONDITIONAL,
    INV,
    ISEL,
    MUL,
    NEG,
    POW,
    SEL,
    AxisType,
    ConcatAxisOpType,
    ConcatDimOpType,
    CoordinateType,
    DimensionType,
    FloatType,
    IndexOpType,
    IntegrateAxisOpType,
    IntegrateDimOpType,
    IntType,
    QuantityType,
    ReductionOpType,
    RelationOpType,
    SelSliceDimOpType,
    SliceAxisOpType,
    SliceDimOpType,
    SymbolType,
)
from bilt.symbolic._domains import Domain, union
from bilt.symbolic._tree import Tree, axis_tree, int_tree

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    from bilt.symbolic._dimensions import Dim
    from bilt.symbolic._expressions import Expr
    from bilt.units import Unit


@dataclasses.dataclass(frozen=True, slots=True)
class LoweredExpression:
    """A tensor expression with named dimensions lowered to positional axes."""

    tree: Tree
    domain: Domain

    def __post_init__(self) -> None:
        if _contains_atom_type(self.tree, DimensionType):
            message = "Lowered expressions cannot contain symbolic dimensions."
            raise BiltSymbolicLoweringError(message)


def lower_expression(
    expression: Expr,
    *,
    symbol_domains: Mapping[str, Domain],
    canonical_order: Domain,
) -> LoweredExpression:
    """Lower a symbolic expression into positional tensor operations."""
    if _contains_atom_type(expression.rep, AxisType):
        message = "Symbolic expressions cannot contain positional axes."
        raise BiltSymbolicLoweringError(message)

    return _lower_tree(
        expression.rep,
        symbol_domains=symbol_domains,
        canonical_order=canonical_order,
    )


def reduce_dimension(
    value_domain: Domain,
    dim: Dim,
    *,
    canonical_order: Domain,
) -> tuple[Domain, int]:
    """Return the output domain and positional axis for a named reduction."""
    if dim not in value_domain.dims:
        message = f"Cannot reduce dimension {dim!s}; it is not present in domain {value_domain!s}."
        raise BiltTensorDomainError(message)

    return _remove_dimension(
        value_domain,
        dim,
        canonical_order=canonical_order,
    )


def _remove_dimension(
    value_domain: Domain,
    dim: Dim,
    *,
    canonical_order: Domain,
) -> tuple[Domain, int]:
    try:
        axis = canonical_order.dims.index(dim)
    except ValueError:
        message = f"Dimension {dim!s} is not present in canonical order {canonical_order!s}."
        raise BiltTensorDomainError(message) from None

    output_domain = Domain(tuple(domain_dim for domain_dim in value_domain.dims if domain_dim != dim))

    return output_domain, axis


def _lower_tree(  # noqa: C901, PLR0911
    tree: Tree,
    *,
    symbol_domains: Mapping[str, Domain],
    canonical_order: Domain,
) -> LoweredExpression:
    if tree.is_leaf:
        return _lower_leaf(tree, symbol_domains)

    if tree.head is CONDITIONAL or tree.head.atom_type is RelationOpType:
        return _lower_predicate_or_conditional(tree, symbol_domains=symbol_domains, canonical_order=canonical_order)

    if tree.head.atom_type is SelSliceDimOpType:
        return _lower_coordinate_slice(tree, symbol_domains=symbol_domains, canonical_order=canonical_order)

    if tree.head.atom_type is SliceDimOpType:
        return _lower_slice(tree, symbol_domains=symbol_domains, canonical_order=canonical_order)

    if tree.head.atom_type is ConcatDimOpType:
        return _lower_concat(tree, symbol_domains=symbol_domains, canonical_order=canonical_order)

    if tree.head.atom_type is IntegrateDimOpType:
        return _lower_integral(tree, symbol_domains=symbol_domains, canonical_order=canonical_order)

    if tree.head.atom_type is ReductionOpType:
        return _lower_reduction(
            tree,
            symbol_domains=symbol_domains,
            canonical_order=canonical_order,
        )

    if tree.head.atom_type is IndexOpType:
        return _lower_index(
            tree,
            symbol_domains=symbol_domains,
            canonical_order=canonical_order,
        )

    if tree.head in (NEG, INV):
        child = _lower_tree(
            tree.args[0],
            symbol_domains=symbol_domains,
            canonical_order=canonical_order,
        )

        return LoweredExpression(
            tree=Tree(tree.head, (child.tree,)),
            domain=child.domain,
        )

    if tree.head in (ADD, MUL, POW):
        children = tuple(
            _lower_tree(
                child,
                symbol_domains=symbol_domains,
                canonical_order=canonical_order,
            )
            for child in tree.args
        )

        output_domain = canonicalized(
            union(*(child.domain for child in children)),
            canonical_order,
        ).domain

        return LoweredExpression(
            tree=Tree(
                tree.head,
                tuple(child.tree for child in children),
            ),
            domain=output_domain,
        )

    message = f"Cannot lower symbolic operation {tree.head!r}."
    raise NotImplementedError(message)


_RELATION_ARITY = 2


def _lower_predicate_or_conditional(
    tree: Tree,
    *,
    symbol_domains: Mapping[str, Domain],
    canonical_order: Domain,
) -> LoweredExpression:
    if tree.head is CONDITIONAL:
        return _lower_conditional(tree, symbol_domains=symbol_domains, canonical_order=canonical_order)
    return _lower_relation(tree, symbol_domains=symbol_domains, canonical_order=canonical_order)


def _lower_relation(
    tree: Tree,
    *,
    symbol_domains: Mapping[str, Domain],
    canonical_order: Domain,
) -> LoweredExpression:
    operands = tuple(
        _lower_tree(child, symbol_domains=symbol_domains, canonical_order=canonical_order) for child in tree.args
    )
    if len(operands) != _RELATION_ARITY or any(not operand.domain.is_scalar for operand in operands):
        message = "Conditional predicates require scalar comparison operands."
        raise BiltTensorDomainError(message)
    return LoweredExpression(Tree(tree.head, tuple(operand.tree for operand in operands)), Domain.scalar())


def _lower_conditional(
    tree: Tree,
    *,
    symbol_domains: Mapping[str, Domain],
    canonical_order: Domain,
) -> LoweredExpression:
    predicate, when_true, when_false = (
        _lower_tree(child, symbol_domains=symbol_domains, canonical_order=canonical_order) for child in tree.args
    )
    if not tree.args[0].is_relation or not predicate.domain.is_scalar:
        message = "Conditional predicates must be scalar relations."
        raise BiltTensorDomainError(message)
    if not when_true.domain.is_scalar or not when_false.domain.is_scalar:
        message = "Conditional branches must both be scalar expressions."
        raise BiltTensorDomainError(message)
    return LoweredExpression(
        Tree(CONDITIONAL, (predicate.tree, when_true.tree, when_false.tree)),
        Domain.scalar(),
    )


def _lower_leaf(
    tree: Tree,
    symbol_domains: Mapping[str, Domain],
) -> LoweredExpression:
    if tree.head.atom_type is SymbolType:
        name = typing.cast("str", tree.head.value)

        try:
            value_domain = symbol_domains[name]
        except KeyError:
            message = f"No domain is known for symbol {name!r}."
            raise BiltTensorDomainError(message) from None

        return LoweredExpression(
            tree=tree,
            domain=value_domain,
        )

    if tree.head.atom_type in (IntType, FloatType, QuantityType, CoordinateType):
        return LoweredExpression(
            tree=tree,
            domain=Domain.scalar(),
        )

    message = f"Cannot lower symbolic leaf {tree.head!r}."
    raise NotImplementedError(message)


_MIN_CONCAT_OPERANDS = 2


def _lower_coordinate_slice(
    tree: Tree,
    *,
    symbol_domains: Mapping[str, Domain],
    canonical_order: Domain,
) -> LoweredExpression:
    """Resolve inclusive coordinate labels, then reuse positional slicing."""
    source, target, first, last, step = typing.cast("tuple[Dim, Dim, object, object, int | None]", tree.head.value)
    selector = current_evaluation().domains.coordinate_slice(source, slice(first, last, step))
    positional = Tree(
        SliceDimOpType((source, target, selector.start, selector.stop, selector.step)),
        tree.args,
    )
    return _lower_slice(positional, symbol_domains=symbol_domains, canonical_order=canonical_order)


def _lower_slice(
    tree: Tree,
    *,
    symbol_domains: Mapping[str, Domain],
    canonical_order: Domain,
) -> LoweredExpression:
    """Lower a named positional slice into static source and target axes."""
    source, target, start, stop, step = typing.cast(
        "tuple[Dim, Dim, int | None, int | None, int | None]", tree.head.value
    )
    value = _lower_tree(tree.args[0], symbol_domains=symbol_domains, canonical_order=canonical_order)
    if source not in value.domain.dims:
        message = f"Cannot slice absent dimension {source} in domain {value.domain}."
        raise BiltTensorDomainError(message)
    if target in value.domain.dims:
        message = f"Slice destination {target} already exists in source domain {value.domain}."
        raise BiltTensorDomainError(message)
    if target not in canonical_order.dims:
        message = f"Slice destination {target} has not been registered."
        raise BiltTensorDomainError(message)

    source_axis = canonical_order.dims.index(source)
    target_axis = canonical_order.dims.index(target)
    output_dims = tuple(target if dim == source else dim for dim in value.domain.dims)
    domain = canonicalized(Domain(output_dims), canonical_order).domain
    return LoweredExpression(
        Tree(SliceAxisOpType((source_axis, target_axis, start, stop, step)), (value.tree,)),
        domain,
    )


def _lower_concat(
    tree: Tree,
    *,
    symbol_domains: Mapping[str, Domain],
    canonical_order: Domain,
) -> LoweredExpression:
    """Lower a multi-input domain transform to a named-axis concatenation."""
    sources, target = typing.cast("tuple[tuple[Dim, ...], Dim]", tree.head.value)
    operands = tuple(
        _lower_tree(child, symbol_domains=symbol_domains, canonical_order=canonical_order) for child in tree.args
    )
    if len(operands) != len(sources) or len(operands) < _MIN_CONCAT_OPERANDS:
        message = "Concatenation requires at least two tensors and a source axis for each."
        raise BiltTensorDomainError(message)

    remaining: tuple[Dim, ...] | None = None
    for operand, source in zip(operands, sources, strict=True):
        dims = operand.domain.dims
        if source not in dims:
            message = f"Concatenation source {source} is absent from operand domain {operand.domain}."
            raise BiltTensorDomainError(message)
        if target in dims:
            message = f"Concatenation target {target} already occurs in operand domain {operand.domain}."
            raise BiltTensorDomainError(message)
        shared = tuple(dim for dim in dims if dim != source)
        if remaining is None:
            remaining = shared
        elif set(remaining) != set(shared):
            message = "All concatenated operands must share the same non-concatenated dimensions."
            raise BiltTensorDomainError(message)

    output_domain = canonicalized(Domain((*typing.cast("tuple[Dim, ...]", remaining), target)), canonical_order).domain
    positions = canonical_order.dims
    source_axes = tuple(positions.index(source) for source in sources)
    target_axis = positions.index(target)
    return LoweredExpression(
        Tree(ConcatAxisOpType((source_axes, target_axis)), tuple(operand.tree for operand in operands)),
        output_domain,
    )


def _lower_integral(
    tree: Tree,
    *,
    symbol_domains: Mapping[str, Domain],
    canonical_order: Domain,
) -> LoweredExpression:
    """Resolve named coordinates to static trapezoidal weights."""
    value_tree, dim_tree = tree.args
    if not dim_tree.is_leaf or dim_tree.head.atom_type is not DimensionType:
        message = "Integration requires a named dimension."
        raise BiltTensorDomainError(message)
    dimension = typing.cast("Dim", dim_tree.head.value)
    value = _lower_tree(value_tree, symbol_domains=symbol_domains, canonical_order=canonical_order)
    domain, axis = reduce_dimension(value.domain, dimension, canonical_order=canonical_order)
    unit = typing.cast("Unit | None", tree.head.value)
    weights = current_evaluation().domains.integration_weights(dimension, coordinate_unit=unit)
    return LoweredExpression(
        tree=Tree(IntegrateAxisOpType((weights, axis)), (value.tree,)),
        domain=domain,
    )


def _lower_reduction(
    tree: Tree,
    *,
    symbol_domains: Mapping[str, Domain],
    canonical_order: Domain,
) -> LoweredExpression:
    value_tree, dim_tree = tree.args

    if not dim_tree.is_leaf or dim_tree.head.atom_type is not DimensionType:
        message = f"{tree.head} reduction requires a symbolic dimension."
        raise BiltTensorDomainError(message)

    dim = typing.cast("Dim", dim_tree.head.value)

    value = _lower_tree(
        value_tree,
        symbol_domains=symbol_domains,
        canonical_order=canonical_order,
    )

    output_domain, axis = reduce_dimension(
        value.domain,
        dim,
        canonical_order=canonical_order,
    )

    return LoweredExpression(
        tree=Tree(
            tree.head,
            (
                value.tree,
                axis_tree(axis),
            ),
        ),
        domain=output_domain,
    )


def _lower_index(
    tree: Tree,
    *,
    symbol_domains: Mapping[str, Domain],
    canonical_order: Domain,
) -> LoweredExpression:
    value_tree, dim_tree, selector_tree = tree.args

    if not dim_tree.is_leaf or dim_tree.head.atom_type is not DimensionType:
        message = f"{tree.head} requires a symbolic dimension."
        raise BiltTensorDomainError(message)

    dim = typing.cast("Dim", dim_tree.head.value)

    value = _lower_tree(
        value_tree,
        symbol_domains=symbol_domains,
        canonical_order=canonical_order,
    )
    selector = _lower_tree(
        selector_tree,
        symbol_domains=symbol_domains,
        canonical_order=canonical_order,
    )

    if not selector.domain.is_scalar:
        message = f"{tree.head} currently requires a scalar selector."
        raise BiltTensorDomainError(message)

    if dim not in value.domain.dims:
        message = f"Cannot index dimension {dim!s}; it is not present in domain {value.domain!s}."
        raise BiltTensorDomainError(message)

    output_domain, axis = _remove_dimension(
        value.domain,
        dim,
        canonical_order=canonical_order,
    )

    operator = tree.head
    lowered_selector = selector.tree

    if operator is SEL:
        if not selector_tree.is_leaf or selector_tree.head.atom_type is not CoordinateType:
            message = "sel requires a literal coordinate selector."
            raise BiltSymbolicLoweringError(message)

        position = current_evaluation().domains.position(
            dim,
            selector_tree.head.value,
        )
        operator = ISEL
        lowered_selector = int_tree(position)

    return LoweredExpression(
        tree=Tree(
            operator,
            (
                value.tree,
                axis_tree(axis),
                lowered_selector,
            ),
        ),
        domain=output_domain,
    )


def _contains_atom_type(tree: Tree, atom_type: object) -> bool:
    stack = [tree]

    while stack:
        node = stack.pop()

        if node.head.atom_type is atom_type:
            return True

        stack.extend(node.args)

    return False
