__all__ = [
    "LoweredExpression",
    "lower_expression",
    "reduce_dimension",
]

import dataclasses
import typing

from bilt.core._canonical_ordering import canonicalized
from bilt.core._tensors._errors import BiltSymbolicLoweringError, BiltTensorDomainError
from bilt.symbolic._atoms import (
    ADD,
    INV,
    MUL,
    NEG,
    POW,
    SEL,
    AxisType,
    CoordinateType,
    DimensionType,
    FloatType,
    IndexOpType,
    IntType,
    ReductionOpType,
    SymbolType,
)
from bilt.symbolic._domains import Domain, union
from bilt.symbolic._tree import Tree, axis_tree, coordinate_axis_tree

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    from bilt.symbolic._dimensions import Dim
    from bilt.symbolic._expressions import Expr


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


def _lower_tree(
    tree: Tree,
    *,
    symbol_domains: Mapping[str, Domain],
    canonical_order: Domain,
) -> LoweredExpression:
    if tree.is_leaf:
        return _lower_leaf(tree, symbol_domains)

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

    if tree.head.atom_type in (IntType, FloatType, CoordinateType):
        return LoweredExpression(
            tree=tree,
            domain=Domain.scalar(),
        )

    message = f"Cannot lower symbolic leaf {tree.head!r}."
    raise NotImplementedError(message)


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

    if tree.head is SEL:
        lowered_tree = Tree(
            tree.head,
            (
                value.tree,
                axis_tree(axis),
                coordinate_axis_tree(dim),
                selector.tree,
            ),
        )
    else:
        lowered_tree = Tree(
            tree.head,
            (
                value.tree,
                axis_tree(axis),
                selector.tree,
            ),
        )

    return LoweredExpression(
        tree=lowered_tree,
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
