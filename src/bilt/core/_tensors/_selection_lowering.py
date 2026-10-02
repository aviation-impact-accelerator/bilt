__all__ = ["resolve_static_selections"]

import typing

from bilt.core._errors import BiltDomainBindingError
from bilt.core._evaluation_context import current_evaluation
from bilt.core._tensors._errors import BiltSymbolicLoweringError
from bilt.symbolic._atoms import (
    ISEL,
    SEL,
    CoordinateAxisType,
    CoordinateType,
)
from bilt.symbolic._tree import Tree, int_tree

if typing.TYPE_CHECKING:
    from bilt.symbolic._dimensions import Dim


def resolve_static_selections(tree: Tree) -> Tree:
    """Resolve coordinate selections whose coordinate axes are known."""
    if tree.is_leaf:
        return tree

    children = tuple(resolve_static_selections(child) for child in tree.args)

    if tree.head is not SEL:
        return Tree(tree.head, children)

    value_tree, axis_tree, coordinate_axis_tree, selector_tree = children

    if not coordinate_axis_tree.is_leaf or coordinate_axis_tree.head.atom_type is not CoordinateAxisType:
        message = "Lowered sel requires a coordinate axis."
        raise BiltSymbolicLoweringError(message)

    if not selector_tree.is_leaf or selector_tree.head.atom_type is not CoordinateType:
        return Tree(tree.head, children)

    dim = typing.cast("Dim", coordinate_axis_tree.head.value)
    domains = current_evaluation().domains

    try:
        domains.axis(dim)
    except BiltDomainBindingError:
        return Tree(tree.head, children)

    position = domains.position(
        dim,
        selector_tree.head.value,
    )

    return Tree(
        ISEL,
        (
            value_tree,
            axis_tree,
            int_tree(position),
        ),
    )
