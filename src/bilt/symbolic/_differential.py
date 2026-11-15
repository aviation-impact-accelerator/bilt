"""Unresolved, context-sensitive derivatives and point conditions.

A deferred derivative is *not* an ordinary JVP. The differential classifier
must establish that its target is an evolving state before it can be solved.
"""

import typing

from bilt.symbolic._atoms import Atom, OpType
from bilt.symbolic._expressions import Expr, TensorExpr, as_tree
from bilt.symbolic._tree import Tree

DEFERRED_GRAD = Atom(OpType, "deferred_grad")
AT = Atom(OpType, "at")


def at[T](value: T, *, wrt: object, where: object) -> T:
    """Symbolically constrain the value of a state at a coordinate."""
    tree = Tree(AT, (as_tree(value), as_tree(wrt), as_tree(where)))
    expression = TensorExpr(tree) if isinstance(value, TensorExpr) else Expr(tree)
    return typing.cast("T", expression)
