import typing

from bilt.symbolic._expressions import Expr
from bilt.symbolic._tree import mean_tree

if typing.TYPE_CHECKING:
    from bilt.symbolic._dimensions import Dim

__all__ = ["mean"]


def mean(value: Expr, dim: Dim) -> Expr:
    """Return a symbolic mean over a dimension."""
    return Expr(mean_tree(value.tree, dim))
