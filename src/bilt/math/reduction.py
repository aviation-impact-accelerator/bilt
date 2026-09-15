from bilt.symbolic._dimensions import Dim
from bilt.symbolic._expressions import Expr
from bilt.symbolic._tree import mean_tree

__all__ = ["mean"]


def mean(value: Expr, dim: Dim) -> Expr:
    """Return a symbolic mean over a dimension."""
    return Expr(mean_tree(value.tree, dim))