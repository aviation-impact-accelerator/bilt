import typing

from bilt.symbolic._expressions import Expr
from bilt.symbolic._tree import mean_tree

if typing.TYPE_CHECKING:
    from bilt.core._tensors._tensors import Tensor
    from bilt.symbolic._dimensions import Dim

__all__ = ["mean"]


def mean(
    value: Expr | Tensor,
    dim: Dim,
) -> Expr:
    """Return a symbolic mean over a dimension."""
    expression = typing.cast("Expr", value)
    return Expr(mean_tree(expression.tree, dim))
