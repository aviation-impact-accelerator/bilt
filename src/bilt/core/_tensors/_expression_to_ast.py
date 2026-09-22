import ast
import typing

from bilt.core._to_ast import tree_to_ast_expr
from bilt.symbolic._atoms import MEAN, AxisType

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from bilt.core._tensors._expression_lowering import LoweredExpression
    from bilt.symbolic._tree import Tree

__all__ = [
    "MEAN_INTRINSIC_NAME",
    "lowered_expression_to_ast",
]

MEAN_INTRINSIC_NAME = "__bilt_mean__"


def lowered_expression_to_ast(
    expression: LoweredExpression,
) -> ast.expr:
    """Convert a lowered tensor expression into backend-neutral Python AST."""
    return tree_to_ast_expr(
        expression.tree,
        operation_handlers={
            MEAN: _mean_to_ast,
        },
    )


def _mean_to_ast(
    tree: Tree,
    convert: Callable[[Tree], ast.expr],
) -> ast.expr:
    value_tree, axis_tree = tree.args

    if not axis_tree.is_leaf or axis_tree.head.atom_type is not AxisType:
        message = "Lowered mean requires a positional axis."
        raise ValueError(message)

    axis = typing.cast("int", axis_tree.head.value)

    return ast.Call(
        func=ast.Name(
            id=MEAN_INTRINSIC_NAME,
            ctx=ast.Load(),
        ),
        args=[convert(value_tree)],
        keywords=[
            ast.keyword(
                arg="axis",
                value=ast.Constant(value=axis),
            ),
            ast.keyword(
                arg="keepdims",
                value=ast.Constant(value=True),
            ),
        ],
    )
