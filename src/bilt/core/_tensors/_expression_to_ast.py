import typing

from bilt.core._to_ast import MEAN_INTRINSIC_NAME, tree_to_ast_expr

if typing.TYPE_CHECKING:
    import ast

    from bilt.core._tensors._expression_lowering import LoweredExpression

__all__ = [
    "MEAN_INTRINSIC_NAME",
    "lowered_expression_to_ast",
]


def lowered_expression_to_ast(
    expression: LoweredExpression,
) -> ast.expr:
    """Convert a lowered tensor expression into backend-neutral Python AST."""
    return tree_to_ast_expr(expression.tree)
