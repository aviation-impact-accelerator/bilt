import ast
import typing

from bilt.symbolic import _atoms as ops
from bilt.symbolic._atoms import AxisType, FloatType, IntType, SymbolType
from bilt.symbolic._expressions import Expr
from bilt.symbolic._isolation._errors import BiltTargetNotFoundError, BiltUnsupportedIsolationError
from bilt.symbolic._isolation._isolate import isolate

if typing.TYPE_CHECKING:
    from bilt.core._equation import BasicEquation
    from bilt.symbolic._isolation._types import IsolatedTree
    from bilt.symbolic._tree import Tree

__all__ = [
    "isolated_tree_to_functiondef",
    "tree_to_ast_expr",
]

MEAN_INTRINSIC_NAME = "__bilt_mean__"


def isolated_tree_to_functiondef(
    isolated: IsolatedTree,
    parameter_order: tuple[str, ...],
) -> ast.FunctionDef:
    rhs_expr = tree_to_ast_expr(isolated.rhs.rep)
    function_name = typing.cast("str", isolated.target.rep.head.value)
    parameter_order = tuple(name for name in parameter_order if name != function_name)

    function_def = ast.FunctionDef(
        name=function_name,
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=name, annotation=None) for name in parameter_order],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
            vararg=None,
            kwarg=None,
        ),
        body=[ast.Return(value=rhs_expr)],
        decorator_list=[],
        returns=None,
        type_comment=None,
    )
    ast.fix_missing_locations(function_def)
    return function_def


def tree_to_ast_expr(tree: Tree) -> ast.expr:  # noqa: PLR0911
    """Convert a symbolic tree to Python AST."""
    if tree.is_leaf:
        return _leaf_to_ast(tree)

    match tree.head:
        case ops.ADD:
            return _fold_binop(tree.args, ast.Add)

        case ops.MUL:
            return _fold_binop(tree.args, ast.Mult)

        case ops.POW:
            return ast.BinOp(
                left=tree_to_ast_expr(tree.args[0]),
                op=ast.Pow(),
                right=tree_to_ast_expr(tree.args[1]),
            )

        case ops.NEG:
            return ast.UnaryOp(
                op=ast.USub(),
                operand=tree_to_ast_expr(tree.args[0]),
            )

        case ops.INV:
            return ast.BinOp(
                left=ast.Constant(value=1),
                op=ast.Div(),
                right=tree_to_ast_expr(tree.args[0]),
            )

        case ops.MEAN:
            return _mean_to_ast(tree)

        case _:
            message = f"Unsupported symbolic head {tree.head!r} in AST conversion."
            raise BiltUnsupportedIsolationError(message)


def _leaf_to_ast(tree: Tree) -> ast.expr:
    if tree.head.atom_type is SymbolType:
        return ast.Name(
            id=typing.cast("str", tree.head.value),
            ctx=ast.Load(),
        )

    if tree.head.atom_type in (IntType, FloatType):
        return ast.Constant(
            value=typing.cast("int | float", tree.head.value),
        )

    message = f"Unsupported symbolic leaf {tree.head!r} in AST conversion."
    raise BiltUnsupportedIsolationError(message)


def _mean_to_ast(tree: Tree) -> ast.expr:
    value_tree, axis_tree = tree.args

    if not axis_tree.is_leaf or axis_tree.head.atom_type is not AxisType:
        message = "Lowered mean requires a positional axis."
        raise BiltUnsupportedIsolationError(message)

    axis = typing.cast("int", axis_tree.head.value)

    return ast.Call(
        func=ast.Name(
            id=MEAN_INTRINSIC_NAME,
            ctx=ast.Load(),
        ),
        args=[tree_to_ast_expr(value_tree)],
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


def _fold_binop(
    children: tuple[Tree, ...],
    operator_type: type[ast.operator],
) -> ast.expr:
    if not children:
        message = "Cannot lower an empty n-ary symbolic operator."
        raise BiltUnsupportedIsolationError(message)

    iterator = iter(children)
    expr = tree_to_ast_expr(next(iterator))

    for child in iterator:
        expr = ast.BinOp(
            left=expr,
            op=operator_type(),
            right=tree_to_ast_expr(child),
        )

    return expr


def equation_to_ast(
    equation: BasicEquation,
    target: str,
) -> ast.FunctionDef:
    if target not in equation.__bilt_equation_symbols__:
        message = f"Target {target!r} is not a parameter of the equation {equation!s}."
        raise BiltTargetNotFoundError(message)

    target_expr = Expr.symbol(target)
    isolated = isolate(
        equation.__bilt_equation_tree__,
        target_expr,
    )

    return isolated_tree_to_functiondef(
        isolated,
        parameter_order=equation.__bilt_parameters__,
    )
