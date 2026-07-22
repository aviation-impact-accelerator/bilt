from __future__ import annotations

import ast
import typing

from bilt.symbolic._atoms import ADD, INV, MUL, NEG, POW, FloatType, IntType, SymbolType
from bilt.symbolic._isolation._errors import UnsupportedIsolationError

if typing.TYPE_CHECKING:
    from bilt.symbolic._isolation._types import IsolatedTree
    from bilt.symbolic._tree import Tree

__all__ = ["isolated_tree_to_functiondef"]


def isolated_tree_to_functiondef(
    isolated: IsolatedTree,
    parameter_order: tuple[str, ...],
) -> ast.FunctionDef:
    rhs_expr = _tree_to_ast_expr(isolated.rhs.rep)
    function_name = isolated.target.rep.head.value
    parameter_order = tuple(name for name in parameter_order if name != function_name)
    if not isinstance(function_name, str):
        message = f"Unsupported symbolic leaf {isolated.target.rep.head!r} in AST conversion."
        raise UnsupportedIsolationError(message)

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


def _tree_to_ast_expr(tree: Tree) -> ast.expr:
    if tree.is_leaf:
        return _leaf_to_ast_expr(tree)

    head = tree.head
    if head is ADD:
        return _fold_binop(tree.args, ast.Add)
    if head is MUL:
        return _fold_binop(tree.args, ast.Mult)
    if head is POW:
        return ast.BinOp(
            left=_tree_to_ast_expr(tree.args[0]),
            op=ast.Pow(),
            right=_tree_to_ast_expr(tree.args[1]),
        )
    if head is NEG:
        return ast.UnaryOp(op=ast.USub(), operand=_tree_to_ast_expr(tree.args[0]))
    if head is INV:
        return ast.BinOp(
            left=ast.Constant(value=1),
            op=ast.Div(),
            right=_tree_to_ast_expr(tree.args[0]),
        )
    message = f"Unsupported symbolic head {head!r} in AST conversion."
    raise UnsupportedIsolationError(message)


def _leaf_to_ast_expr(tree: Tree) -> ast.expr:
    atom = tree.head
    atom_type = atom.atom_type
    value = atom.value

    if atom_type is SymbolType:
        return ast.Name(id=typing.cast("str", value), ctx=ast.Load())
    if atom_type in (IntType, FloatType):
        return ast.Constant(value=typing.cast("int | float", value))
    message = f"Unsupported symbolic leaf {atom!r} in AST conversion."
    raise UnsupportedIsolationError(message)


def _fold_binop(children: tuple[Tree, ...], operator_type: type[ast.operator]) -> ast.expr:
    if not children:
        message = "Cannot lower an empty n-ary symbolic operator."
        raise UnsupportedIsolationError(message)

    iterator = iter(children)
    expr = _tree_to_ast_expr(next(iterator))
    for child in iterator:
        expr = ast.BinOp(left=expr, op=operator_type(), right=_tree_to_ast_expr(child))
    return expr
