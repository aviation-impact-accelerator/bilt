import ast
import typing

from bilt.symbolic import _atoms as ops
from bilt.symbolic._atoms import Atom, FloatType, IntType, SymbolType
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


def tree_to_ast_expr(
    tree: Tree,
    *,
    operation_handlers: typing.Mapping[
        Atom,
        typing.Callable[
            [Tree, typing.Callable[[Tree], ast.expr]],
            ast.expr,
        ],
    ]
    | None = None,
) -> ast.expr:
    """Convert a symbolic tree to Python AST."""
    handlers = operation_handlers or {}

    def convert(node: Tree) -> ast.expr:
        if node.is_leaf:
            return _leaf_to_ast(node)

        built_in = _built_in_operation_to_ast(node, convert)
        if built_in is not None:
            return built_in

        handler = handlers.get(node.head)
        if handler is not None:
            return handler(node, convert)

        message = f"Unsupported symbolic head {node.head!r} in AST conversion."
        raise BiltUnsupportedIsolationError(message)

    return convert(tree)


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


def _built_in_operation_to_ast(
    tree: Tree,
    convert: typing.Callable[[Tree], ast.expr],
) -> ast.expr | None:
    match tree.head:
        case ops.ADD:
            return _fold_binop(tree.args, ast.Add, convert)

        case ops.MUL:
            return _fold_binop(tree.args, ast.Mult, convert)

        case ops.POW:
            return ast.BinOp(
                left=convert(tree.args[0]),
                op=ast.Pow(),
                right=convert(tree.args[1]),
            )

        case ops.NEG:
            return ast.UnaryOp(
                op=ast.USub(),
                operand=convert(tree.args[0]),
            )

        case ops.INV:
            return ast.BinOp(
                left=ast.Constant(value=1),
                op=ast.Div(),
                right=convert(tree.args[0]),
            )

        case _:
            return None


def _fold_binop(
    children: tuple[Tree, ...],
    operator_type: type[ast.operator],
    convert: typing.Callable[[Tree], ast.expr],
) -> ast.expr:
    if not children:
        message = "Cannot lower an empty n-ary symbolic operator."
        raise BiltUnsupportedIsolationError(message)

    iterator = iter(children)
    expr = convert(next(iterator))

    for child in iterator:
        expr = ast.BinOp(
            left=expr,
            op=operator_type(),
            right=convert(child),
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
