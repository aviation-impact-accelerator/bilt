import ast
import typing

from bilt.symbolic import _atoms as ops
from bilt.symbolic._atoms import AxisType, FloatType, IntType, SymbolType
from bilt.symbolic._isolation._errors import BiltUnsupportedIsolationError

if typing.TYPE_CHECKING:
    from bilt.symbolic._tree import Tree

__all__ = [
    "reduction_intrinsic_ast",
    "tree_to_functiondef",
]

MEAN_INTRINSIC_NAME = "__bilt_mean__"
SUM_INTRINSIC_NAME = "__bilt_sum__"
MIN_INTRINSIC_NAME = "__bilt_min__"
MAX_INTRINSIC_NAME = "__bilt_max__"
PROD_INTRINSIC_NAME = "__bilt_prod__"


def tree_to_functiondef(
    tree: Tree,
    *,
    name: str,
    parameter_order: tuple[str, ...],
) -> ast.FunctionDef:
    """Convert a symbolic tree into a Python function returning its value."""
    parameter_order = tuple(parameter for parameter in parameter_order if parameter != name)

    function_def = ast.FunctionDef(
        name=name,
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=parameter, annotation=None) for parameter in parameter_order],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
            vararg=None,
            kwarg=None,
        ),
        body=[ast.Return(value=_tree_to_ast_expr(tree))],
        decorator_list=[],
        returns=None,
        type_comment=None,
    )
    ast.fix_missing_locations(function_def)
    return function_def


def _tree_to_ast_expr(tree: Tree) -> ast.expr:  # noqa: C901, PLR0911
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
                left=_tree_to_ast_expr(tree.args[0]),
                op=ast.Pow(),
                right=_tree_to_ast_expr(tree.args[1]),
            )

        case ops.NEG:
            return ast.UnaryOp(
                op=ast.USub(),
                operand=_tree_to_ast_expr(tree.args[0]),
            )

        case ops.INV:
            return ast.BinOp(
                left=ast.Constant(value=1),
                op=ast.Div(),
                right=_tree_to_ast_expr(tree.args[0]),
            )

        case ops.MEAN:
            return _reduction_to_ast(tree, MEAN_INTRINSIC_NAME)

        case ops.SUM:
            return _reduction_to_ast(tree, SUM_INTRINSIC_NAME)

        case ops.MIN:
            return _reduction_to_ast(tree, MIN_INTRINSIC_NAME)

        case ops.MAX:
            return _reduction_to_ast(tree, MAX_INTRINSIC_NAME)

        case ops.PROD:
            return _reduction_to_ast(tree, PROD_INTRINSIC_NAME)

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


def _reduction_to_ast(
    tree: Tree,
    intrinsic_name: str,
) -> ast.expr:
    value_tree, axis_tree = tree.args

    if not axis_tree.is_leaf or axis_tree.head.atom_type is not AxisType:
        message = "Lowered reduction requires a positional axis."
        raise BiltUnsupportedIsolationError(message)

    axis = typing.cast("int", axis_tree.head.value)

    return reduction_intrinsic_ast(
        intrinsic_name,
        _tree_to_ast_expr(value_tree),
        axis,
    )


def _fold_binop(
    children: tuple[Tree, ...],
    operator_type: type[ast.operator],
) -> ast.expr:
    if not children:
        message = "Cannot lower an empty n-ary symbolic operator."
        raise BiltUnsupportedIsolationError(message)

    iterator = iter(children)
    expr = _tree_to_ast_expr(next(iterator))

    for child in iterator:
        expr = ast.BinOp(
            left=expr,
            op=operator_type(),
            right=_tree_to_ast_expr(child),
        )

    return expr


def reduction_intrinsic_ast(
    intrinsic_name: str,
    value: ast.expr,
    axis: int,
) -> ast.Call:
    """Construct a backend reduction intrinsic call."""
    return ast.Call(
        func=ast.Name(
            id=intrinsic_name,
            ctx=ast.Load(),
        ),
        args=[value],
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
