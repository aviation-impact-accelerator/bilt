import ast
import typing

from bilt.symbolic import _atoms as ops
from bilt.symbolic._atoms import (
    AxisType,
    ConcatAxisOpType,
    FloatType,
    IntType,
    QuantityType,
    SliceAxisOpType,
    SymbolType,
)
from bilt.symbolic._isolation._errors import BiltUnsupportedIsolationError

if typing.TYPE_CHECKING:
    from bilt.symbolic._tree import Tree
    from bilt.units import Dimension

__all__ = [
    "CONCAT_INTRINSIC_NAME",
    "SLICE_INTRINSIC_NAME",
    "isel_intrinsic_ast",
    "reduction_intrinsic_ast",
    "tree_to_functiondef",
]

MEAN_INTRINSIC_NAME = "__bilt_mean__"
SUM_INTRINSIC_NAME = "__bilt_sum__"
MIN_INTRINSIC_NAME = "__bilt_min__"
MAX_INTRINSIC_NAME = "__bilt_max__"
PROD_INTRINSIC_NAME = "__bilt_prod__"

ISEL_INTRINSIC_NAME = "__bilt_isel__"
CONCAT_INTRINSIC_NAME = "__bilt_concat__"
SLICE_INTRINSIC_NAME = "__bilt_slice__"

_REDUCTION_INTRINSICS = {
    ops.MEAN: MEAN_INTRINSIC_NAME,
    ops.SUM: SUM_INTRINSIC_NAME,
    ops.MIN: MIN_INTRINSIC_NAME,
    ops.MAX: MAX_INTRINSIC_NAME,
    ops.PROD: PROD_INTRINSIC_NAME,
}


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


def _tree_to_ast_expr(tree: Tree) -> ast.expr:
    """Convert a symbolic tree to Python AST."""
    if tree.is_leaf:
        return _leaf_to_ast(tree)

    if tree.head.atom_type is SliceAxisOpType:
        return _slice_to_ast(tree)

    if tree.head.atom_type is ConcatAxisOpType:
        return _concat_to_ast(tree)

    if tree.head in _REDUCTION_INTRINSICS:
        return _reduction_to_ast(tree, _REDUCTION_INTRINSICS[tree.head])

    if tree.is_relation:
        return _relation_to_ast(tree)

    return _operation_to_ast(tree)


def _operation_to_ast(tree: Tree) -> ast.expr:  # noqa: PLR0911
    match tree.head:
        case ops.CONDITIONAL:
            predicate, when_true, when_false = tree.args
            return ast.IfExp(
                test=_tree_to_ast_expr(predicate),
                body=_tree_to_ast_expr(when_true),
                orelse=_tree_to_ast_expr(when_false),
            )

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

        case ops.ISEL:
            return _isel_to_ast(tree)

        case _:
            message = f"Unsupported symbolic head {tree.head!r} in AST conversion."
            raise BiltUnsupportedIsolationError(message)


_RELATION_ARITY = 2

_RELATION_OPERATORS: dict[object, type[ast.cmpop]] = {
    ops.EQUALS: ast.Eq,
    ops.NOT_EQUALS: ast.NotEq,
    ops.LESS_THAN: ast.Lt,
    ops.LESS_THAN_OR_EQUALS: ast.LtE,
    ops.GREATER_THAN: ast.Gt,
    ops.GREATER_THAN_OR_EQUALS: ast.GtE,
}


def _relation_to_ast(tree: Tree) -> ast.Compare:
    if len(tree.args) != _RELATION_ARITY or tree.head not in _RELATION_OPERATORS:
        message = f"Unsupported symbolic predicate {tree!s}."
        raise BiltUnsupportedIsolationError(message)
    return ast.Compare(
        left=_tree_to_ast_expr(tree.args[0]),
        ops=[_RELATION_OPERATORS[tree.head]()],
        comparators=[_tree_to_ast_expr(tree.args[1])],
    )


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

    if tree.head.atom_type is QuantityType:
        return ast.Constant(value=typing.cast("tuple[float, Dimension, bool]", tree.head.value)[0])

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


def _slice_to_ast(tree: Tree) -> ast.expr:
    """Lower an already resolved positional slice to the backend intrinsic."""
    source_axis, target_axis, start, stop, step = typing.cast(
        "tuple[int, int, int | None, int | None, int | None]", tree.head.value
    )
    return ast.Call(
        func=ast.Name(id=SLICE_INTRINSIC_NAME, ctx=ast.Load()),
        args=[_tree_to_ast_expr(tree.args[0])],
        keywords=[
            ast.keyword(
                arg="bounds",
                value=ast.Tuple(
                    elts=[ast.Constant(value=bound) for bound in (start, stop, step)],
                    ctx=ast.Load(),
                ),
            ),
            ast.keyword(arg="source_axis", value=ast.Constant(value=source_axis)),
            ast.keyword(arg="target_axis", value=ast.Constant(value=target_axis)),
        ],
    )


def _concat_to_ast(tree: Tree) -> ast.expr:
    """Call the backend with aligned operands and source/target axis positions."""
    source_axes, target_axis = typing.cast("tuple[tuple[int, ...], int]", tree.head.value)
    return ast.Call(
        func=ast.Name(id=CONCAT_INTRINSIC_NAME, ctx=ast.Load()),
        args=[
            ast.Tuple(elts=[_tree_to_ast_expr(value) for value in tree.args], ctx=ast.Load()),
        ],
        keywords=[
            ast.keyword(
                arg="source_axes",
                value=ast.Tuple(elts=[ast.Constant(value=axis) for axis in source_axes], ctx=ast.Load()),
            ),
            ast.keyword(arg="target_axis", value=ast.Constant(value=target_axis)),
        ],
    )


def _isel_to_ast(tree: Tree) -> ast.expr:
    value_tree, axis_tree, selector_tree = tree.args

    if not axis_tree.is_leaf or axis_tree.head.atom_type is not AxisType:
        message = "Lowered isel requires a positional axis."
        raise BiltUnsupportedIsolationError(message)

    axis = typing.cast("int", axis_tree.head.value)

    return isel_intrinsic_ast(
        _tree_to_ast_expr(value_tree),
        _tree_to_ast_expr(selector_tree),
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


def isel_intrinsic_ast(
    value: ast.expr,
    selector: ast.expr,
    axis: int,
) -> ast.Call:
    """Construct a backend positional-selection intrinsic call."""
    return ast.Call(
        func=ast.Name(
            id=ISEL_INTRINSIC_NAME,
            ctx=ast.Load(),
        ),
        args=[
            value,
            selector,
        ],
        keywords=[
            ast.keyword(
                arg="axis",
                value=ast.Constant(value=axis),
            ),
        ],
    )
