"""Symbolic parameter derivatives usable directly inside @equation bodies.

These operators differentiate *expressions*. They do not infer an implicit
dependency between fields defined by separate model equations.
"""

import typing

from bilt.symbolic._atoms import SymbolType
from bilt.symbolic._differential import DEFERRED_GRAD
from bilt.symbolic._differentiation import jvp_tree
from bilt.symbolic._expressions import Expr, TensorExpr, as_tree
from bilt.symbolic._tree import Tree, float_tree

__all__ = ["grad", "hessian", "jacobian"]


def _symbol_name(selector: object) -> str:
    if not isinstance(selector, Expr) or not selector.is_symbol:
        msg = "wrt must refer to a symbolic equation parameter, not a numerical value or expression."
        raise TypeError(msg)
    if selector.tree.head.atom_type is not SymbolType:
        msg = "wrt must be a symbolic equation parameter."
        raise TypeError(msg)
    return typing.cast("str", selector.tree.head.value)


def _selectors(selectors: tuple[object, ...]) -> tuple[str, ...]:
    if not selectors:
        msg = "wrt must contain at least one symbolic equation parameter."
        raise ValueError(msg)
    names = tuple(_symbol_name(selector) for selector in selectors)
    if len(names) != len(set(names)):
        msg = "wrt contains duplicate equation parameters."
        raise ValueError(msg)
    return names


def _partial(expression: Tree, parameter: str) -> Tree:
    return jvp_tree(expression, {parameter: float_tree(1.0)})


def _gradient(expression: object, selector: object) -> Expr:
    """Retain a tensor state's rate with respect to a distinct tensor coordinate.

    Scalar expression derivatives and explicit tensor derivatives still use
    ordinary symbolic JVPs. Formal rates are classified at implementation
    level, where all model equations and initial conditions are available.
    """
    parameter = _symbol_name(selector)
    tree = as_tree(expression)
    if (
        isinstance(expression, TensorExpr)
        and isinstance(selector, TensorExpr)
        and expression.is_symbol
        and expression.tree is not selector.tree
    ):
        return TensorExpr(Tree(DEFERRED_GRAD, (tree, selector.tree)))
    return Expr(_partial(tree, parameter))


@typing.overload
def grad(expression: object, *, wrt: tuple[object, ...]) -> tuple[Expr, ...]: ...


@typing.overload
def grad(expression: object, *, wrt: object) -> Expr: ...


def grad(expression: object, *, wrt: object | tuple[object, ...]) -> Expr | tuple[Expr, ...]:
    """Differentiate a local symbolic expression with respect to equation parameters.

    A scalar selector returns an Expr; an ordered tuple returns a tuple of Exprs.
    """
    if isinstance(wrt, tuple):
        _selectors(wrt)
        return tuple(_gradient(expression, selector) for selector in wrt)
    return _gradient(expression, wrt)


def jacobian(expressions: tuple[object, ...], *, wrt: tuple[object, ...]) -> tuple[tuple[Expr, ...], ...]:
    """Return the Jacobian of local expressions (output rows, input columns)."""
    if not isinstance(expressions, tuple) or not expressions:
        msg = "jacobian requires a non-empty tuple of output expressions."
        raise ValueError(msg)
    names = _selectors(wrt)
    return tuple(tuple(Expr(_partial(as_tree(expression), name)) for name in names) for expression in expressions)


def hessian(expression: object, *, wrt: tuple[object, ...]) -> tuple[tuple[Expr, ...], ...]:
    """Return a full Hessian using two exact symbolic differentiation passes."""
    names = _selectors(wrt)
    tree = as_tree(expression)
    first = tuple(_partial(tree, name) for name in names)
    return tuple(tuple(Expr(_partial(derivative, name)) for name in names) for derivative in first)
