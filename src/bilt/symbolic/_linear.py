__all__ = [
    "LinearisationError",
    "SymbolicLinearSystem",
    "linearise",
]

import dataclasses

from bilt.symbolic._atoms import (
    ADD,
    INV,
    MUL,
    NEG,
    POW,
    FloatType,
    IntType,
    SymbolType,
)
from bilt.symbolic._expressions import Expr
from bilt.symbolic._isolation._expr_ops import (
    make_add,
    make_mul,
    make_neg,
    simplify,
)
from bilt.symbolic._tree import Tree, int_tree


class LinearisationError(ValueError):
    """Raised when equations cannot be represented as a linear system."""


@dataclasses.dataclass(frozen=True, slots=True)
class SymbolicLinearSystem:
    """A symbolic linear system of the form A x = b."""

    parameters: tuple[str, ...]
    unknowns: tuple[str, ...]
    coefficients: tuple[tuple[Tree, ...], ...]
    rhs: tuple[Tree, ...]


@dataclasses.dataclass(slots=True)
class _AffineExpression:
    """An expression represented as a constant plus linear coefficients."""

    constant: Expr
    coefficients: dict[str, Expr]


def linearise(
    equations: tuple[Tree, ...],
    unknowns: tuple[str, ...],
) -> SymbolicLinearSystem:
    """Represent equality equations as a symbolic linear system."""
    if len(equations) != len(unknowns):
        message = (
            "A linear system must contain the same number of equations "
            f"and unknowns, got {len(equations)} equations and "
            f"{len(unknowns)} unknowns."
        )
        raise LinearisationError(message)

    if len(set(unknowns)) != len(unknowns):
        message = f"Linear system contains duplicate unknowns: {unknowns!r}."
        raise LinearisationError(message)

    unknown_set = frozenset(unknowns)

    coefficient_rows: list[tuple[Tree, ...]] = []
    rhs: list[Tree] = []

    for equation in equations:
        if not equation.is_equality:
            message = f"Coupled linear solving requires equality equations, got {equation!s}."
            raise LinearisationError(message)

        left = _analyse_expression(
            equation.left,
            unknown_set,
        )
        right = _analyse_expression(
            equation.right,
            unknown_set,
        )

        row = _subtract(left, right)

        coefficient_rows.append(tuple(row.coefficients.get(unknown, _zero()).rep for unknown in unknowns))

        rhs.append(_negate(row.constant).rep)

    coefficients = tuple(coefficient_rows)
    rhs_tuple = tuple(rhs)

    return SymbolicLinearSystem(
        parameters=_linear_system_parameters(
            coefficients,
            rhs_tuple,
            unknown_set,
        ),
        unknowns=unknowns,
        coefficients=coefficients,
        rhs=rhs_tuple,
    )


def _analyse_expression(
    tree: Tree,
    unknowns: frozenset[str],
) -> _AffineExpression:
    """Return the affine representation of a symbolic expression."""
    if not _depends_on_unknown(tree, unknowns):
        return _AffineExpression(
            constant=Expr(tree),
            coefficients={},
        )

    if tree.is_leaf:
        if tree.head.atom_type is SymbolType and tree.head.value in unknowns:
            return _AffineExpression(
                constant=_zero(),
                coefficients={
                    str(tree.head.value): _one(),
                },
            )

        message = f"Unsupported unknown-dependent leaf {tree!r}."
        raise LinearisationError(message)

    match tree.head:
        case head if head is ADD:
            result = _AffineExpression(
                constant=_zero(),
                coefficients={},
            )

            for child in tree.args:
                result = _add(
                    result,
                    _analyse_expression(child, unknowns),
                )

            return result

        case head if head is NEG:
            return _scale(
                _analyse_expression(tree.args[0], unknowns),
                _negate(_one()),
            )

        case head if head is MUL:
            dependent = tuple(child for child in tree.args if _depends_on_unknown(child, unknowns))

            if len(dependent) != 1:
                message = f"Expression is nonlinear in {sorted(unknowns)!r}: {tree!s}."
                raise LinearisationError(message)

            dependent_child = dependent[0]

            known_factors = [Expr(child) for child in tree.args if child is not dependent_child]

            return _scale(
                _analyse_expression(
                    dependent_child,
                    unknowns,
                ),
                _multiply_many(known_factors),
            )

        case head if head is INV or head is POW:
            message = f"Expression is nonlinear in {sorted(unknowns)!r}: {tree!s}."
            raise LinearisationError(message)

        case _:
            message = f"Unsupported unknown-dependent expression: {tree!s}."
            raise LinearisationError(message)


def _depends_on_unknown(
    tree: Tree,
    unknowns: frozenset[str],
) -> bool:
    if tree.is_leaf:
        return tree.head.atom_type is SymbolType and tree.head.value in unknowns

    return any(_depends_on_unknown(child, unknowns) for child in tree.args)


def _add(
    left: _AffineExpression,
    right: _AffineExpression,
) -> _AffineExpression:
    coefficients: dict[str, Expr] = {}

    for unknown in left.coefficients.keys() | right.coefficients.keys():
        coefficients[unknown] = _add_expr(
            left.coefficients.get(unknown, _zero()),
            right.coefficients.get(unknown, _zero()),
        )

    return _AffineExpression(
        constant=_add_expr(
            left.constant,
            right.constant,
        ),
        coefficients=coefficients,
    )


def _subtract(
    left: _AffineExpression,
    right: _AffineExpression,
) -> _AffineExpression:
    return _add(
        left,
        _scale(
            right,
            _negate(_one()),
        ),
    )


def _scale(
    expression: _AffineExpression,
    factor: Expr,
) -> _AffineExpression:
    if _is_minus_one(factor):
        return _AffineExpression(
            constant=_negate(expression.constant),
            coefficients={unknown: _negate(coefficient) for unknown, coefficient in expression.coefficients.items()},
        )

    return _AffineExpression(
        constant=_multiply(
            expression.constant,
            factor,
        ),
        coefficients={
            unknown: _multiply(coefficient, factor) for unknown, coefficient in expression.coefficients.items()
        },
    )


def _add_expr(
    left: Expr,
    right: Expr,
) -> Expr:
    if _is_zero(left):
        return right

    if _is_zero(right):
        return left

    return simplify(
        make_add(
            left,
            right,
        )
    )


def _multiply(
    left: Expr,
    right: Expr,
) -> Expr:
    if _is_zero(left) or _is_zero(right):
        return _zero()

    if _is_one(left):
        return right

    if _is_one(right):
        return left

    if _is_minus_one(left):
        return _negate(right)

    if _is_minus_one(right):
        return _negate(left)

    return simplify(
        make_mul(
            left,
            right,
        )
    )


def _multiply_many(
    expressions: list[Expr],
) -> Expr:
    result = _one()

    for expression in expressions:
        result = _multiply(
            result,
            expression,
        )

    return result


def _negate(expression: Expr) -> Expr:
    if _is_zero(expression):
        return expression

    return simplify(make_neg(expression))


def _linear_system_parameters(
    coefficients: tuple[tuple[Tree, ...], ...],
    rhs: tuple[Tree, ...],
    unknowns: frozenset[str],
) -> tuple[str, ...]:
    names: set[str] = set()

    for row in coefficients:
        for tree in row:
            names.update(_symbol_names(tree))

    for tree in rhs:
        names.update(_symbol_names(tree))

    return tuple(sorted(names - unknowns))


def _symbol_names(tree: Tree) -> set[str]:
    names: set[str] = set()
    pending = [tree]

    while pending:
        node = pending.pop()

        if node.is_leaf and node.head.atom_type is SymbolType:
            names.add(str(node.head.value))

        pending.extend(node.args)

    return names


def _zero() -> Expr:
    return Expr(int_tree(0))


def _one() -> Expr:
    return Expr(int_tree(1))


def _is_zero(expression: Expr) -> bool:
    tree = expression.rep

    return tree.is_leaf and tree.head.atom_type in (IntType, FloatType) and tree.head.value == 0


def _is_one(expression: Expr) -> bool:
    tree = expression.rep

    return tree.is_leaf and tree.head.atom_type in (IntType, FloatType) and tree.head.value == 1


def _is_minus_one(expression: Expr) -> bool:
    tree = expression.rep

    return (
        tree.head is NEG
        and len(tree.args) == 1
        and tree.args[0].is_leaf
        and tree.args[0].head.atom_type in (IntType, FloatType)
        and tree.args[0].head.value == 1
    )
