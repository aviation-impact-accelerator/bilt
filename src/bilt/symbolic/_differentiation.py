"""Forward-mode differentiation of Bilt's symbolic expression trees."""

import typing

from bilt.symbolic._algebra._expr_ops import add, divide, is_zero, multiply, multiply_many, negate, zero
from bilt.symbolic._atoms import (
    ADD,
    CONDITIONAL,
    INV,
    ISEL,
    MEAN,
    MUL,
    NEG,
    POW,
    SUM,
    FloatType,
    IntType,
    SymbolType,
)
from bilt.symbolic._tree import Tree, float_tree, int_tree

if typing.TYPE_CHECKING:
    from collections.abc import Mapping


class UnsupportedDerivativeError(ValueError):
    """An expression cannot be differentiated using the supported IR operations."""


def jvp_tree(tree: Tree, tangents: Mapping[str, Tree]) -> Tree:  # noqa: C901
    """Build a symbolic directional derivative without needing a numerical backend."""
    cache: dict[Tree, Tree] = {}

    def tangent(node: Tree) -> Tree:  # noqa: C901, PLR0912
        if node in cache:
            return cache[node]
        if node.is_leaf:
            result = (
                tangents.get(typing.cast("str", node.head.value), zero())
                if node.head.atom_type is SymbolType
                else zero()
            )
        elif node.head is ADD:
            result = zero()
            for child in node.args:
                result = add(result, tangent(child))
        elif node.head is NEG:
            result = negate(tangent(node.args[0]))
        elif node.head is MUL:
            result = zero()
            for index, child in enumerate(node.args):
                derivative = tangent(child)
                if not is_zero(derivative):
                    factors = (derivative if i == index else factor for i, factor in enumerate(node.args))
                    result = add(result, multiply_many(factors))
        elif node.head is INV:
            child = node.args[0]
            derivative = tangent(child)
            result = zero() if is_zero(derivative) else divide(negate(derivative), Tree(POW, (child, int_tree(2))))
        elif node.head is POW:
            base, exponent = node.args
            derivative = tangent(base)
            if exponent.is_leaf and exponent.head.atom_type in (IntType, FloatType):
                power = typing.cast("int | float", exponent.head.value)
                if is_zero(derivative) or power == 0:
                    result = zero()
                elif power == 1:
                    result = derivative
                else:
                    reduced_power = int_tree(power - 1) if isinstance(power, int) else float_tree(power - 1)
                    result = multiply(multiply(exponent, Tree(POW, (base, reduced_power))), derivative)
            else:
                msg = "Differentiation of a variable exponent is unsupported."
                raise UnsupportedDerivativeError(msg)
        elif node.head in (SUM, MEAN):
            value, axis = node.args
            result = Tree(node.head, (tangent(value), axis))
        elif node.head is ISEL:
            value, axis, selector = node.args
            result = Tree(ISEL, (tangent(value), axis, selector))
        elif node.head is CONDITIONAL:
            predicate, when_true, when_false = node.args
            result = Tree(CONDITIONAL, (predicate, tangent(when_true), tangent(when_false)))
        else:
            msg = f"Unsupported differentiable operation: {node.head}."
            raise UnsupportedDerivativeError(msg)
        cache[node] = result
        return result

    return tangent(tree)
