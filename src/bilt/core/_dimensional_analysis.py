"""Work out the dimensionality of a symbolic tree from the dimensionalities of its symbols."""

__all__ = ["dimensionality_of"]

import functools
import operator
import typing

from bilt.symbolic import _atoms as ops
from bilt.symbolic._atoms import FloatType, IntType, SymbolType
from bilt.units._errors import BiltDimensionalityError
from bilt.units.dimensions import DIMENSIONLESS

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.symbolic._tree import Tree
    from bilt.units import Dimension


def dimensionality_of(  # noqa: PLR0911
    tree: Tree,
    dimensionalities: collections.abc.Mapping[str, Dimension],
) -> Dimension | None:
    """Return the dimensionality of `tree`, given the dimensionality of each symbol in `dimensionalities`.

    A symbol absent from `dimensionalities` is unknown, and so is a literal zero, which has whatever
    dimensionality its context demands. An unknown addend takes the dimensionality of the addends beside
    it; anything else an unknown reaches is None, while every subexpression of known dimensionalities is
    still checked.

    Raises:
        BiltDimensionalityError: The tree is dimensionally inhomogeneous.
    """
    if tree.is_leaf:
        return _leaf(tree, dimensionalities)
    if tree.head is ops.MEAN:
        return dimensionality_of(tree.args[0], dimensionalities)

    operands = [dimensionality_of(operand, dimensionalities) for operand in tree.args]
    known = [operand for operand in operands if operand is not None]
    match tree.head:
        case ops.ADD:
            return functools.reduce(_add, known) if known else None
        case ops.MUL:
            return functools.reduce(operator.mul, known) if len(known) == len(operands) else None
        case ops.NEG:
            return operands[0]
        case ops.INV:
            return None if operands[0] is None else operands[0] ** -1
        case ops.POW:
            base, exponent = operands
            return _power(base, exponent, tree.args[1])
        case _:
            msg = f"Cannot work out the dimensionality of symbolic operation {tree.head!r}."
            raise NotImplementedError(msg)


def _leaf(tree: Tree, dimensionalities: collections.abc.Mapping[str, Dimension]) -> Dimension | None:
    """Look a symbol's dimensionality up, and read a nonzero numeric literal as dimensionless."""
    if tree.head.atom_type is SymbolType:
        return dimensionalities.get(typing.cast("str", tree.head.value))
    if tree.head.atom_type in (IntType, FloatType):
        return DIMENSIONLESS if tree.head.value != 0 else None
    msg = f"Cannot work out the dimensionality of symbolic leaf {tree.head!r}."
    raise NotImplementedError(msg)


def _add(left: Dimension, right: Dimension) -> Dimension:
    """Demand that two addends share a dimensionality."""
    if left is not right:
        msg = f"Cannot add {left} to {right}: addends must share a dimensionality."
        raise BiltDimensionalityError(msg)
    return left


def _power(base: Dimension | None, exponent: Dimension | None, exponent_tree: Tree) -> Dimension | None:
    """Raise a base to an exponent, which only an integer literal may do to a dimensional base."""
    if exponent_tree.head.atom_type is IntType:
        return None if base is None else base ** typing.cast("int", exponent_tree.head.value)

    if exponent is not None and not exponent.is_dimensionless:
        msg = f"Cannot raise to an exponent of {exponent}: an exponent must be dimensionless."
        raise BiltDimensionalityError(msg)
    if base is not None and not base.is_dimensionless:
        msg = (
            f"Cannot raise {base} to anything but an integer literal: "
            "any other exponent makes the result's dimensionality depend on a value."
        )
        raise BiltDimensionalityError(msg)
    return DIMENSIONLESS
