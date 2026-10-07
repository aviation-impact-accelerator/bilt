"""Work out the dimensionality of a symbolic tree from the dimensionalities of its symbols."""

__all__ = ["dimensionality_of"]

import functools
import typing

from bilt.symbolic import _atoms as ops
from bilt.symbolic._atoms import CoordinateType, DimensionType, FloatType, IntType, SymbolType
from bilt.units._errors import BiltDimensionalityError
from bilt.units.dimensions import DIMENSIONLESS

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.symbolic._tree import Tree
    from bilt.units import Dimension


def dimensionality_of(
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
    operands = [dimensionality_of(operand, dimensionalities) for operand in tree.args]
    return _dimensionality_at(tree, operands)


def _leaf(tree: Tree, dimensionalities: collections.abc.Mapping[str, Dimension]) -> Dimension | None:
    """Look a symbol's dimensionality up, read a nonzero literal as dimensionless, and ignore an axis or coordinate."""
    if tree.head.atom_type is SymbolType:
        return dimensionalities.get(typing.cast("str", tree.head.value))
    if tree.head.atom_type in (IntType, FloatType):
        return DIMENSIONLESS if tree.head.value != 0 else None
    if tree.head.atom_type in (DimensionType, CoordinateType):
        return None
    msg = f"Cannot work out the dimensionality of symbolic leaf {tree.head!r}."
    raise NotImplementedError(msg)


def _dimensionality_at(tree: Tree, operands: list[Dimension | None]) -> Dimension | None:  # noqa: PLR0911
    """Return the dimensionality of the operation at the head of `tree`, given its operands' dimensionalities."""
    if tree.is_relation:
        _shared(operands)
        return DIMENSIONLESS
    match tree.head:
        case ops.ADD | ops.NEG | ops.MEAN | ops.SUM | ops.MIN | ops.MAX:
            return _shared(operands)
        case ops.PROD | ops.ISEL | ops.SEL:
            return None
        case ops.MUL:
            return _product(operands, [1] * len(tree.args))
        case ops.INV:
            return _product(operands, [-1])
        case ops.POW if tree.args[1].head.atom_type is IntType:
            return _product(operands, [typing.cast("int", tree.args[1].head.value), 0])
        case ops.POW:
            return _power(*operands)
        case _:
            msg = f"Cannot work out the dimensionality of symbolic operation {tree.head!r}."
            raise NotImplementedError(msg)


def _shared(slots: list[Dimension | None]) -> Dimension | None:
    """Return the dimensionality every known slot shares."""
    known = [slot for slot in slots if slot is not None]
    return functools.reduce(_add, known) if known else None


def _add(left: Dimension, right: Dimension) -> Dimension:
    """Demand that two addends share a dimensionality."""
    if left is not right:
        msg = f"Cannot add {left} to {right}: addends must share a dimensionality."
        raise BiltDimensionalityError(msg)
    return left


def _product(operands: list[Dimension | None], exponents: list[int]) -> Dimension | None:
    """Return the dimensionality of a product of `operands` raised to `exponents`, or None if one is unknown."""
    product = DIMENSIONLESS
    for operand, exponent in zip(operands, exponents, strict=True):
        if exponent == 0:
            continue
        if operand is None:
            return None
        product *= operand**exponent
    return product


def _power(base: Dimension | None, exponent: Dimension | None) -> Dimension:
    """Raise a base to an exponent other than an integer literal, which demands that both are dimensionless."""
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
