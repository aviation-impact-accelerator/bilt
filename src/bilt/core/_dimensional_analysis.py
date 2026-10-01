"""Work out the dimensionality of a symbolic tree from the dimensionalities of its symbols."""

__all__ = ["dimensionality_of", "implied_dimensionality"]

import functools
import typing

from bilt.symbolic import _atoms as ops
from bilt.symbolic._atoms import DimensionType, FloatType, IntType, SymbolType
from bilt.symbolic._tree import symbol_tree
from bilt.symbolic._walk import contains
from bilt.units import Dimension
from bilt.units._errors import BiltDimensionalityError
from bilt.units.dimensions import DIMENSIONLESS

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.symbolic._tree import Tree


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
    return _dimensionality_at(tree, [None, *operands], 0)


def _leaf(tree: Tree, dimensionalities: collections.abc.Mapping[str, Dimension]) -> Dimension | None:
    """Look a symbol's dimensionality up, read a nonzero numeric literal as dimensionless, and ignore a mean's axis."""
    if tree.head.atom_type is SymbolType:
        return dimensionalities.get(typing.cast("str", tree.head.value))
    if tree.head.atom_type in (IntType, FloatType):
        return DIMENSIONLESS if tree.head.value != 0 else None
    if tree.head.atom_type is DimensionType:
        return None
    msg = f"Cannot work out the dimensionality of symbolic leaf {tree.head!r}."
    raise NotImplementedError(msg)


def _dimensionality_at(tree: Tree, slots: list[Dimension | None], index: int) -> Dimension | None:
    """Return the dimensionality `tree` requires in `slots[index]`, given the others: its own, then its operands'."""
    if tree.is_relation:
        sides = _shared(slots[1:])
        return sides if index else DIMENSIONLESS
    match tree.head:
        case ops.ADD | ops.NEG | ops.MEAN:
            return _shared(slots)
        case ops.MUL:
            return _product(slots, index, [1] * len(tree.args))
        case ops.INV:
            return _product(slots, index, [-1])
        case ops.POW if tree.args[1].head.atom_type is IntType:
            return _product(slots, index, [typing.cast("int", tree.args[1].head.value), 0])
        case ops.POW:
            return _power(*slots[1:])
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


def _product(slots: list[Dimension | None], index: int, exponents: list[int]) -> Dimension | None:
    """Return the dimensionality in `slots[index]` of a product of operands raised to `exponents`."""
    product, *operands = slots
    others = DIMENSIONLESS
    for position, (operand, exponent) in enumerate(zip(operands, exponents, strict=True), start=1):
        if position == index or exponent == 0:
            continue
        if operand is None:
            return None
        others *= operand**exponent
    if index == 0:
        return others
    return None if product is None else _root(product / others, exponents[index - 1])


def _root(dimensionality: Dimension, degree: int) -> Dimension | None:
    """Return the dimensionality whose `degree`-th power is `dimensionality`, or None when every dimensionality's is."""
    if degree == 0 and dimensionality.is_dimensionless:
        return None
    if degree == 0 or any(exponent % degree for exponent in dimensionality.dimensions):
        msg = f"Cannot raise any dimensionality to the power {degree} to give {dimensionality}."
        raise BiltDimensionalityError(msg)
    return Dimension(*(exponent // degree for exponent in dimensionality.dimensions))


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


def implied_dimensionality(
    tree: Tree,
    symbol: str,
    dimensionalities: collections.abc.Mapping[str, Dimension],
) -> Dimension | None:
    """Return the dimensionality `tree` implies for `symbol`, given the dimensionality of each other symbol."""
    if symbol in dimensionalities:
        dimensionalities = {name: dimensionality for name, dimensionality in dimensionalities.items() if name != symbol}
    demands = _demands(tree, symbol_tree(symbol), None, dimensionalities)
    implied = list(dict.fromkeys(demand for demand in demands if demand is not None))
    if len(implied) > 1:
        msg = f"Cannot give {symbol!r} both {implied[0]} and {implied[1]}: its occurrences demand both."
        raise BiltDimensionalityError(msg)
    return implied[0] if implied else None


def _demands(
    tree: Tree,
    symbol: Tree,
    demand: Dimension | None,
    dimensionalities: collections.abc.Mapping[str, Dimension],
) -> collections.abc.Iterator[Dimension | None]:
    """Yield the demand reaching each occurrence of `symbol` in `tree`, which is demanded `demand`."""
    if tree is symbol:
        yield demand
    elif not tree.is_leaf:
        slots = [demand, *(dimensionality_of(operand, dimensionalities) for operand in tree.args)]
        for index, operand in enumerate(tree.args, start=1):
            if contains(operand, symbol):
                demanded = _dimensionality_at(tree, [*slots[:index], None, *slots[index + 1 :]], index)
                yield from _demands(operand, symbol, demanded, dimensionalities)
