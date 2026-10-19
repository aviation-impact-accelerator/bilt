"""Work out the dimensionality and the absoluteness of a symbolic tree from what is known of its symbols."""

__all__ = ["absoluteness_of", "dimensionality_of"]

import functools
import typing

from bilt.core._errors import BiltEquationDimensionalityError
from bilt.symbolic import _atoms as ops
from bilt.symbolic._atoms import CoordinateType, DimensionType, FloatType, IntType, QuantityType, SymbolType
from bilt.units._errors import BiltDimensionalityError
from bilt.units.dimensions import DIMENSIONLESS

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._port_declaration import PortDeclaration
    from bilt.symbolic._tree import Tree
    from bilt.units import Dimension


_EITHER = frozenset({0, 1})


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
    if tree.head.atom_type is QuantityType:
        return typing.cast("tuple[float, Dimension, bool]", tree.head.value)[1]
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
        case ops.ISEL | ops.SEL:
            return operands[0]
        case ops.PROD:
            return _reduced_product(operands[0])
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


def _reduced_product(operand: Dimension | None) -> Dimension | None:
    """Return the dimensionality of a product along an axis, which only a dimensionless tensor has."""
    if operand is not None and not operand.is_dimensionless:
        msg = f"Cannot multiply {operand} along an axis: the result's dimensionality would depend on the axis's length."
        raise BiltDimensionalityError(msg)
    return operand


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


def absoluteness_of(  # noqa: PLR0911
    equation_name: str,
    tree: Tree,
    declarations: collections.abc.Mapping[str, PortDeclaration],
) -> frozenset[int]:
    """Return every legal reading of `tree`, each the number of fixed zeros it is measured from.

    An absolute quantity is measured from one fixed zero and a relative quantity from none, so a sum adds its
    addends' and a negation negates its operand's. A port absent from `declarations`, a literal, and a product,
    quotient or power are neither absolute nor relative, and are read either way.
    """
    if tree.is_leaf:
        if tree.head.atom_type is SymbolType:
            declaration = declarations.get(typing.cast("str", tree.head.value))
            return _EITHER if declaration is None else frozenset({int(declaration.absolute)})
        if tree.head.atom_type is QuantityType:
            return frozenset({int(typing.cast("tuple[float, Dimension, bool]", tree.head.value)[2])})
        return _EITHER
    operands = [absoluteness_of(equation_name, operand, declarations) for operand in tree.args]
    if tree.is_relation:
        if not operands[0] & operands[1]:
            msg = "relates an absolute quantity to a relative one: both sides must be absolute, or both relative."
            raise BiltEquationDimensionalityError(equation_name, msg)
        return frozenset({0})
    match tree.head:
        case ops.NEG:
            return frozenset(-zeros for zeros in operands[0])
        case ops.ADD:
            return functools.reduce(functools.partial(_sum_readings, equation_name), operands)
        case ops.MEAN | ops.MIN | ops.MAX | ops.ISEL | ops.SEL:
            return operands[0]
        case ops.SUM:
            return _summed_along_an_axis(equation_name, operands[0])
        case _:
            return _EITHER


def _summed_along_an_axis(equation_name: str, operand: frozenset[int]) -> frozenset[int]:
    """Read a sum along an axis, which adds the operand's own values and so is never absolute."""
    readings = operand - {1}
    if not readings:
        msg = "adds absolute quantities along an axis: they subtract to a relative one and never add."
        raise BiltEquationDimensionalityError(equation_name, msg)
    return readings


def _sum_readings(equation_name: str, left: frozenset[int], right: frozenset[int]) -> frozenset[int]:
    """Read the sum of two addends every way that leaves it absolute or relative."""
    totals = {ours + theirs for ours in left for theirs in right}
    readings = _EITHER & totals
    if not readings:
        msg = (
            "adds two absolute quantities: they subtract to a relative one and never add."
            if min(totals) > 1
            else "subtracts an absolute quantity from a relative one: it subtracts only from another absolute one."
        )
        raise BiltEquationDimensionalityError(equation_name, msg)
    return readings
