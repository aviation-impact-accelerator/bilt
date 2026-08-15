__all__ = [
    "AffineAnalysisError",
    "AffineForm",
    "add_affine",
    "as_affine",
    "scale_affine",
    "subtract_affine",
]

import dataclasses
import types
import typing

from bilt.symbolic._algebra._expr_ops import (
    add,
    is_zero,
    multiply,
    multiply_many,
    negate,
    one,
    subtract,
    zero,
)
from bilt.symbolic._atoms import ADD, INV, MUL, NEG, POW, SymbolType

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    from bilt.symbolic._tree import Tree


class AffineAnalysisError(ValueError):
    """Raised when an expression is not affine in the requested symbols."""


@dataclasses.dataclass(frozen=True, slots=True)
class AffineForm:
    """An expression represented as symbolic coefficients plus a constant."""

    coefficients: Mapping[str, Tree]
    constant: Tree

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coefficients",
            types.MappingProxyType(dict(self.coefficients)),
        )


def as_affine(
    tree: Tree,
    symbols: frozenset[str],
) -> AffineForm:
    """Represent ``tree`` as an affine form in ``symbols``."""
    if not _depends_on(tree, symbols):
        return AffineForm(
            coefficients={},
            constant=tree,
        )

    if tree.is_leaf:
        if tree.head.atom_type is SymbolType and tree.head.value in symbols:
            return AffineForm(
                coefficients={str(tree.head.value): one()},
                constant=zero(),
            )

        message = f"Unsupported symbol-dependent leaf {tree!r}."
        raise AffineAnalysisError(message)

    match tree.head:
        case head if head is ADD:
            result = AffineForm(
                coefficients={},
                constant=zero(),
            )

            for child in tree.args:
                result = add_affine(
                    result,
                    as_affine(child, symbols),
                )

            return result

        case head if head is NEG:
            return scale_affine(
                as_affine(tree.args[0], symbols),
                negate(one()),
            )

        case head if head is MUL:
            dependent = tuple(child for child in tree.args if _depends_on(child, symbols))

            if len(dependent) != 1:
                message = f"Expression is nonlinear in {sorted(symbols)!r}: {tree!s}."
                raise AffineAnalysisError(message)

            dependent_child = dependent[0]

            return scale_affine(
                as_affine(dependent_child, symbols),
                multiply_many(child for child in tree.args if child is not dependent_child),
            )

        case head if head is INV or head is POW:
            message = f"Expression is nonlinear in {sorted(symbols)!r}: {tree!s}."
            raise AffineAnalysisError(message)

        case _:
            message = f"Unsupported symbol-dependent expression: {tree!s}."
            raise AffineAnalysisError(message)


def add_affine(
    left: AffineForm,
    right: AffineForm,
) -> AffineForm:
    """Add two affine forms."""
    coefficients = {
        symbol: coefficient
        for symbol in left.coefficients.keys() | right.coefficients.keys()
        if not is_zero(
            coefficient := add(
                left.coefficients.get(symbol, zero()),
                right.coefficients.get(symbol, zero()),
            )
        )
    }

    return AffineForm(
        coefficients=coefficients,
        constant=add(left.constant, right.constant),
    )


def subtract_affine(
    left: AffineForm,
    right: AffineForm,
) -> AffineForm:
    """Subtract ``right`` from ``left``."""
    coefficients = {
        symbol: coefficient
        for symbol in left.coefficients.keys() | right.coefficients.keys()
        if not is_zero(
            coefficient := subtract(
                left.coefficients.get(symbol, zero()),
                right.coefficients.get(symbol, zero()),
            )
        )
    }

    return AffineForm(
        coefficients=coefficients,
        constant=subtract(left.constant, right.constant),
    )


def scale_affine(
    form: AffineForm,
    factor: Tree,
) -> AffineForm:
    """Multiply an affine form by a symbol-independent factor."""
    if is_zero(factor):
        return AffineForm(
            coefficients={},
            constant=zero(),
        )

    return AffineForm(
        coefficients={
            symbol: coefficient
            for symbol, value in form.coefficients.items()
            if not is_zero(coefficient := multiply(value, factor))
        },
        constant=multiply(form.constant, factor),
    )


def _depends_on(
    tree: Tree,
    symbols: frozenset[str],
) -> bool:
    if tree.is_leaf:
        return tree.head.atom_type is SymbolType and tree.head.value in symbols

    return any(_depends_on(child, symbols) for child in tree.args)
