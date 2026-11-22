"""Bounded polynomial normalisation for Bilt's symbolic expression trees.

This deliberately supports only exact rational scalar polynomial algebra. It is
not a general computer algebra system or a source of conservation laws.
"""

__all__ = ["Monomial", "Polynomial", "PolynomialAnalysisError", "as_polynomial"]

import dataclasses
import math
import types
import typing
from fractions import Fraction

from bilt.symbolic._atoms import ADD, INV, MUL, NEG, POW, FloatType, IntType, SymbolType

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    from bilt.symbolic._tree import Tree

# Sorted symbol names, with repetition representing exponents; () is constant.
type Monomial = tuple[str, ...]


class PolynomialAnalysisError(ValueError):
    """An expression cannot be normalised within the supported polynomial subset."""


@dataclasses.dataclass(frozen=True, slots=True)
class Polynomial:
    """Immutable sparse polynomial with exact rational coefficients."""

    coefficients: Mapping[Monomial, Fraction]

    def __post_init__(self) -> None:
        cleaned: dict[Monomial, Fraction] = {}
        for monomial, coefficient in self.coefficients.items():
            key = tuple(sorted(monomial))
            cleaned[key] = cleaned.get(key, Fraction()) + Fraction(coefficient)
        object.__setattr__(self, "coefficients", types.MappingProxyType({k: v for k, v in cleaned.items() if v}))

    @property
    def degree(self) -> int:
        """Maximum total degree, or zero for a constant polynomial."""
        return max((len(term) for term in self.coefficients), default=0)

    @property
    def is_zero(self) -> bool:
        """Whether all coefficients cancel exactly."""
        return not self.coefficients


def as_polynomial(  # noqa: C901, PLR0912, PLR0915
    tree: Tree, *, max_degree: int = 4, max_terms: int = 256
) -> Polynomial:
    """Expand supported scalar arithmetic and combine equal monomials.

    Integers and decimal spellings of floating literals become exact Fractions.
    Supports +, unary -, *, nonnegative integer powers and division by
    nonzero scalar constants. Unsupported operators, degree and size limits
    raise PolynomialAnalysisError; no approximate algebra is attempted.

    This conversion is deliberately independent of quantity dimensionality.
    Callers must still check homogeneity before interpreting any relationship.
    """
    if max_degree < 0 or max_terms < 1:
        message = "max_degree must be nonnegative and max_terms must be positive."
        raise ValueError(message)

    def bounded(terms: dict[Monomial, Fraction]) -> Polynomial:
        result = Polynomial(terms)
        if result.degree > max_degree:
            message = f"Polynomial degree exceeds maximum of {max_degree}."
            raise PolynomialAnalysisError(message)
        if len(result.coefficients) > max_terms:
            message = f"Polynomial has more than {max_terms} terms."
            raise PolynomialAnalysisError(message)
        return result

    def addition(left: Polynomial, right: Polynomial) -> Polynomial:
        result = dict(left.coefficients)
        for monomial, coefficient in right.coefficients.items():
            value = result.get(monomial, Fraction()) + coefficient
            if value:
                result[monomial] = value
            else:
                result.pop(monomial, None)
        return bounded(result)

    def multiplication(left: Polynomial, right: Polynomial) -> Polynomial:
        result: dict[Monomial, Fraction] = {}
        for left_monomial, left_coefficient in left.coefficients.items():
            for right_monomial, right_coefficient in right.coefficients.items():
                monomial = tuple(sorted((*left_monomial, *right_monomial)))
                if len(monomial) > max_degree:
                    message = f"Polynomial degree exceeds maximum of {max_degree}."
                    raise PolynomialAnalysisError(message)
                result[monomial] = result.get(monomial, Fraction()) + left_coefficient * right_coefficient
                if len(result) > max_terms:
                    message = f"Polynomial has more than {max_terms} terms."
                    raise PolynomialAnalysisError(message)
        return bounded(result)

    def power(base: Polynomial, exponent: int) -> Polynomial:
        result = bounded({(): Fraction(1)})
        while exponent:
            if exponent & 1:
                result = multiplication(result, base)
            exponent >>= 1
            if exponent:
                base = multiplication(base, base)
        return result

    def convert(node: Tree) -> Polynomial:  # noqa: C901, PLR0911
        head = node.head
        if node.is_leaf:
            if head.atom_type is SymbolType:
                return bounded({(str(head.value),): Fraction(1)})
            if head.atom_type is IntType:
                return bounded({(): Fraction(typing.cast("int", head.value))})
            if head.atom_type is FloatType:
                value = typing.cast("float", head.value)
                if not math.isfinite(value):
                    message = "Non-finite polynomial coefficient."
                    raise PolynomialAnalysisError(message)
                return bounded({(): Fraction(str(value))})
            message = f"Unsupported polynomial leaf: {head!r}."
            raise PolynomialAnalysisError(message)
        if head is ADD:
            result = bounded({})
            for child in node.args:
                result = addition(result, convert(child))
            return result
        if head is NEG and len(node.args) == 1:
            return bounded({term: -value for term, value in convert(node.args[0]).coefficients.items()})
        if head is MUL:
            result = bounded({(): Fraction(1)})
            for child in node.args:
                result = multiplication(result, convert(child))
            return result
        if head is POW and len(node.args) == 2:  # noqa: PLR2004
            exponent_tree = node.args[1]
            if not exponent_tree.is_leaf or exponent_tree.head.atom_type is not IntType:
                message = "Only nonnegative integer polynomial powers are supported."
                raise PolynomialAnalysisError(message)
            exponent = typing.cast("int", exponent_tree.head.value)
            if exponent < 0:
                message = "Negative polynomial powers are unsupported."
                raise PolynomialAnalysisError(message)
            return power(convert(node.args[0]), exponent)
        if head is INV and len(node.args) == 1:
            denominator = convert(node.args[0]).coefficients
            if len(denominator) != 1 or () not in denominator or not denominator[()]:
                message = "Only division by nonzero scalar constants is supported."
                raise PolynomialAnalysisError(message)
            return bounded({(): Fraction(1, 1) / denominator[()]})
        message = f"Unsupported polynomial operator: {head!r}."
        raise PolynomialAnalysisError(message)

    return convert(tree)
