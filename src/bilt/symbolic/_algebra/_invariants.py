"""Experimental exact elimination of internal polynomial terms.

Find constant-coefficient combinations of equations whose residuals involve
only selected boundary ports. This is not complete polynomial ideal elimination.
"""

__all__ = ["DiscoveredInvariant", "discover_boundary_invariants"]

import dataclasses
import math
import typing
from fractions import Fraction

from bilt.symbolic._algebra._expr_ops import subtract
from bilt.symbolic._algebra._polynomial import Monomial, Polynomial, as_polynomial

if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from bilt.symbolic._tree import Tree


@dataclasses.dataclass(frozen=True, slots=True)
class DiscoveredInvariant:
    """A derived polynomial residual, with exact source-equation weights."""

    residual: Polynomial
    source_coefficients: Mapping[str, Fraction]


def _primitive(vector: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    """Normalise rational coefficients to a primitive, sign-canonical vector."""
    nonzero = [value for value in vector if value]
    if not nonzero:
        return vector
    lcm = math.lcm(*(value.denominator for value in nonzero))
    integers = [int(value * lcm) for value in vector]
    common = math.gcd(*integers)
    integers = [value // common for value in integers]
    if next(value for value in integers if value) < 0:
        integers = [-value for value in integers]
    return tuple(Fraction(value) for value in integers)


def _nullspace(matrix: list[list[Fraction]], columns: int) -> tuple[tuple[Fraction, ...], ...]:
    """Compute a basis of the exact rational nullspace using RREF."""
    if columns <= 0:
        return ()
    rows = [row.copy() for row in matrix]
    pivots: list[int] = []
    pivot_row = 0
    for col in range(columns):
        pivot = next((row for row in range(pivot_row, len(rows)) if rows[row][col]), None)
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        coefficient = rows[pivot_row][col]
        rows[pivot_row] = [value / coefficient for value in rows[pivot_row]]
        for row in range(len(rows)):
            if row == pivot_row or not rows[row][col]:
                continue
            factor = rows[row][col]
            rows[row] = [value - factor * target for value, target in zip(rows[row], rows[pivot_row], strict=True)]
        pivots.append(col)
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return tuple(
        _primitive(
            tuple(
                Fraction(1) if idx == free else -rows[pivots.index(idx)][free] if idx in pivots else Fraction()
                for idx in range(columns)
            )
        )
        for free in range(columns)
        if free not in pivots
    )


def discover_boundary_invariants(  # noqa: C901
    equations: Sequence[tuple[str, Tree]],
    boundary_ports: frozenset[str],
    *,
    max_degree: int = 4,
    max_terms: int = 256,
    max_equations: int = 32,
) -> tuple[DiscoveredInvariant, ...]:
    """Find additive polynomial consequences over the specified boundary ports.

    Each original equation must be an equality tree. Only combinations using
    constant rational multipliers are searched. All non-boundary monomials,
    higher-degree boundary monomials and constants must cancel.

    For example, output = eta * input and heat = (1-eta) * input
    imply input = output + heat without declaring a balance equation.
    """
    if len(equations) > max_equations:
        message = f"Too many equations for exact elimination (limit {max_equations})."
        raise ValueError(message)
    if len({name for name, _ in equations}) != len(equations):
        message = "Equation labels must be unique."
        raise ValueError(message)
    residuals: list[Polynomial] = []
    for _name, tree in equations:
        if not tree.is_equality:
            message = "Invariant discovery requires equality equations."
            raise ValueError(message)
        residuals.append(as_polynomial(subtract(tree.left, tree.right), max_degree=max_degree, max_terms=max_terms))

    def allowed(monomial: Monomial) -> bool:
        return len(monomial) == 1 and monomial[0] in boundary_ports

    unwanted = sorted({key for poly in residuals for key in poly.coefficients if not allowed(key)})
    elimination = [[poly.coefficients.get(monomial, Fraction()) for poly in residuals] for monomial in unwanted]
    candidates: list[DiscoveredInvariant] = []
    seen: set[tuple[tuple[Monomial, Fraction], ...]] = set()
    for weights in _nullspace(elimination, len(equations)):
        terms: dict[Monomial, Fraction] = {}
        for weight, polynomial in zip(weights, residuals, strict=True):
            if not weight:
                continue
            for monomial, coefficient in polynomial.coefficients.items():
                terms[monomial] = terms.get(monomial, Fraction()) + weight * coefficient
        result = Polynomial(terms)
        if result.is_zero or any(not allowed(monomial) for monomial in result.coefficients):
            continue
        key = tuple(sorted(result.coefficients.items()))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            DiscoveredInvariant(
                residual=result,
                source_coefficients={
                    label: weight for (label, _tree), weight in zip(equations, weights, strict=True) if weight
                },
            )
        )
    return tuple(candidates)
