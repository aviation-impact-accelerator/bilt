__all__ = [
    "LinearisationError",
    "SymbolicLinearSystem",
    "linearise",
]

import dataclasses

from bilt.symbolic._algebra._affine import (
    AffineAnalysisError,
    as_affine,
    subtract_affine,
)
from bilt.symbolic._algebra._expr_ops import negate, zero
from bilt.symbolic._tree import Tree
from bilt.symbolic._walk import symbol_names


class LinearisationError(ValueError):
    """Raised when equations cannot be represented as a linear system."""


@dataclasses.dataclass(frozen=True, slots=True)
class SymbolicLinearSystem:
    """A symbolic linear system of the form A x = b."""

    parameters: tuple[str, ...]
    unknowns: tuple[str, ...]
    coefficients: tuple[tuple[Tree, ...], ...]
    rhs: tuple[Tree, ...]


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

        try:
            left = as_affine(
                equation.left,
                unknown_set,
            )
            right = as_affine(
                equation.right,
                unknown_set,
            )
        except AffineAnalysisError as error:
            raise LinearisationError(str(error)) from error

        row = subtract_affine(left, right)

        coefficient_rows.append(tuple(row.coefficients.get(unknown, zero()) for unknown in unknowns))
        rhs.append(negate(row.constant))

    coefficients = tuple(coefficient_rows)
    rhs_tuple = tuple(rhs)

    return SymbolicLinearSystem(
        parameters=_linear_system_parameters(
            coefficients,
            rhs_tuple,
        ),
        unknowns=unknowns,
        coefficients=coefficients,
        rhs=rhs_tuple,
    )


def _linear_system_parameters(
    coefficients: tuple[tuple[Tree, ...], ...],
    rhs: tuple[Tree, ...],
) -> tuple[str, ...]:
    names: set[str] = set()

    for row in coefficients:
        for tree in row:
            names.update(symbol_names(tree))

    for tree in rhs:
        names.update(symbol_names(tree))

    return tuple(sorted(names))
