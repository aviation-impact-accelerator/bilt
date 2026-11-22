"""Experimental scalar equation audit using Bilt's normal evaluation plans.

This deliberately does not add the checked equations as solve constraints.
"""

from __future__ import annotations

import dataclasses
import math
import typing

from bilt._backend._python import PYTHON_BACKEND
from bilt.core._equation import BasicEquation
from bilt.core._errors import BiltAnalysisError
from bilt.symbolic._atoms import ADD, INV, MUL, NEG, POW, FloatType, IntType, SymbolType
from bilt.units import Quantity

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    from bilt._backend import Backend
    from bilt.core._system._implementation import Implementation
    from bilt.symbolic._tree import Tree


@dataclasses.dataclass(frozen=True)
class EquationCheck:
    """Residual and provenance of one original equation."""

    name: str
    status: typing.Literal["satisfied", "violated", "unavailable"]
    residual: float | None
    independent: bool
    reason: str | None = None


@dataclasses.dataclass(frozen=True)
class AuditReport:
    """Evaluation output, additional values and checks."""

    output: object
    values: dict[str, object]
    checks: tuple[EquationCheck, ...]


def _evaluate_expr(tree: Tree, values: Mapping[str, object]) -> object:  # noqa: PLR0911
    """Evaluate the supported scalar arithmetic subset of a Bilt expression tree."""
    if tree.is_leaf:
        if tree.head.atom_type is SymbolType:
            return values[str(tree.head.value)]
        if tree.head.atom_type in (IntType, FloatType):
            return tree.head.value
        message = f"Unsupported literal {tree.head!r}."
        raise ValueError(message)

    args = tuple(_evaluate_expr(child, values) for child in tree.args)
    if tree.head is ADD:
        return sum(args)
    if tree.head is MUL:
        result: object = args[0]
        for value in args[1:]:
            result = result * value  # type: ignore[operator]
        return result
    if tree.head is NEG:
        return -args[0]  # type: ignore[operator]
    if tree.head is INV:
        return 1 / args[0]  # type: ignore[operator]
    if tree.head is POW:
        return args[0] ** args[1]  # type: ignore[operator]
    message = f"Unsupported expression operator {tree.head!r}."
    raise ValueError(message)


def _residual(left: object, right: object) -> tuple[float, float]:
    """Return residual and comparison scale in a common unit, for scalar values."""
    if isinstance(left, Quantity) and isinstance(right, Quantity):
        right = right.convert_to(left.unit)
        return float(left.value - right.value), max(abs(float(left.value)), abs(float(right.value)))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left - right), max(abs(float(left)), abs(float(right)))
    message = "Only scalar numbers or compatible scalar quantities are supported."
    raise ValueError(message)


def audit_equations(
    implementation: Implementation,
    inputs: Mapping[str, object],
    output: str,
    *,
    observations: Mapping[str, object] | None = None,
    backend: Backend = PYTHON_BACKEND,
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> AuditReport:  # noqa: C901, PLR0913
    """Evaluate one output and check all direct equations in the implementation.

    Missing values are requested through ordinary Bilt evaluation, reusing
    previously computed intermediates. Independent observations affect checking
    only, never the solution. This scalar-only prototype skips child equations,
    transforms, conditionals, tensors and unsupported expression operators.
    """
    if atol < 0 or rtol < 0:
        message = "Tolerances must be nonnegative."
        raise ValueError(message)
    observed = dict(observations or {})
    overlap = set(observed) & set(inputs)
    if overlap:
        message = f"Observations duplicate solver inputs: {sorted(overlap)}."
        raise ValueError(message)

    values = dict(inputs)
    values.update(implementation._evaluate_with_intermediates(dict(inputs), output, backend))  # noqa: SLF001
    result = values[output]
    checks: list[EquationCheck] = []

    for constituent in implementation._components.values():  # noqa: SLF001
        if not isinstance(constituent, BasicEquation):
            continue
        name = constituent.__bilt_name__
        required = set(constituent.__bilt_parameters__)
        missing = sorted(required - values.keys() - observed.keys())
        error: str | None = None
        for port in missing:
            try:
                values.update(
                    implementation._evaluate_with_intermediates(  # noqa: SLF001
                        dict(inputs), port, backend, cached={key: val for key, val in values.items() if key not in inputs}
                    )
                )
            except BiltAnalysisError as exc:
                error = f"Cannot calculate {port}: {exc}"
                break

        present = values | observed
        independent = required <= (inputs.keys() | observed.keys())
        if not required <= present.keys():
            checks.append(EquationCheck(name, "unavailable", None, independent, error or "Missing variables."))
            continue

        tree = constituent.__bilt_equation_tree__
        if not tree.is_equality:
            checks.append(EquationCheck(name, "unavailable", None, independent, "Not a scalar equality."))
            continue
        try:
            residual, scale = _residual(_evaluate_expr(tree.left, present), _evaluate_expr(tree.right, present))
            if not math.isfinite(residual) or not math.isfinite(scale):
                checks.append(EquationCheck(name, "unavailable", None, independent, "Non-finite residual."))
                continue
        except (ValueError, TypeError, KeyError, ZeroDivisionError) as exc:
            checks.append(EquationCheck(name, "unavailable", None, independent, str(exc)))
            continue
        satisfied = abs(residual) <= atol + rtol * scale
        checks.append(EquationCheck(name, "satisfied" if satisfied else "violated", residual, independent))

    return AuditReport(result, values | observed, tuple(checks))
