"""Experimental discovery of *candidate* linear invariants from model evaluations.

A numerical nullspace is evidence of a relationship, not a symbolic proof.
"""

import dataclasses
import math
import typing

import numpy as np

from bilt._backend._python import PYTHON_BACKEND
from bilt.core._errors import BiltAnalysisError
from bilt.units import Quantity

if typing.TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from bilt._backend import Backend
    from bilt.core._system._implementation import Implementation


@dataclasses.dataclass(frozen=True)
class LinearInvariant:
    """A candidate homogeneous relation between compatible model ports."""

    coefficients: dict[str, float]
    maximum_residual: float
    dimension: str
    status: typing.Literal["candidate"] = "candidate"


def _numeric(value: object) -> float:
    """Use coherent values, rather than mixing user-selected units."""
    if isinstance(value, Quantity):
        return float(value.convert_to(value.unit).value * value.unit.scale)  # type: ignore[attr-defined]
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"Cannot use {type(value).__name__} in scalar invariant discovery.")


def _port_groups(implementation: Implementation) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Automatically group quantity-typed ports by dimension.

    Unlike plain floats, declared physical dimensions allow a safe first filter.
    """
    groups: dict[str, list[str]] = {}
    for name, declaration in implementation._port_declarations.items():  # noqa: SLF001
        groups.setdefault(str(declaration.dimensionality), []).append(name)
    return tuple(
        (dimension, tuple(sorted(set(names))))
        for dimension, names in sorted(groups.items())
        if len(set(names)) >= 3
    )


def discover_linear_invariants(
    implementation: Implementation,
    scenarios: Sequence[Mapping[str, object]],
    *,
    backend: Backend = PYTHON_BACKEND,
    tolerance: float = 1e-9,
    maximum_terms: int = 5,
) -> tuple[LinearInvariant, ...]:
    """Sample system outputs and detect dimensionally compatible linear relations.

    Every candidate is derived from a numerical nullspace across the supplied
    scenarios. The result may depend on the sampled operating points and must
    not be described as a proven conservation law.

    The prototype supports scalar quantity ports only. Unsupported or
    uncomputable groups are omitted, rather than silently filled with zero.
    """
    if len(scenarios) < 4:
        raise ValueError("Supply at least four diverse operating points.")
    if tolerance <= 0 or maximum_terms < 2:
        raise ValueError("Tolerance must be positive and maximum_terms at least 2.")

    candidates: list[LinearInvariant] = []
    for dimension, ports in _port_groups(implementation):
        if len(scenarios) <= len(ports):
            # Need more independent observations than unknown coefficients.
            continue
        rows: list[list[float]] = []
        for scenario in scenarios:
            known = dict(scenario)
            failed = False
            for port in ports:
                if port in known:
                    continue
                try:
                    known.update(
                        implementation._evaluate_with_intermediates(  # noqa: SLF001
                            dict(scenario), port, backend, cached={k: v for k, v in known.items() if k not in scenario}
                        )
                    )
                except BiltAnalysisError:
                    failed = True
                    break
            if failed:
                break
            try:
                row = [_numeric(known[port]) for port in ports]
            except (ValueError, KeyError, AttributeError):
                failed = True
                break
            if not all(math.isfinite(value) for value in row):
                failed = True
                break
            rows.append(row)
        if failed:
            continue
        matrix = np.asarray(rows, dtype=float)
        scale = np.maximum(np.linalg.norm(matrix, axis=0), np.finfo(float).tiny)
        normalised = matrix / scale
        _left, singular, vt = np.linalg.svd(normalised, full_matrices=True)
        rank = int(np.sum(singular > tolerance * max(singular[0], 1.0)))
        if rank == len(ports):
            continue

        for vector in vt[rank:]:
            coefficients = vector / scale
            largest = max(abs(value) for value in coefficients)
            if largest == 0:
                continue
            coefficients = coefficients / largest
            coefficients[np.abs(coefficients) < tolerance * 10] = 0
            active = np.flatnonzero(coefficients)
            if len(active) < 2 or len(active) > maximum_terms:
                continue
            # Exclude undersampled/constant apparent relationships.
            if any(np.ptp(matrix[:, i]) < tolerance * max(np.max(np.abs(matrix[:, i])), 1.0) for i in active):
                continue
            coefficients /= abs(coefficients[active[0]])
            if coefficients[active[0]] < 0:
                coefficients *= -1
            residual = matrix @ coefficients
            magnitude = np.max(np.abs(matrix) @ np.abs(coefficients))
            if np.max(np.abs(residual)) > tolerance * max(float(magnitude), 1.0):
                continue
            candidates.append(
                LinearInvariant(
                    coefficients={ports[i]: float(coefficients[i]) for i in active},
                    maximum_residual=float(np.max(np.abs(residual))),
                    dimension=dimension,
                )
            )
    return tuple(candidates)
