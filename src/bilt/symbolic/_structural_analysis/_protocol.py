from __future__ import annotations

from collections.abc import Callable, Iterable

from bilt.symbolic._structural_analysis._types import Case, StructuralCandidate, StructuralPlan

StructuralBackend = Callable[
    [tuple[StructuralCandidate, ...], Case],
    Iterable[StructuralPlan],
]


__all__ = ["StructuralBackend"]
